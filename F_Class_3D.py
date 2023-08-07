import numpy as np
import cubepy as cp
import matplotlib.pyplot as plt

def get_C1s(norms):
    """
    Calculates constant 1 from flat norm function.

    Parameters
    ----------
    norms : list of list of floats
        length of each vector specific to each vertex and its connections.

    Returns
    -------
    C1s : list of list of floats
        Constant 1 for each vector in vectors for each vertex.

    """
    C1s = [(4*np.pi*norms[i][j]**2)/3 for i in range(len(norms)) \
            for j in range(len(norms[i]))]
    return C1s

def get_C2s(norms):
    """
    Calculates constant 2 from flat norm function.

    Parameters
    ----------
    norms : list of list of floats
        length of each vector specific to each vertex and its connections.

    Returns
    -------
    C2s : list of list of floats
        Constant 2 for each vector in vectors for each vertex.

    """
    C2s = [(2*np.pi*norms[i][j]) for i in range(len(norms)) \
            for j in range(len(norms[i]))]
    return C2s

def get_C3(vector_pair,num):
    def sphere_integrand_explicit(theta,phi):
        return np.sin(theta)*(np.abs(np.sin(theta)*np.cos(phi)*vector_pair[0][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[0][1]+np.cos(theta)*vector_pair[0][2])\
            *np.abs(np.sin(theta)*np.cos(phi)*vector_pair[1][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[1][1]+np.cos(theta)*vector_pair[1][2]))
    low = [0.0,0.0]
    high = [np.pi,2*np.pi]
    value, error = cp.integrate(sphere_integrand_explicit, low, high,itermax=num)
    return value       

def make_sym_matrix(n,vals,diag):
    """
    Builds a symmetric matrix.

    Parameters
    ----------
    n : int
        Size of matrix to be nxn.
    vals : list of floats
        Values to be input into nondiagonal matrix components.
        Ordered left to right, then up down.
    diag : list of floats
        Diagonal of matrix.

    Returns
    -------
    m : matrix
        Result is an nxn symmetric matrix.

    """
    m = np.zeros((n,n))
    xs,ys = np.triu_indices(n,k=1)
    m[xs,ys] = vals
    m[ys,xs] = vals
    m[ np.diag_indices(n) ] = diag
    return m

def get_C3s_sym(vectors, C1s,num):
    """
    Builds list of constant 3 matrices, all symmetric, for each vertex.

    Parameters
    ----------
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class
    C1s : list of list of floats
        Constant 1 for each vector in vectors for each vertex.
    num : int
        Number of times iteration is used to approximate integrand.

    Returns
    -------
    C3s_highd : list of matrices of floats
        A list of matrices for each vertex in graph containing each constant 3 value for every vector pair.

    """
    C3s_highd = []
    for k in range(0,len(vectors)):
        n = len(vectors[k])
        C3s = [get_C3((vectors[k][i], vectors[k][j]),num) for i in range(n) \
               for j in range(i+1,n)]
        C3s_sym = list(make_sym_matrix(n,C3s, C1s[k]))
        C3s_highd.append(C3s_sym)
    return C3s_highd

def dF_dwa(w_alpha, ws, us, C1, C2, C3s, alpha):
    """
    Finds exact value for derivative of flat norm function with respect to specific vertex.

    Parameters
    ----------
    w_alpha : float
        Specific edge weight
    ws : list of floats
        All other nonspecific edge weights.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1 : float
        Constant 1 for specific edge.
    C2 : float
        Constant 2 for specific edge.
    C3s : list of floats
        Constant 3 for specific edge specific to a vertex and its other connections.
    alpha : int
        Iterator of specific edge.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    float
        flat norm funciton value.

    """

    # Sum term
    summation = 0
    for i in range(len(us)):
        summation = summation
        sums = ws[i]*C3s[i]
        summation = sums + summation
        
    return 2 * (summation - C2)

def jacob(guess, vectors, C1s, C2s, C3s):
    """
    Finds jacobian vector for flat norm functional for specific vertex.

    Parameters
    ----------
    guess : list of floats
        Initial guess of weights to minimize flat norm functional.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1s : list of floats
        List of constant 1 value for each vector.
    C2s : list of floats
        List of constant 2 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    jacobian: array of floats
        List of derivatives of flat norm functional specific to a vertex.

    """
    jacobian = []
    for i in range(len(vectors)):
        w_alpha = guess[i]
        derivative = dF_dwa(w_alpha, guess, vectors, C1s[i], C2s[i], C3s[i], i)
        jacobian.append(derivative)
    return np.array(jacobian)

    
def d2F_dwa_dwb(w_alpha, w_beta, us, C1, C3, alpha, beta):
    """
    Compute the 2nd partial derivative of F, first with w_alpha then
    with w_beta.
    
    Parameters
    ----------
    w_alpha : float, 
        Weight between point alpha and its nearest neghbour.
    w_beta : float, 
        Weight between point alpha and point beta.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1 : float
        Constant 1 value for specific alpha vector.
    C3 : float
        Constant 3 value for specific alpha vector and beta pair vector.
    alpha : int,
        Used for iteration through all points of graph.
    beta : int,
        Used for iteration through all points of graph
    num : int
        Number of time iteration is used to approximate constant 3s integrand.
        
    Returns
    -------
    float
        Value of second derivative of flat norm function and the specific alpha weight.
        
    """
        
    if alpha == beta:
        return 2 *C1
    
    return 2*C3

def hess(guess, vectors, C1s, C3s):
    """
    Builds hessian matrix for all weights specific to each vertex.

    Parameters
    ----------
    guess : list of floats
        Initial guess of weights to minimize flat norm functional.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1s : list of floats
        List of constant 1 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    hessian: array of arrays of floats
        Symmetric matrix of values of second derivative of flat norm function with respect to alpha weight and beta weight.

    """
    hessian = []
    for j in range(0,len(vectors)):
        hessian_row = []
        for k in range(0,len(vectors)):
            w_alpha = guess[j]
            w_beta = guess[k]
            double_derivative = d2F_dwa_dwb(w_alpha, w_beta, vectors, C1s[j], C3s[j][k], j, k) 
            hessian_row.append(double_derivative)
        hessian.append(hessian_row)
    return np.array(hessian)
    



def newton_iter(wn, vectors, C1s, C2s, C3s):
    """
    Iterates through the jacobian and hessian of flat norm to minimize weights.

    Parameters
    ----------
    wn : list of floats
        Initial guess of weights to minimize flat norm functional.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1s : list of floats
        List of constant 1 value for each vector.
    C2s : list of floats
        List of constant 2 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    newton_dir : float
        Approximate direction of newton's method.

    """
    jacobian = jacob(wn, vectors, C1s, C2s, C3s)
    hessian = hess(wn, vectors, C1s, C3s)
    newton_dir = np.linalg.solve(hessian,-jacobian)
    return newton_dir

def newton(w0, vectors, C1s, C2s, C3s, eps=1e-8, max_it=int(10000)):
    """
    Uses iterative methods to minimize function.

    Parameters
    ----------
    w0 : list of floats
        Initial guess of weights to minimize flat norm functional.
    vectors : list of lists of arrays of two floats
        Vectors list generated from graph class for a specific vertex.
    C1s : list of floats
        List of constant 1 value for each vector.
    C2s : list of floats
        List of constant 2 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.
    eps : float, optional
        Tolerance of zero approximation. The default is 10e-9.
    max_it : int, optional
        Max iteration to break loop if zero not found. The default is int(10000).

    Raises
    ------
    Exception
        If zero not found, function can't be minimized.

    Returns
    -------
    w_n : array of floats
        Weights list for each connection to specific vertex.

    """
    w_n = w0
    for i in range(max_it):
        w_pre = w_n
        w_n = w_pre + newton_iter(w_pre, vectors, C1s, C2s, C3s)
        if np.linalg.norm(w_pre - w_n) < eps:
            return w_n

    raise Exception("Zero not found in newton")
    
def get_weights_avg(graph, C1s, C2s, C3s):
    """
    Finds average weight between each edge in graph.

    Parameters
    ----------
    graph : Graph Class
        Generated from graph class.
    C1s : list of floats
        List of constant 1 value for each vector.
    C2s : list of floats
        List of constant 2 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    list of list of floats
        Average weights between each edge of the graph.

    """
    ws_mat = []
    for i in range(0,len(graph.vectors)):
        guess = np.ones(len(graph.vectors[i]))
        ws_mat.append(newton(guess, graph.vectors[i], C1s[i], C2s[i], C3s[i]))

    avg_ws = []
    for i in range(0,len(graph.vert)):
        for j in range(0,len(graph.connections[i])):
            for k in range(i,len(graph.vert)):
                if graph.vert[k] == graph.vert[graph.neighbors_lists[i][j]]:
                    for l in range(0,len(graph.connections[k])):
                         if graph.connections[k][l][1] == graph.vert[i]:
                             avg_ws.append(0.5*(ws_mat[i][j]+ws_mat[k][l]))
    return np.array(avg_ws)

def get_weights_full(graph, C1s, C2s, C3s):
    """
    Finds weight between each connection in graph.

    Parameters
    ----------
    graph : Graph Class
        Generated from graph class.
    C1s : list of floats
        List of constant 1 value for each vector.
    C2s : list of floats
        List of constant 2 value for each vector.
    C3s : list of list of floats
        Symmetric matrix of constant 3 values for each vector and all its pairs.
    num : int
        number of time iteration is used to approximate constant 3s integrand.

    Returns
    -------
    array of floats
        List of all weights ordered by specific vertex and then nearest neighbor.
        Decomposed into a single array.

    """
    ws_mat = []
    ws_matrix = []
    for i in range(0,len(graph.vectors)):
        guess = np.ones(len(graph.vectors[i]))
        ws_mat.append(newton(guess, graph.vectors[i], C1s[i], C2s[i], C3s[i]))
        
    for j in range(0,len(ws_mat)):
         for entry in ws_mat[j]:
             ws_matrix.append(entry)
             
    return np.array(ws_matrix)

def plot_graph_with_weights(graph, full_weights):
    """
    Plots a group of subplots with each vertex in its own plot and shows all connections.

    Parameters
    ----------
    graph : Graph Class
        Generated from graph class.
    full_weights : array of floats
        List of all weights ordered by specific vertex and then nearest neighbor.
        Decomposed into a single array.

    Returns
    -------
    None.

    """
    full_x_list = []
    full_y_list = []
    full_z_list = []
    fig = plt.figure(2)
    fig.suptitle("Weights for Edges of Specific Vertices", fontsize=16)
    for i in range(len(graph.connections)+1):
        if i < len(graph.connections):
            x_list = []
            y_list = []
            z_list = []
            for j in range(len(graph.connections[i])):
                x_list.append(graph.connections[i][j][0][0])
                y_list.append(graph.connections[i][j][0][1])
                z_list.append(graph.connections[i][j][0][2])
                x_list.append(graph.connections[i][j][1][0])
                y_list.append(graph.connections[i][j][1][1])
                z_list.append(graph.connections[i][j][1][2])
                full_x_list.append(graph.connections[i][j][0][0])
                full_y_list.append(graph.connections[i][j][0][1])
                full_z_list.append(graph.connections[i][j][0][2])
                full_x_list.append(graph.connections[i][j][1][0])
                full_y_list.append(graph.connections[i][j][1][1])
                full_z_list.append(graph.connections[i][j][1][2])
        scale_x = 1
        scale_y = 1
        col_num = 3
        row_num = len(graph.vert)//3
        if len(graph.vert) % 3 == 0:
            row_num = row_num
        if len(graph.vert) % 3 != 0:
            row_num = row_num + 1
        ax1 = plt.subplot(row_num, col_num, (i+1,i+1,i+1), projection='3d')
        title = ("Vertex", i)
        #ax.set_title(title)
        #ax.set_title('Weights of Edges for Vertex', i)
        plt.scatter(graph.vert[i][0], graph.vert[i][1], graph.vert[i][2])
        plt.plot(x_list,y_list,z_list)
        plt.show()
