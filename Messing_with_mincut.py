import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def min_cut_max_flow(graph, ws_mat, lam, lam_mask):
    """
    Uses many contraints and linprog to generate the minimum cut for our graph.
    
    Parameters
    ----------
    graph : Graph Class
        Graph generated from any graph class.
        
    ws_mat : array of floats
        Weights ordered specific to each point then by nearest neighbor.
        
    lam : float
        Scaling parameter for multiscale flat norm.
        
    lam_mask : list of ints
        List of ones and zeros describing vertex connected to source or sink.
    
    Returns
    -------
    min_cut : linprog.scipy.optimize returns,
    res : OptimizeResult,
        A scipy.optimize.OptimizeResult consisting of the fields below. 
        Note that the return types of the fields may depend on whether 
        the optimization was successful, 
        therefore it is recommended to check 
        OptimizeResult.status before relying on the other fields:

            x : 1-D array of floats,
                The values of the decision variables that minimizes 
                the objective function while satisfying the constraints.

            fun : float,
                The optimal value of the objective function n @ x.

            slack : 1-D array,
                The (nominally positive) values of the slack variables,
                w_s - num_pix @ x.

            con : 1-D array,
                The (nominally zero) residuals of the equality constraints,
                lam_mask - lam @ x.

            success : bool,
                True when the algorithm succeeds in finding an optimal solution.

            status : int,
                An integer representing the exit status of the algorithm.
                    0 : Optimization terminated successfully.
                    1 : Iteration limit reached.
                    2 : Problem appears to be infeasible.
                    3 : Problem appears to be unbounded.
                    4 : Numerical difficulties encountered.

            nit : int
                The total number of iterations performed in all phases.

            message : str
                A string descriptor of the exit status of the algorithm.
    """
    n = len(graph.vert)
    nE = 2*len(graph.edges) #adding times two to allow for nonsymmetric weights
    nj = graph.vert_degree
    lam_caps = np.array([lam for i in range(2*n)])
    lam_cap = lam_mask*lam_caps
    w_row = ws_mat.ravel()
    c = np.concatenate((np.zeros(n),lam_cap,w_row))
    
    #Number of variables
    n_var = 3*n + nE

    #Constraints matrix
    const_1 = np.zeros((nE,n_var))
    m = 0
    for i in range(0,n):
        for j in range(0,nj[i]):
            const_a = np.eye(1, n_var, k=3*n + m) - np.eye(1, n_var, k=i) + np.eye(1, n_var, k=graph.neighbors_lists[i][j])
            const_1[m] = const_a
            m += 1
    #z_o, z_p, d_so, d_sp, d_to, d_tp, d_op, d_po
    #accounts for d_sp + z_p >= 1 and d_so + z_o >= 1. 
    const_2 = np.eye(n, n_var) + np.eye(n, n_var, k=n)

    #accounts for d_tp - z_p >= 0 and d_to - z_o >= 0
    const_3 = np.eye(n, n_var, k=2*n) - np.eye(n, n_var)

    Constraints = -np.concatenate((const_1, const_2, const_3)) #Multiply by -1 to switch inequality sign
    
    #Inequality side of constraints
    rhs_1 = np.zeros(nE) 
    rhs_2 = np.ones(n)
    rhs_3 = np.zeros(n)
    constraints_rhs = -np.concatenate((rhs_1, rhs_2, rhs_3))
    bounds = [(0,1) for i in range(n_var)]
    min_cut = sc.optimize.linprog(c=c, A_ub=Constraints, b_ub=constraints_rhs, bounds=bounds)
    return min_cut

def graph_2D_min_cut(graph, labels,source,sink, x):
    """
    Graphs min cut in 3D, visual representation only.

    Parameters
    ----------
    graph : Graph Class
        Graph generated from graph class.
    labels : list of ints
        List of ones and zeros indicating connection between source and sink.
    source : tuple of floats
        Point of source.
    sink : tuple of floats
        Point of sink.

    Raises
    ------
    Exception
        If there are an incorrect number of labels, exception raised.
        Each vertex should have two labels, one for source and one for sink.

    Returns
    -------
    None.

    """
    z_list = []
    s_labels = []
    t_labels = []
    for i in range(int(len(labels)/2)):
        s_labels.append(labels[i])
    for j in range(int(len(labels)/2),len(labels)):
        t_labels.append(labels[j])
    if len(s_labels) != len(t_labels):
        raise Exception("Incorrect number of labels.")
    x_list = []
    y_list = []
    k=0
    for i in range(0,len(s_labels)):
        if s_labels[i] == 1:
            x_list.append(graph.vert[i][0])
            y_list.append(graph.vert[i][1])
            z_list.append(0)
            x_list.append(source[0])
            y_list.append(source[1])
            z_list.append(1)
            k+=1
        if t_labels[i] == 1:
            x_list.append(graph.vert[i][0])
            y_list.append(graph.vert[i][1])
            z_list.append(0)
            x_list.append(sink[0])
            y_list.append(sink[1])
            z_list.append(-1)
            k+=1
    x_con_list = []
    y_con_list = []
    z_con_list = []
    #n = len(graph.edges) - k
    n = len(graph.vert) #- len(s_labels)
    # for i in range(n):
    #     if x[i] == 1:
    #         x_con_list.append(graph.edges[i][0][0])
    #         y_con_list.append(graph.edges[i][0][1])
    #         z_con_list.append(0)
    #         x_con_list.append(graph.edges[i][1][0])
    #         y_con_list.append(graph.edges[i][1][1])
    #         z_con_list.append(0)
    k = 3*n - 1
    for i in range(len(graph.connections)):
        for j in range(len(graph.connections[i])):
            if x[k] == 1:
                x_con_list.append(graph.connections[i][j][0][0])
                y_con_list.append(graph.connections[i][j][0][1])
                z_con_list.append(0)
                x_con_list.append(graph.connections[i][j][1][0])
                y_con_list.append(graph.connections[i][j][1][1])
                z_con_list.append(0)
            k+=1
    print("\nInfo for Graph, k, xlen:\n", k, len(x))
    print(graph.connections)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot(x_list,y_list,z_list, zdir='z', c= 'blue')
    ax.scatter(x_list, y_list, z_list, c='red')
    #ax.plot(x_con_list,y_con_list,z_con_list, c='black')
    plt.show()
    
def plot_new_thing(graph, labels,source,sink, x):
    n = len(graph.vert) #- len(s_labels)
    k = 3*n - 1
    for i in range(len(graph.connections)):
        for j in range(len(graph.connections[i])):
            if x[k] == 1:
                x_con_list = []
                y_con_list = []
                z_con_list = []
                x_con_list.append(graph.connections[i][j][0][0])
                y_con_list.append(graph.connections[i][j][0][1])
                z_con_list.append(0)
                x_con_list.append(graph.connections[i][j][1][0])
                y_con_list.append(graph.connections[i][j][1][1])
                z_con_list.append(0)
                plt.plot(x_con_list,y_con_list, c='black')
                plt.show()
            k+=1
    print("\nInfo for Graph, k, xlen:\n", k, len(x))
    print(graph.connections)
