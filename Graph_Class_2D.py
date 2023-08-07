import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Graph_2D_edges:
    """
    Creates a graph from a list of edges.
    """
    def __init__(self, edges):
        """
        Initializes parameters, list of edges and vertices, for graph class. 
        Edges passed in and copies and symmetric edges are removed. 
        Copied vertices also removed.
        
        Parameters
        ----------
        edges : tuple list,
            Edges for your particular graph.
        
        Returns
        -------
        None.
        """
        #to remove duplicate edges
        self.edges = set(edges)
        self.edges = list(self.edges)
        self.vert = []
        self.connections = []
        self.vectors = []
        self.norms = []
        self.vert_degree = []
        self.neighbors_mat = []
        self.neighbors_lists = []
        for entry in self.edges:
            self.vert.append(entry[0])
            self.vert.append(entry[1])
        self.vert = set(self.vert)
        self.vert = list(self.vert)
        #to remove symmetric data
        for k in range(0,len(self.edges)):
            iterbreak = 0
            for i in range(0,len(self.edges)):
                iterlen = len(self.edges)-1
                for j in range(0,iterlen):
                    if self.edges[i][0] == self.edges[j][1]:
                        if self.edges[i][1] == self.edges[j][0]:
                            self.edges.remove(self.edges[i])
                            iterbreak = 1
                            break
                if iterbreak == 1:
                    break
    
    def get_neighbors(self):
        """
        Uses sklearn to find the nearest neighbors of each vertex.

        Returns
        -------
        nxn matrix of floats:
            entries correspond to the index of the vertex in the vertices
            array ordered by nearest neighbors.

        """
        knn = NearestNeighbors(n_neighbors=len(self.vert))
        knn.fit(self.vert)
        distance_mat, self.neighbors_mat = knn.kneighbors(self.vert)
        return self.neighbors_mat
    
    def get_neighbors_reduced(self,num_neigh):
        """
        Uses sklearn to find the nearest neighbors of each vertex.

        Returns
        -------
        nxn matrix of floats:
            entries correspond to the index of the vertex in the vertices
            array ordered by nearest neighbors.

        """
        knn = NearestNeighbors(n_neighbors=num_neigh+1)
        knn.fit(self.vert)
        distance_mat, self.neighbors_mat = knn.kneighbors(self.vert)
        return self.neighbors_mat
    
    def get_connection(self, vertex, edges, neighbors_list):
        """
        Builds list of connections for a specific vertex ordered by nearest neighbors.
        
        Parameters
        ----------
        vertex : tuple of floats
            Specific vertex to generate from.
        edges : list of tuples of vertices.
            Connections between vertices.
        neighbors_list : list of ints
            A list of indices of nearest neighbored vertices to specific vertex.

        Returns
        -------
        ordered_connection : list of tuples of vertices
            A list of each connection to specific vertex ordered by nearest neighbor.

        """
        connection = []
        for i in range(0,len(edges)):
            if vertex == edges[i][0]:
                connection.append((vertex, edges[i][1]))
            if vertex == edges[i][1]:
                connection.append((vertex,edges[i][0]))
        connection = set(connection)
        connection = list(connection)
        ordered_connection = [connection[j] for k in range(1,len(neighbors_list))\
                              for j in range(0,len(connection))\
                                  if connection[j][1] == self.vert[neighbors_list[k]]]
        return ordered_connection
    
    def get_neighbors_list(self, connection, neighbors_mat_seg):
        """
        Builds list of nearest neighbored indices by removing all 
        nonconnected indices from neighbors matrix.

        Parameters
        ----------
        connection : list of tuples of vertices.
            List of connections for a specific vertex.
        neighbors_mat_seg : list of ints
            Segment of neighbors matrix for specific vertex.

        Returns
        -------
        neighbors_list : list of ints
            Nearest neighbors indices for specific vertex with connected
            vertices only.

        """
        neighbors_list = [i for j in range(0,len(connection)) \
                          for i in range(0,len(self.vert)) \
                              if connection[j][1] == self.vert[i]]
        return neighbors_list
    
    def get_neighbors_lists(self, connections,num_neigh=None):
        """
        Builds list of lists of nearest neighbors based on connected vertices.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        neighbors_lists: list of lists of ints
            list of each nearest neighbor based on connections for each vertex in list of vertices.

        """
        if num_neigh == None:
            self.neighbors_mat = Graph_2D_edges.get_neighbors(self)
        if num_neigh != None:
            self.neighbors_mat = Graph_2D_edges.get_neighbors_reduced(self,num_neigh)
        self.neighbors_lists = [Graph_2D_edges.get_neighbors_list(self, connections[i], self.neighbors_mat[i]) \
                                for i in range(0,len(self.vert))]
        return self.neighbors_lists
    
    def get_vert_degree(self, connections):
        """
        Finds degree, number of connections, of each vertex.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        vert_degree: list of ints
            Degree of each vertex in list of vertices.

        """
        self.vert_degree = [len(connection) for connection in connections]
        return self.vert_degree
        
    def get_connections(self, edges,num_neigh=None):
        """
        Builds list of connections for a particular vertex, removes duplicates
        and stores vertex degree.
        
        Parameters
        ----------
        vertex : tuple of ints,
            a particular vertex passed to build list of those connections.
        edges : list of tuple of tuple of ints,
            edges particular to what connections you want to build.

        Returns
        -------
        connection : list of tuple of tuple of ints,
            Builds list of connections for particular point 
            and removes duplicates.

        """
        if num_neigh == None:
            self.neighbors_mat = Graph_2D_edges.get_neighbors(self)
        if num_neigh != None:
            self.neighbors_mat = Graph_2D_edges.get_neighbors_reduced(self,num_neigh)
        self.connections = [Graph_2D_edges.get_connection(self, self.vert[i], edges, self.neighbors_mat[i]) \
                            for i in range(0,len(self.vert))]
        return self.connections
    
    def get_vector(self, connection):
        """
        Computes a list of vectors for each connection to a specific vertex.

        Parameters
        ----------
        connection : list of tuples of vertices.
            Connections from one vertex to another.

        Returns
        -------
        vector: list of list of two floats representing a vector.
            list of vectors between a specific vertex and its connections.

        """
        vector = [[(connection[i][1][0] - connection[i][0][0]),(connection[i][1][1] - connection[i][0][1])] \
                  for i in range(0,len(connection))]
        return vector
    
    def get_vectors(self, connections):
        """
        Builds list of list of vectors for each vertex in list of vertices.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        vectors: list of list of list of two floats representing a vector.
            list of list of vectors for each specific vertex and its connections.

        """
        self.vectors = [Graph_2D_edges.get_vector(self, connections[i]) \
                        for i in range(0,len(connections))]
        return self.vectors
    
    def get_norms(self):
        """
        Finds length of each vector in vectors.

        Returns
        -------
        norms: list of list of floats
            list of list of lengths of each vector in vectors specific to each vertex.

        """
        for i in range(0,len(self.vectors)):
            norm = []
            for j in range(0,len(self.vectors[i])):
                norm.append(np.linalg.norm(self.vectors[i][j]))
            self.norms.append(norm)
        return self.norms
    
    def plot(self):
        """
        Plots the graph on an xy axis.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        for entry in self.edges:
            x_list = []
            y_list = []
            x_list.append(entry[0][0])
            y_list.append(entry[0][1])
            x_list.append(entry[1][0])
            y_list.append(entry[1][1])
            plt.plot(x_list,y_list)
            plt.show()
            
    def reduce_edges(self,connections):
        """
        Reduces edges after building reduced graph to ensure no symmetric or duplicates.

        Parameters
        ----------
        connections : list of lists of tuple of tuple of ints,
            Builds list of connections for particular point 
            and removes duplicates.

        Returns
        -------
        edges : list of tuple of tuple of ints,
            edges for reduced graph with no dupicates or symmetric edges.

        """
        self.edges = []
        for i in range(len(connections)):
            for j in range(len(connections[i])):
                self.edges.append(connections[i][j])
        #to remove symmetric data
        for k in range(0,len(self.edges)):
            iterbreak = 0
            for i in range(0,len(self.edges)):
                iterlen = len(self.edges)-1
                for j in range(0,iterlen):
                    if self.edges[i][0] == self.edges[j][1]:
                        if self.edges[i][1] == self.edges[j][0]:
                            self.edges.remove(self.edges[i])
                            iterbreak = 1
                            break
                    if i != j:
                        if self.edges[i][0] == self.edges[j][0]:
                            if self.edges[i][1] == self.edges[j][1]:
                                self.edges.remove(self.edges[i])
                                iterbreak = 1
                                break
                if iterbreak == 1:
                    break
        return self.edges
    
    def complete_graph(self):
        """
        Completes and graphs the complete graph, and returns a list of
        the complete edges.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        com_edges : list,
            Complete edges for graph.
        """
        connections = itertools.combinations(self.vert,2)
        com_vert = []
        com_edges = []
        for entry in connections:
            com_edges.append(entry)
            com_vert.append(entry[0])
            com_vert.append(entry[1])
        return com_edges
    def save_graph(self):
        """
        Saves the graph information as a text file named graph_data.txt.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        graph_file = open("./graph_data.txt", "w")
        graph_file.write('[')
        for entry in self.edges[:len(self.edges)-1]:
            edge = entry
            edge = ''.join(str(edge))
            graph_file.write(edge + ',')
        graph_file.write(str(self.edges[-1]))
        graph_file.write(']')
        graph_file.close()
        
class Graph_2D_vertices:
    """
    Creates a graph from a list of edges.
    """
    def __init__(self, vertices):
        """
        Initializes parameters, list of edges and vertices, for graph class. 
        Vertices passed in and copies are removed. 
        
        Parameters
        ----------
        vertices : tuple list,
            Vertices for your particular data set turned into graph.
        
        Returns
        -------
        None.
        """
        self.vert = vertices
        self.vert = set(self.vert)
        self.vert = list(self.vert)
        self.edges = []
        self.connections = []
        self.vectors = []
        self.norms = []
        self.vert_degree = []
        self.neighbors_mat = []
        self.neighbors_lists = []
    
    def get_neighbors(self):
        """
        Uses sklearn to find the nearest neighbors of each vertex.

        Returns
        -------
        nxn matrix of floats:
            entries correspond to the index of the vertex in the vertices
            array ordered by nearest neighbors.

        """
        knn = NearestNeighbors(n_neighbors=len(self.vert))
        knn.fit(self.vert)
        distance_mat, self.neighbors_mat = knn.kneighbors(self.vert)
        return self.neighbors_mat
    
    def get_neighbors_reduced(self,num_neigh):
        """
        Uses sklearn to find the nearest neighbors of each vertex.

        Returns
        -------
        nxn matrix of floats:
            entries correspond to the index of the vertex in the vertices
            array ordered by nearest neighbors.

        """
        knn = NearestNeighbors(n_neighbors=num_neigh+1)
        knn.fit(self.vert)
        distance_mat, self.neighbors_mat = knn.kneighbors(self.vert)
        return self.neighbors_mat
    
    def get_connection(self, vertex, edges, neighbors_list):
        """
        Builds list of connections for a specific vertex ordered by nearest neighbors.
        
        Parameters
        ----------
        vertex : tuple of floats
            Specific vertex to generate from.
        edges : list of tuples of vertices.
            Connections between vertices.
        neighbors_list : list of ints
            A list of indices of nearest neighbored vertices to specific vertex.

        Returns
        -------
        ordered_connection : list of tuples of vertices
            A list of each connection to specific vertex ordered by nearest neighbor.

        """
        connection = []
        for i in range(0,len(edges)):
            if vertex == edges[i][0]:
                connection.append((vertex, edges[i][1]))
            if vertex == edges[i][1]:
                connection.append((vertex,edges[i][0]))
        connection = set(connection)
        connection = list(connection)
        ordered_connection = [connection[j] for k in range(1,len(neighbors_list))\
                              for j in range(0,len(connection))\
                                  if connection[j][1] == self.vert[neighbors_list[k]]]
        return ordered_connection
    
    def get_neighbors_list(self, connection, neighbors_mat_seg):
        """
        Builds list of nearest neighbored indices by removing all 
        nonconnected indices from neighbors matrix.

        Parameters
        ----------
        connection : list of tuples of vertices.
            List of connections for a specific vertex.
        neighbors_mat_seg : list of ints
            Segment of neighbors matrix for specific vertex.

        Returns
        -------
        neighbors_list : list of ints
            Nearest neighbors indices for specific vertex with connected
            vertices only.

        """
        neighbors_list = [i for j in range(0,len(connection)) \
                          for i in range(0,len(self.vert)) \
                              if connection[j][1] == self.vert[i]]
        return neighbors_list
    
    def get_neighbors_lists(self, connections,num_neigh=None):
        """
        Builds list of lists of nearest neighbors based on connected vertices.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        neighbors_lists: list of lists of ints
            list of each nearest neighbor based on connections for each vertex in list of vertices.

        """
        if num_neigh == None:
            self.neighbors_mat = Graph_2D_vertices.get_neighbors(self)
        if num_neigh != None:
            self.neighbors_mat = Graph_2D_vertices.get_neighbors_reduced(self,num_neigh)
        self.neighbors_list = [Graph_2D_vertices.get_neighbors_list(self, connections[i], self.neighbors_mat[i]) \
                               for i in range(0,len(self.vert))]
        return self.neighbors_lists
    
    def get_vert_degree(self, connections):
        """
        Finds degree, number of connections, of each vertex.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        vert_degree: list of ints
            Degree of each vertex in list of vertices.

        """
        self.vert_degree = [len(connection) for connection in connections]
        return self.vert_degree
        
    def get_connections(self, edges,num_neigh=None):
        """
        Builds list of connections for a particular vertex, removes duplicates
        and stores vertex degree.
        
        Parameters
        ----------
        vertex : tuple of ints,
            a particular vertex passed to build list of those connections.
        edges : list of tuple of tuple of ints,
            edges particular to what connections you want to build.

        Returns
        -------
        connection : list of tuple of tuple of ints,
            Builds list of connections for particular point 
            and removes duplicates.

        """
        if num_neigh == None:
            self.neighbors_mat = Graph_2D_vertices.get_neighbors(self)
        if num_neigh != None:
            self.neighbors_mat = Graph_2D_vertices.get_neighbors_reduced(self,num_neigh)
        self.connections = [Graph_2D_vertices.get_connection(self, self.vert[i], edges, self.neighbors_mat[i]) \
                                for i in range(0,len(self.vert))]
        return self.connections
    
    def reduce_edges(self,connections):
        """
        Reduces edges after building reduced graph to ensure no symmetric or duplicates.

        Parameters
        ----------
        connections : list of lists of tuple of tuple of ints,
            Builds list of connections for particular point 
            and removes duplicates.

        Returns
        -------
        edges : list of tuple of tuple of ints,
            edges for reduced graph with no dupicates or symmetric edges.

        """
        self.edges = []
        for i in range(len(connections)):
            for j in range(len(connections[i])):
                self.edges.append(connections[i][j])
        #to remove symmetric data
        for k in range(0,len(self.edges)):
            iterbreak = 0
            for i in range(0,len(self.edges)):
                iterlen = len(self.edges)-1
                for j in range(0,iterlen):
                    if self.edges[i][0] == self.edges[j][1]:
                        if self.edges[i][1] == self.edges[j][0]:
                            self.edges.remove(self.edges[i])
                            iterbreak = 1
                            break
                    if i != j:
                        if self.edges[i][0] == self.edges[j][0]:
                            if self.edges[i][1] == self.edges[j][1]:
                                self.edges.remove(self.edges[i])
                                iterbreak = 1
                                break
                if iterbreak == 1:
                    break
        return self.edges
        
    
    def get_vector(self, connection):
        """
        Computes a list of vectors for each connection to a specific vertex.

        Parameters
        ----------
        connection : list of tuples of vertices.
            Connections from one vertex to another.

        Returns
        -------
        vector: list of list of two floats representing a vector.
            list of vectors between a specific vertex and its connections.

        """
        vector = [[(connection[i][1][0] - connection[i][0][0]),(connection[i][1][1] - connection[i][0][1])] \
                  for i in range(0,len(connection))]
        return vector
    
    def get_vectors(self, connections):
        """
        Builds list of list of vectors for each vertex in list of vertices.

        Parameters
        ----------
        connections : list of lists of tuples of vertices
            List of connected vertices for each vertex in list of vertices.

        Returns
        -------
        vectors: list of list of list of two floats representing a vector.
            list of list of vectors for each specific vertex and its connections.

        """
        self.vectors = [Graph_2D_vertices.get_vector(self, connections[i]) \
                        for i in range(0,len(connections))]
        return self.vectors
    
    def get_norms(self):
        """
        Finds length of each vector in vectors.

        Returns
        -------
        norms: list of list of floats
            list of list of lengths of each vector in vectors specific to each vertex.

        """
        for i in range(0,len(self.vectors)):
            norm = []
            for j in range(0,len(self.vectors[i])):
                norm.append(np.linalg.norm(self.vectors[i][j]))
            self.norms.append(norm)
        return self.norms
    
    def plot(self):
        """
        Plots the graph on an xy axis.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        for entry in self.edges:
            x_list = []
            y_list = []
            x_list.append(entry[0][0])
            y_list.append(entry[0][1])
            x_list.append(entry[1][0])
            y_list.append(entry[1][1])
            plt.plot(x_list,y_list)
            plt.show()
    
    def complete_graph(self):
        """
        Completes and graphs the complete graph, and returns a list of
        the complete edges.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        com_edges : list,
            Complete edges for graph.
        """
        connections = itertools.combinations(self.vert,2)
        com_vert = []
        com_edges = []
        for entry in connections:
            com_edges.append(entry)
            com_vert.append(entry[0])
            com_vert.append(entry[1])
        return com_edges
    def save_graph(self):
        """
        Saves the graph information as a text file named graph_data.txt.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        graph_file = open("./graph_data.txt", "w")
        graph_file.write('[')
        for entry in self.edges[:len(self.edges)-1]:
            edge = entry
            edge = ''.join(str(edge))
            graph_file.write(edge + ',')
        graph_file.write(str(self.edges[-1]))
        graph_file.write(']')
        graph_file.close()

def load_data(file_name):
    """
    Turns a txt file of edges into edges that can be turned into a graph.
    Returns edges as list.

    Parameters
    ----------
    file_name : txt file, 
        Contains graph edges.
    
    Returns
    -------
    graph_list : list,
        Edges from data file.
    """
    my_file = open(file_name, "r")
    data = my_file.read()
    graph_list = eval(data)
    my_file.close()
    return graph_list

def build_edge_graph_2D(edges):
    """
    Builds graph from edges.

    Parameters
    ----------
    edges : List of tuples of tuples of floats.
        Describes edges between vertices.

    Returns
    -------
    x_graph : Graph Class 2D
        Initializes all parameters used in graph class.

    """
    x_graph = Graph_2D_edges(edges)
    x_graph.connections = x_graph.get_connections(x_graph.edges)
    x_graph.vert_degree = x_graph.get_vert_degree(x_graph.connections)
    x_graph.vectors = x_graph.get_vectors(x_graph.connections)
    x_graph.neighbors_mat = x_graph.get_neighbors()
    x_graph.neighbors_list = x_graph.get_neighbors_lists(x_graph.connections)
    x_graph.norms = x_graph.get_norms()
    x_graph.plot()
    return x_graph

def build_vert_complete_graph_2D(vertices):
    """
    Builds a complete graph from list of vertices.

    Parameters
    ----------
    vertices : list of tuples of floats
        Describe points in 2D space.

    Returns
    -------
    x_graph : Graph Class from Verts
        Graph with all attributes of Graph Class 2D but with different initial perameters.

    """
    x_graph = Graph_2D_vertices(vertices)
    x_graph.edges = x_graph.complete_graph()
    x_graph.connections = x_graph.get_connections(x_graph.edges)
    x_graph.vert_degree = x_graph.get_vert_degree(x_graph.connections)
    x_graph.vectors = x_graph.get_vectors(x_graph.connections)
    x_graph.neighbors_mat = x_graph.get_neighbors()
    x_graph.neighbors_list = x_graph.get_neighbors_lists(x_graph.connections)
    x_graph.norms = x_graph.get_norms()
    x_graph.plot()
    return x_graph

def build_edge_complete_graph_2D(edges):
    """
    Builds complete graph from edges.

    Parameters
    ----------
    edges : List of tuples of tuples of floats.
        Describes edges between vertices.

    Returns
    -------
    x_graph : Graph Class 2D
        Initializes all parameters used in graph class.

    """
    x_graph = Graph_2D_edges(edges)
    x_graph.edges = x_graph.complete_graph()
    x_graph.connections = x_graph.get_connections(x_graph.edges)
    x_graph.vert_degree = x_graph.get_vert_degree(x_graph.connections)
    x_graph.vectors = x_graph.get_vectors(x_graph.connections)
    x_graph.neighbors_mat = x_graph.get_neighbors()
    x_graph.neighbors_list = x_graph.get_neighbors_lists(x_graph.connections)
    x_graph.norms = x_graph.get_norms()
    x_graph.plot()
    return x_graph

def build_edge_reduced_graph_2D(edges,num_neigh):
    """
    Builds reduced complete graph from edges.

    Parameters
    ----------
    edges : List of tuples of tuples of floats.
        Describes edges between vertices.
        
    num_neigh : int,
        Number of neighbors used for building connections.

    Returns
    -------
    x_graph : Graph Class 2D
        Initializes all parameters used in graph class.

    """
    x_graph = Graph_2D_edges(edges)
    x_graph.edges = x_graph.complete_graph()
    x_graph.connections = x_graph.get_connections(x_graph.edges,num_neigh=num_neigh)
    x_graph.edges = x_graph.reduce_edges(x_graph.connections)
    x_graph.vert_degree = x_graph.get_vert_degree(x_graph.connections)
    x_graph.vectors = x_graph.get_vectors(x_graph.connections)
    x_graph.neighbors_mat = x_graph.get_neighbors_reduced(num_neigh)
    x_graph.neighbors_list = x_graph.get_neighbors_lists(x_graph.connections,num_neigh=num_neigh)
    x_graph.norms = x_graph.get_norms()
    x_graph.plot()
    return x_graph

def build_vert_reduced_graph_2D(vertices,num_neigh):
    """
    Builds reduced complete graph from vertices.

    Parameters
    ----------
    vertices : list of tuples of floats
        Describe points in 2D space.
        
    num_neigh : int,
        Number of neighbors used for building connections.

    Returns
    -------
    x_graph : Graph Class 2D
        Initializes all parameters used in graph class.

    """
    x_graph = Graph_2D_vertices(vertices)
    x_graph.edges = x_graph.complete_graph()
    x_graph.connections = x_graph.get_connections(x_graph.edges,num_neigh=num_neigh)
    x_graph.edges = x_graph.reduce_edges(x_graph.connections)
    x_graph.vert_degree = x_graph.get_vert_degree(x_graph.connections)
    x_graph.vectors = x_graph.get_vectors(x_graph.connections)
    x_graph.neighbors_mat = x_graph.get_neighbors_reduced(num_neigh)
    x_graph.neighbors_list = x_graph.get_neighbors_lists(x_graph.connections,num_neigh=num_neigh)
    x_graph.norms = x_graph.get_norms()
    x_graph.plot()
    return x_graph
