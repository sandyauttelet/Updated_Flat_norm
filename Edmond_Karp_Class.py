import numpy as np
import collections

def adj_list_to_matrix(adj_list,weights,graph):
    n = len(graph.vert)
    adj_matrix = np.nan*np.ones((n,n))
    np.fill_diagonal(adj_matrix,0)
    k=0
    for i in range(n):
        for j in range(len(adj_list[i])):
            for q in range(len(weights)):
                if weights[k] == 0:
                    k=k+1
                if weights[k] != 0:
                    break
            adj_matrix[i,adj_list[i][j]+2] = weights[k]
            k+=1
    return adj_matrix

def add_source_sink_connections(labels,graph,source,sink,vor_area_list):
    s_labels = []
    t_labels = []
    new_graph_vert = []
    for i in range(int(len(labels)/2)):
        s_labels.append(labels[i])
    for j in range(int(len(graph.vert)),len(labels)):
        t_labels.append(labels[j])
    if int(len(labels)/2) != len(graph.vert) or len(s_labels) != len(t_labels):
        raise Exception("Inncorrect number of labels.")
    old_neighbors_lists = graph.neighbors_lists.copy()
    new_neighbors_lists = []
    s_neighbors_list = []
    for i in range(len(s_labels)):
        if s_labels[i] == 1 and t_labels[i] == 1:
            raise Exception("Vertex cannot be connected to source and sink.")
        if s_labels[i] == 0 and t_labels[i] == 0:
            raise Exception("One or more vertices are not connected to source or sink.")
        if s_labels[i] == 1:
            graph.edges.append((source,graph.vert[i]))
            s_neighbors_list.append(i)
    new_neighbors_lists.append(s_neighbors_list)
    t_neighbors_list = []
    for i in range(len(t_labels)):
        if t_labels[i] == 1:
            graph.edges.append((sink,graph.vert[i]))
            t_neighbors_list.append(i)
    new_neighbors_lists.append(t_neighbors_list)
    for i in range(len(old_neighbors_lists)):
        new_neighbors_lists.append(old_neighbors_lists[i])
    graph.neighbors_lists = new_neighbors_lists
    new_graph_vert.append(source)
    new_graph_vert.append(sink)
    for i in range(0,len(graph.vert)):
        new_graph_vert.append(graph.vert[i])
    graph.vert = new_graph_vert
    for i in range(len(s_labels)):
        s_labels[i] = s_labels[i]*vor_area_list[i]
    for i in range(len(t_labels)):
        t_labels[i] = t_labels[i]*vor_area_list[i]
    return graph, s_labels, t_labels

def rebuild_weights_st(weights,scale,s_labels,t_labels):
    new_weights = []
    for i in range(len(s_labels)):
        #if s_labels[i] != 0:
        new_weights.append(scale*s_labels[i])
    for i in range(len(t_labels)):
        #if t_labels[i] != 0:
        new_weights.append(scale*t_labels[i])
    for i in range(len(weights)):
        new_weights.append(weights[i])
    return new_weights

class Edmond_Karp:
    """
    This class represents a directed graph using
    adjacency matrix representation.
    """

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.row = len(graph)

    def bfs(self, s, t, parent):
        """
        Returns true if there is a path from
        source 's' to sink 't' in residual graph.
        Also fills parent[] to store the path.
        """

        # Mark all the vertices as not visited
        visited = [False] * self.row

        # Create a queue for BFS
        queue = collections.deque()

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS loop
        while queue:
            u = queue.popleft()

            # Get all adjacent vertices of the dequeued vertex u
            # If an adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if (visited[ind] == False) and (val > 0):
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return visited[t]

    # Returns the maximum flow from s to t in the given graph
    def edmonds_karp(self, source, sink):
        # This array is filled by BFS and to store path
        parent = [-1] * self.row

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.bfs(source, sink, parent):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow
