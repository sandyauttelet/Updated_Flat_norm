import numpy as np
import Graph_Class_2D as GC2
import Graph_Class_3D as GC3
import Messing_with_F as FC2
import F_Class_3D as FC3
import Messing_with_mincut as MMC
import Edmond_Karp_Class as EKC
import Voronoi_Class_2D as VC2
import Voronoi_Class_3D as VC3
import Random_Sample as RS
import time


"""2D Graph"""

#User Inputs for Execution
random_edges_2D = [((0,1),(4,0)),((4,0),(2,3)),((5,7),(4,2)),((2,3),(8,9)),((13,4),(8,9)),((2,3),(13,4))]
random_verts_2D = RS.generate_random_verts_2D(100, 5, -5, 10, -10)
sphere = np.linspace(0,2*np.pi,100)
sphere_verts_2D = [(np.cos(sphere_i),np.sin(sphere_i)) for sphere_i in sphere]
num_2D = 100
num_neigh_2D = 5
labels_rand_2D = [1,1,0,1,0,1,0,\
                  0,0,1,0,1,0,1]
lam_2D = 0.75
source_2D = (3,12)
sink_2D = (22,4)

#Build Graph
start_time_2D = time.time() #times entire process
#graph_2D = GC2.build_vert_reduced_graph_2D(sphere_verts_2D,num_neigh_2D)
graph_2D = GC2.build_edge_reduced_graph_2D(random_edges_2D,num_neigh_2D)
print("\nVerts\n",graph_2D.vert)
print("\nConnections\n",graph_2D.connections)
print("\nEdges\n",graph_2D.edges)
print("\nNeighbors\n",graph_2D.neighbors_lists)

# #Find Weights
C1s_2D = FC2.get_C1s(graph_2D.norms)
print(C1s_2D)
# C2s_2D = FC2.get_C2s(graph_2D.norms)
# C3s_2D = FC2.get_C3s_sym(graph_2D.vectors,C1s_2D,num_2D)
# C3s_error = FC2.get_C3s_errors(graph_2D.vectors, num_2D)
# weights_2D = FC2.get_weights_avg(graph_2D, C1s_2D, C2s_2D, C3s_2D, num_2D)
# full_weights_2D = FC2.get_weights_full(graph_2D, C1s_2D, C2s_2D, C3s_2D, num_2D)
# FC2.plot_graph_with_weights(graph_2D, full_weights_2D)
# voronoi_2D = VC2.find_voronoi_polygon_areas(graph_2D.vert)
# vor_area_list_2D = VC2.find_voronoi_area_list(voronoi_2D)

# # #Finds Min Cut Based on Jared and I's Previous Work and Prints Min Cut Class
# # min_cut_2D = MMC.min_cut_max_flow(graph_2D, full_weights_2D, lam_2D, labels_rand_2D)
# # MMC.graph_2D_min_cut(graph_2D, labels_rand_2D,source_2D, sink_2D, min_cut_2D.x)
# # MMC.plot_new_thing(graph_2D, labels_rand_2D,source_2D, sink_2D, min_cut_2D.x)
# # print(min_cut_2D.x)

# #Recalibrates Graph and Weights to Account for Source and Sink Nodes
# old_neighbors_lists = graph_2D.neighbors_lists.copy()
# graph_st_2D, s_labels_2D, t_labels_2D = EKC.add_source_sink_connections(labels_rand_2D,graph_2D,source_2D,sink_2D,vor_area_list_2D)
# full_weights_st_2D = EKC.rebuild_weights_st(full_weights_2D,lam_2D,s_labels_2D,t_labels_2D)
# print(graph_st_2D.neighbors_lists)
# print(full_weights_st_2D)
# adj_matrix_2D = EKC.adj_list_to_matrix(graph_st_2D.neighbors_lists,full_weights_st_2D,graph_st_2D)
# print(adj_matrix_2D)

# #Executes Edmond Karp and Prints Max Flow
# source_loc_2D = 0
# sink_loc_2D = 1
# #parent_list = np.ones(len(graph_st_2D.vert))
# Ed_Karp_2D = EKC.Edmond_Karp(adj_matrix_2D)
# #BFS = Ed_Karp_2D.bfs(source_loc_2D,sink_loc_2D,parent_list)
# max_flow_2D = Ed_Karp_2D.edmonds_karp(source_loc_2D,sink_loc_2D)
# print("\nFlow:", max_flow_2D)
print("--- %s seconds 2D ---" % (time.time() - start_time_2D)) #prints time of execution

"""3D Graph"""

# #User Inputs for Execution
# random_edges_3D = [((1,2,4),(5,6,3)),((8,9,5),(11,12,2)),((8,9,5),(11,12,2)),((5,6,3),(1,2,4)),((8,9,5),(1,12,2)),((18,9.4,5.3),(1.7,12.8,20)),((78,91,50),(12,12.6,2.7))]
# random_verts_3D = RS.generate_random_verts_3D(5, 5, -5, 10, -10, 15, -15)
# num_3D = 100
# labels_rand_3D = [1,1,1,0,0,1,1,0,0\
#                   ,0,0,0,1,1,0,0,1,1]
# lam_3D = 0.75
# source_3D = (0,3,7)
# sink_3D = (4,-12,2)

# #Build Graph
# start_time_3D = time.time() #times entire process
# graph_3D = GC3.build_edge_complete_graph_3D(random_edges_3D)
# #print(graph_3D.norms)

# #Find Weights
# C1s_3D = FC3.get_C1s(graph_3D.norms)
# print(C1s_3D)
# C2s_3D = FC3.get_C2s(graph_3D.norms)
# C3s_3D = FC3.get_C3s_sym(graph_3D.vectors,C1s_3D,num_3D)
# print(C3s_3D)
# # weights_3D = FC3.get_weights_avg(graph_3D, C1s_3D, C2s_3D, C3s_3D)
# full_weights_3D = FC3.get_weights_full(graph_3D, C1s_3D, C2s_3D, C3s_3D)
# #FC3.plot_graph_with_weights(graph_3D, full_weights_3D)

# #Finds Min Cut Based on Jared and I's Previous Work and Prints Min Cut Class
# #min_cut_3D = MMC.min_cut_max_flow(graph_3D, weights_3D, lam_3D, labels_rand_3D)
# #print(min_cut_3D)

# #Recalibrates Graph and Weights to Account for Source and Sink Nodes
# old_neighbors_lists = graph_3D.neighbors_lists.copy()
# graph_st_3D, s_labels_3D, t_labels_3D = EKC.add_source_sink_connections(labels_rand_3D,graph_3D,source_3D,sink_3D)
# full_weights_st_3D = EKC.rebuild_weights_st(full_weights_3D,lam_3D,s_labels_3D,t_labels_3D)
# #adj_matrix_3D = EKC.adj_list_to_matrix(graph_st_3D.neighbors_lists,full_weights_st_3D)
# adj_matrix_3D = EKC.adj_list_to_matrix(old_neighbors_lists,full_weights_3D)
# print("\nNieghbors\n", old_neighbors_lists)
# print("\nFull Weights\n", full_weights_3D)
# print("\n Adjenceny Matrix\n", adj_matrix_3D)
# # #MMC.graph_3D_min_cut(graph_3D, labels_rand_3D,source_3D, sink_3D)

# # #Executes Edmond Karp and Prints Max Flow
# # source_loc_3D = 0
# # sink_loc_3D = 1
# # parent_list = np.ones(len(graph_st_3D.vert))
# # Ed_Karp_3D = EKC.Edmond_Karp(adj_matrix_3D)
# # BFS = Ed_Karp_3D.bfs(source_loc_3D,sink_loc_3D,parent_list)
# # max_flow_3D = Ed_Karp_3D.edmonds_karp(source_loc_3D,sink_loc_3D)
# # print("\nFlow:", max_flow_3D)
# print("--- %s seconds 3D ---" % (time.time() - start_time_3D)) #prints time of execution
