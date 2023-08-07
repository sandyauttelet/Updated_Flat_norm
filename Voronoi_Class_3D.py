import numpy as np
import matplotlib.pyplot as plt
import Random_Sample as RS
import time

class Voronoi_Region_3D:
    def __init__(self,point):
        self.point = np.array(point)
        self.area = 0.0
        self.volume = 0.0
        self.samples = []
    def __str__(self):
        return "point: " + str(self.point) + " | volume: "\
    + "{:0.2f}".format(self.volume)

def get_bounding_box(points):
    """finds the smallest box containing the points and its area"""
    x_coords, y_coords, z_coords = zip(*points)
    x_range = (np.min(x_coords),np.max(x_coords))
    y_range = (np.min(y_coords),np.max(y_coords))
    z_range = (np.min(z_coords),np.max(z_coords))
    volume = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))*np.linalg.norm(np.diff(z_range))
    return x_range, y_range, z_range, volume

def add_sample_region(point,regions,total_volume,N,plot=True):
    distances = [np.linalg.norm(point-region.point) for region in regions]
    closest_region = regions[np.argmin(distances)]
    closest_region.volume += total_volume/N
    if plot:
        closest_region.samples.append(point)
    return
    
def get_sample(x_range,y_range,z_range,N):
    sample_points_x = np.random.uniform(*x_range,N)
    sample_points_y = np.random.uniform(*y_range,N)
    sample_points_z = np.random.uniform(*z_range,N)
    return zip(sample_points_x,sample_points_y,sample_points_z)
    
def find_voronoi_polygon_volumes(points,N=100000, plot=True):
    """finds the area of voronoi polygons for a set of points in R^2, takes in
    a list of points [(float,float)] and a number of samples to draw"""
    regions = [Voronoi_Region_3D(point) for point in points]
    x_range, y_range, z_range, total_volume = get_bounding_box(points)
    sample_points = get_sample(x_range, y_range,z_range, N)
    for point in sample_points:
        add_sample_region(point,regions,total_volume,N,plot)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for region in regions:
            ax.scatter(*zip(*region.samples))
            ax.text(region.point[0],region.point[1],region.point[2],"{:0.2f}".format(region.volume))
            ax.scatter(*zip(*points),c=[[0,0,0]])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
    return regions

def find_voronoi_volume_list(solution):
    vor_volume_list = []
    for region in solution:
        vor_volume_list.append(region.volume)
    return vor_volume_list

# =============================================================================
# generate a regular grid from (-1,-1) to (1,1)
# =============================================================================
#points = [(float(x),float(y)) for x in range(-1,2) for y in range(-1,2)]
# start_time_2D = time.time()
# random_verts_3D = RS.generate_random_verts_3D(100, 5, -5, 10, -10, -15, 15)
# solution = find_voronoi_polygon_volumes(random_verts_3D)
# print("--- %s seconds 2D ---" % (time.time() - start_time_2D))

# =============================================================================
# transform regular grid into a non-regular grid
# =============================================================================
# theta = np.pi/4
# A = np.array([[np.cos(theta), -np.sin(theta), -np.cos(theta)],
#               [np.sin(theta),np.cos(theta), np.sin(theta)],
#               [-np.cos(theta),np.sin(theta), np.cos(theta)]])
# points = np.array(points)
# points = [A @ point for point in points]

#solution = find_voronoi_polygon_volumes(points)
