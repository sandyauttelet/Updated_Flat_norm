import numpy as np
import matplotlib.pyplot as plt
import Random_Sample as RS

class Voronoi_Region_2D:
    def __init__(self,point):
        self.point = np.array(point)
        self.area = 0.0
        self.samples = []
    def __str__(self):
        return "point: " + str(self.point) + " | area: "\
    + "{:0.2f}".format(self.area)

def get_bounding_box(points):
    """finds the smallest box containing the points and its area"""
    x_coords, y_coords = zip(*points)
    x_range = (np.min(x_coords),np.max(x_coords))
    y_range = (np.min(y_coords),np.max(y_coords))
    area = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))
    return x_range, y_range, area

def add_sample_region(point,regions,total_area,N,plot=True):
    distances = [np.linalg.norm(point-region.point) for region in regions]
    closest_region = regions[np.argmin(distances)]
    closest_region.area += total_area/N
    if plot:
        closest_region.samples.append(point)
    return
    
def get_sample(x_range,y_range,N):
    sample_points_x = np.random.uniform(*x_range,N)
    sample_points_y = np.random.uniform(*y_range,N)
    return zip(sample_points_x,sample_points_y)
    
def find_voronoi_polygon_areas(points,N=10000, plot=True):
    """finds the area of voronoi polygons for a set of points in R^2, takes in
    a list of points [(float,float)] and a number of samples to draw"""
    regions = [Voronoi_Region_2D(point) for point in points]
    x_range, y_range, total_area = get_bounding_box(points)
    sample_points = get_sample(x_range, y_range, N)
    for point in sample_points:
        add_sample_region(point,regions,total_area,N,plot)
    if plot:
        for region in regions:
            plt.scatter(*zip(*region.samples))
            plt.annotate("{:0.2f}".format(region.area), region.point,\
                         xytext=(5,-3), textcoords='offset points',\
                             color='black')
        plt.scatter(*zip(*points),c=[[0,0,0]])  
    return regions

def find_voronoi_area_list(solution):
    vor_area_list = []
    for region in solution:
        vor_area_list.append(region.area)
    return vor_area_list
        

# =============================================================================
# generate a regular grid from (-1,-1) to (1,1)
# =============================================================================
#points = [(float(x),float(y)) for x in range(-1,2) for y in range(-1,2)]

# =============================================================================
# transform regular grid into a non-regular grid
# =============================================================================
# theta = np.pi/4
# A = np.array([[np.cos(theta), -np.sin(theta)],
#               [np.sin(theta),np.cos(theta)]])
# points = np.array(points)
# points = [A @ point for point in points]

# solution = find_voronoi_polygon_areas(points)
