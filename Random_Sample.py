from time import time
import numpy as np

class Random_Sample_2D:
    """
    Creates a 2D random sample of data depending on the bounds and number
    of points imputed by the user.
    """
    def __init__(self, num, max_x, min_x, max_y, min_y):
        """
        Initializes class parameters.
        
        Parameters
        ----------
        num : int, number of points desired for sample.
        max_x : int, 
            Highest number you want the x coordinate point to be.
        min_x : int, 
            Lowest number you want the x coordinate to be.
        max_y : int, 
            Highest number you want the y coordinate to be.
        min_y : int, 
            Lowest number you want the y coordinate to be.
        
        Returns
        -------
        None.
        """
        self.num = num
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.x_list = []
        self.y_list = []
        self.coord_arr = []

    def generate_random_sample(self, max_bound, min_bound, rand):
        """
        Generate samples with `n` within the interval (`min`, `max`). 
        This random sample is for desired random testing purposes.
     
        Parameters
        ----------
        max : int, 
            Highest number you want the coordinate point to be.
        min : int, 
            Lowest number you want the coordinate point to be.
        rand : int, 
            Iterator chosen to ensure x doesn't equal y.
        
        Returns
        -------
        array of random floats within the specific bounds.
        """
        rng = np.random.default_rng(seed=int(time())+rand)
        diff = np.abs(max_bound - min_bound)
        return rng.random(self.num) * diff + min_bound

    def generate_vert(self):
        """
        Builds an array of random coordinate points.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        coord_arr : array,
            x and y coordinate array of floats randomly generated
            within specified bounds.
        """
        rand=0
        self.x_list = self.generate_random_sample(self.max_x, self.min_x, rand)
        rand = 1
        self.y_list = self.generate_random_sample(self.max_y, self.min_y, rand)
        for i in range(len(self.x_list)):
            entry = (round(self.x_list[i],2),round(self.y_list[i],2))
            self.coord_arr.append(entry)
        return self.coord_arr
    
class Random_Sample_3D:
    """
    Creates a 3D random sample of data depending on the bounds and number
    of points imputed by the user.
    """
    def __init__(self, num, max_x, min_x, max_y, min_y, max_z, min_z):
        """
        Initializes class parameters.
        
        Parameters
        ----------
        num : int, number of points desired for sample.
        max_x : int, 
            Highest number you want the x coordinate point to be.
        min_x : int, 
            Lowest number you want the x coordinate to be.
        max_y : int, 
            Highest number you want the y coordinate to be.
        min_y : int, 
            Lowest number you want the y coordinate to be.
        
        Returns
        -------
        None.
        """
        self.num = num
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.x_list = []
        self.y_list = []
        self.max_z = max_z
        self.min_z = min_z
        self.z_list = []
        self.coord_arr = []

    def generate_random_sample(self, max_bound, min_bound, rand):
        """
        Generate samples with `n` within the interval (`min`, `max`). 
        This random sample is for desired random testing purposes.
     
        Parameters
        ----------
        max : int, 
            Highest number you want the coordinate point to be.
        min : int, 
            Lowest number you want the coordinate point to be.
        rand : int, 
            Iterator chosen to ensure x doesn't equal y.
        
        Returns
        -------
        array of random floats within the specific bounds.
        """
        rng = np.random.default_rng(seed=int(time())+rand)
        diff = np.abs(max_bound - min_bound)
        #return rng.random(self.num) * diff + min_bound
        return rng.random(self.num) * diff + min_bound

    def generate_vert(self):
        """
        Builds an array of random coordinate points.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        coord_arr : array,
            x and y coordinate array of floats randomly generated
            within specified bounds.
        """
        rand = 0
        self.x_list = self.generate_random_sample(self.max_x, self.min_x, rand)
        rand = 1
        self.y_list = self.generate_random_sample(self.max_y, self.min_y, rand)
        rand = 2
        self.z_list = self.generate_random_sample(self.max_z, self.min_z, rand)
        for i in range(len(self.x_list)):
            entry = (round(self.x_list[i],2),round(self.y_list[i],2),round(self.z_list[i],2))
            self.coord_arr.append(entry)
        # self.coord_arr = np.stack([self.x_list, self.y_list, self.z_list]).T
        # self.coord_arr = list(self.coord_arr)
        return self.coord_arr
    
def generate_random_verts_2D(num, max_x, min_x, max_y, min_y):
    RS = Random_Sample_2D(num, max_x, min_x, max_y, min_y)
    coord_arr = RS.generate_vert()
    return coord_arr
    
def generate_random_verts_3D(num, max_x, min_x, max_y, min_y, max_z, min_z):
    RS = Random_Sample_3D(num, max_x, min_x, max_y, min_y, max_z, min_z)
    coord_arr = RS.generate_vert()
    return coord_arr
