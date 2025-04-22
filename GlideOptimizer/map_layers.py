import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import elevation
from GlideOptimizer.sim_params import SimParams
from scipy.interpolate import RegularGridInterpolator

"""

TODOs: 
- need to validate indexing on the elevation array between lat/lon and x/y, essentially mapping from x/y and lat/lon
- RegularGridInterpolator is stupid slow for some reason, went fom 120 it/s to 20 it/s, probably can make this faster
- move to an abstract map class
- make the map setup DRY
"""


class FlatMap:
    """
    Just a flat map with some size
    """
    def __init__(self, params: SimParams, x_size, y_size):
        x = int(x_size/params.map_droplet_resolution)
        y = int(y_size/params.map_droplet_resolution)
        self.elevation_array = np.zeros((x, y))
        self.params = params

        self.x_size = x_size
        self.y_size = y_size

        self.dx = self.x_size / self.elevation_array.shape[0]
        self.dy = self.y_size / self.elevation_array.shape[1]

        nrows, ncols = self.elevation_array.shape
        x_coords = np.arange(ncols) * self.dx
        y_coords = np.arange(nrows) * self.dy
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.grid = RegularGridInterpolator((y_coords, x_coords), self.elevation_array)

        droplet_array_x = int(round(self.x_size / params.map_droplet_resolution))
        droplet_array_y = int(round(self.y_size / params.map_droplet_resolution))
        self.droplet_array = np.zeros((droplet_array_x, droplet_array_y)) 

    def get_elevation_at(self, x, y) -> float:
        return 0.
    
    def add_droplet_at(self, x, y):
        try:
            x_idx = int((x / self.x_size) * self.droplet_array.shape[0])
            y_idx = int((y / self.y_size) * self.droplet_array.shape[1])
            self.droplet_array[y_idx, x_idx] += 1
        except IndexError:
            return None

    def clear_droplet_array(self):
        self.droplet_array = 0

    def get_droplet_variances(self):
        return _calculate_droplet_variance(self, self.params)
    
    def get_droplet_cep(self, percent):
        return _calculate_droplet_cep(self, self.params, percent)
    
    def get_droplet_histogram(self, bins=30):
        distances = _get_droplet_distances(self, self.params)
        histogram_data = np.histogram(distances, bins)
        return histogram_data
    
    def get_droplet_centroid_distance(self):
        center =  _calculate_droplet_centroid(self, self.params)
        return np.linalg.norm(center)


class Map:
    """
    We assume latitude correlates to y and longitude to x. Handles the elevation and other map overlays.
    """
    def __init__(self, params: SimParams):
        self.elevation_array, self.resolution = generate_elevation_array(params)

        # TODO(Heidt) this is ok for a rough estimate, but makes some major assumptions, move to proper UTM reprojection 
        self.x_size = (params.lon_max - params.lon_min) * 40000.0e3 * np.cos((params.lat_max + params.lat_min) * np.pi / 360.0) / 360.0
        self.y_size = (params.lat_max - params.lat_min) * 40000.0e3 / 360.0

        self.dx = self.x_size / self.elevation_array.shape[0]
        self.dy = self.y_size / self.elevation_array.shape[1]

        nrows, ncols = self.elevation_array.shape
        x_coords = np.arange(ncols) * self.dx
        y_coords = np.arange(nrows) * self.dy
        #xx, yy = np.meshgrid(x_coords, y_coords)
        #self.grid = RegularGridInterpolator((x_coords, y_coords), self.elevation_array)

        droplet_array_x = int(round(self.x_size / params.map_droplet_resolution))
        droplet_array_y = int(round(self.y_size / params.map_droplet_resolution))
        self.droplet_array = np.zeros((droplet_array_x, droplet_array_y)) 

    def get_elevation_at(self, x, y) -> float:
        # TODO(Heidt) this should actually interpolate between points
        # TODO(Heidt) return None or raise error if out of bounds
        # TODO(Heidt) for this func and add_droplet, we scale by shape index 0,1, but index by 1,0 why?
        nrows, ncols = self.elevation_array.shape
        x_idx = int((x / self.x_size) * self.elevation_array.shape[0])
        y_idx = int((y / self.y_size) * self.elevation_array.shape[1])
        try:
            return self.elevation_array[y_idx, x_idx]
        except IndexError:
            return None
    
    def add_droplet_at(self, x, y):
        x_idx = int((x / self.x_size) * self.droplet_array.shape[0])
        y_idx = int((y / self.y_size) * self.droplet_array.shape[1])
        self.droplet_array[y_idx, x_idx] += 1

    def clear_droplet_array(self):
        self.droplet_array = 0

        

# TODO(Heidt) find a more elegant solution than passing the dem path string
def generate_elevation_array(
    params: SimParams,
    dem_path: str = "/home/nathan/Projects/GlideOptimizer/data/temp_dem.tif",
) -> tuple[np.ndarray, float]:
    """
    Generate a numpy array of elevation data for the specified geographic bounds
    at the given resolution (in meters) using the elevation and rasterio libraries.

    Parameters:
      dem_path (str): Path to store the temporary DEM file.

    Returns:
      numpy.ndarray: 2D array with elevation data.
    """
    # Download and clip the DEM for the bounding box.
    # The bounds are specified as (west, south, east, north)
    bounds = (params.lon_min, params.lat_min, params.lon_max, params.lat_max)
    # Using SRTM1 which typically provides ~30m resolution data.
    resolution = 30.0
    elevation.clip(bounds=bounds, output=dem_path, product="SRTM3")
    with rasterio.open(dem_path) as src:
        # Determine a target CRS using UTM based on the center of the bounding box.
        dem_data = src.read(1).astype("float64")
        print(dem_data.shape)

    # Clean up the temporary DEM file.
    os.remove(dem_path)
    return (dem_data, resolution)


def get_elevation_data(params: SimParams):
    arr = generate_elevation_array(params)
    return arr

from numba import njit

@njit
def _count_droplets(x_size, y_size, resolution, drops):
    distance_array = []
    for x in range(drops.shape[0]):
        for y in range(drops.shape[1]):
            x_position = x * resolution - x_size / 2.0
            y_position = y * resolution - y_size / 2.0
            for i in range(int(drops[x, y])):
                distance_array.append((x_position, y_position))
    return distance_array

def _get_droplet_distances(map: FlatMap, params: SimParams):
    drops = map.droplet_array
    distance_array = _count_droplets(map.x_size, map.y_size, params.map_droplet_resolution, drops)
    mean = np.mean(np.array(distance_array), axis=0)
    distances = np.linalg.norm(distance_array -  mean, axis=1)
    return distances

def _calculate_droplet_variance(map: FlatMap, params: SimParams):
    """ 
    calculates variance of droplets from the center of the map in both axes

    TODO(Heidt) might want to do covariance instead if we ever care
    """
    distances = _get_droplet_distances(map, params)
    return np.var(distances)

def _calculate_droplet_cep(map: FlatMap, params: SimParams, percent):
    """ 
    calculates circular error probably
    """
    distances = _get_droplet_distances(map, params)
    return np.percentile(distances, percent)

def _calculate_droplet_centroid(map: FlatMap, params: SimParams):
    #TODO(Heidt) This is copied from _get_droplet_distances, unify
    drops = map.droplet_array
    distance_array = _count_droplets(map.x_size, map.y_size, params.map_droplet_resolution, drops)
    mean = np.mean(np.array(distance_array), axis=0)
    return mean
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sim_params import test_params
    arr = get_elevation_data(test_params)
    plt.imshow(arr)
    plt.show()
