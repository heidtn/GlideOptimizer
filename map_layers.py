import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import elevation
from sim_params import SimParams
from scipy.interpolate import RegularGridInterpolator

"""

TODOs: 
- need to validate indexing on the elevation array between lat/lon and x/y, essentially mapping from x/y and lat/lon
- RegularGridInterpolator is stupid slow for some reason, went fom 120 it/s to 20 it/s, probably can make this faster

"""


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
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.grid = RegularGridInterpolator((y_coords, x_coords), self.elevation_array)

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
        #return self.grid(np.array((y, x)))
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sim_params import default_params
    arr = get_elevation_data(default_params)
    plt.imshow(arr)
    plt.show()
