import numpy as np
import pyvista as pv
from rasterio.transform import from_origin
import map_layers
from particles import ParticleManager

from sim_params import default_params

class Visualizer:
    def __init__(self, map: map_layers.Map, particles: ParticleManager):
        self.map = map
        self.particles = particles
        #self.plotter = pv.Plotter(notebook=False, off_screen=True)
        self.plotter = pv.Plotter()
        plot_elevation_with_pyvista(self.plotter, map)
        self.particle_points = None
        #self.plotter.open_gif("test.gif")

    def create_points(self):
        self.particle_points = pv.PolyData(self.particles.droplet_positions)
        self.plotter.add_points(self.particle_points, render_points_as_spheres=True, point_size=100)
        self.plotter.show(auto_close=False)
        print("creating particles")

    def update(self):
        if not self.particle_points:
            if self.particles.droplet_positions:
                self.create_points()
            else:
                return
        self.particle_points.points = self.particles.droplet_positions
        self.plotter.update_coordinates(self.particles.droplet_positions, mesh=self.particle_points, render=True)
        #self.plotter.write_frame()
        self.plotter.render()

    def close(self):
        self.plotter.close()
        


def plot_elevation_with_pyvista(plotter: pv.Plotter, map: map_layers.Map):
    nrows, ncols = map.elevation_array.shape
    # For an affine transform from_origin(west, north, resolution, resolution),
    # the x coordinates (columns) are:
    x_coords = np.arange(ncols) * map.x_res
    # and the y coordinates (rows) are (note: y decreases as row index increases)
    y_coords = np.arange(nrows) * map.y_res

    # Create a meshgrid of x and y coordinates.
    xx, yy = np.meshgrid(x_coords, y_coords)
    # z values are your elevation data.
    zz = map.elevation_array

    # Create a structured grid for PyVista.
    grid = pv.StructuredGrid(xx, yy, zz)

    plotter.add_mesh(grid, cmap="terrain", show_scalar_bar=True)

if __name__ == "__main__":
    map = map_layers.Map(default_params)
    plotter = pv.Plotter()
    plot_elevation_with_pyvista(plotter, map)
    plotter.show()
