import numpy as np
import pyvista as pv
from rasterio.transform import from_origin
import map_layers
from particles import ParticleManager
from scipy.interpolate import griddata
from sim_params import default_params

class Visualizer:
    def __init__(self, map: map_layers.Map, particles: ParticleManager):
        self.map = map
        self.particles = particles
        self.plotter = pv.Plotter(notebook=False, off_screen=True)
        #self.plotter = pv.Plotter()
        #plot_elevation_with_pyvista(self.plotter, map)
        self.particle_points = None
        self.plotter.open_gif("test.gif")

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
        self.plotter.write_frame()
        #self.plotter.render()

    def close(self):
        self.plotter.close()

class ParticlePlotter:
    def __init__(self, map: map_layers.Map, particles: ParticleManager):
        self.map = map
        self.particles = particles.pc
        self.plotter = pv.Plotter(shape=(2, 1))
        self.plotter.subplot(0, 0)
        self.particle_points = None
        self.particle_paths: list = None

    def update(self):
        if self.particle_paths is None:
            if self.particles.droplet_positions.shape[0] > 0:
                self.particle_paths = [[np.array(drop)] for drop in self.particles.droplet_positions]
            else:
                return
        for i, particle_pos in enumerate(self.particles.droplet_positions):
            if self.particles.droplet_active[i]:
                self.particle_paths[i].append(np.array(particle_pos))

    def close(self):
        plot_elevation_with_pyvista(self.plotter, self.map)
        from tqdm import tqdm
        for i in tqdm(range(len(self.particle_paths))):
            spline = pv.Spline(np.array(self.particle_paths[i]))
            self.plotter.add_mesh(spline, scalars='arc_length', render_lines_as_tubes=False, line_width=3)

        self.plotter.subplot(1, 0)
        plot_droplets_with_pyvista(self.plotter, self.map)
        self.plotter.show()
        

def plot_elevation_with_pyvista(plotter: pv.Plotter, map: map_layers.Map):
    nrows, ncols = map.elevation_array.shape
    x_coords = np.arange(ncols) * map.dx
    y_coords = np.arange(nrows) * map.dy

    xx, yy = np.meshgrid(x_coords, y_coords)
    zz = map.elevation_array

    elevation_grid = pv.StructuredGrid(xx, yy, map.elevation_array)
    plotter.add_mesh(elevation_grid, cmap="terrain", show_scalar_bar=True)

def plot_droplets_with_pyvista(plotter: pv.Plotter, map: map_layers.Map):
    drop_nrows, drop_ncols = map.droplet_array.shape
    drop_x_coords = np.arange(drop_ncols) * (map.x_size / drop_ncols)
    drop_y_coords = np.arange(drop_nrows) * (map.y_size / drop_nrows)
    drop_xx, drop_yy = np.meshgrid(drop_x_coords, drop_y_coords)
    drop_zz = map.droplet_array / 100.0
    droplets = pv.StructuredGrid(drop_xx, drop_yy, drop_zz)
    plotter.add_mesh(droplets, show_scalar_bar=True)


if __name__ == "__main__":
    map = map_layers.Map(default_params)
    plotter = pv.Plotter()
    plot_elevation_with_pyvista(plotter, map)
    plotter.show()
