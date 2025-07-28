from GlideOptimizer import growth_sim
from GlideOptimizer import sim_params

import matplotlib.pyplot as plt
from tqdm import tqdm
from GlideOptimizer import map_layers
from scipy.ndimage import zoom


def main():
    plant_params = sim_params.generic_plant_params
    plant_params.R = 2.0
    plant_params.d = 0.38
    plant_params.v = 30.0
    map = map_layers.Map(sim_params.default_params)
    heightmap = map.elevation_array
    scale_factor = 32 
    heightmap = zoom(heightmap, scale_factor, order=3)
    res = map.dx / scale_factor

    grid = growth_sim.GrowthGrid(heightmap.shape[0], heightmap.shape[1], heightmap, plant_params, resolution=res)
    grid.random_seed()
    step = 0.001
    im = None
    plt.ion()
    fig, (ax, ax1) = plt.subplots(2)
    imax = ax.imshow(grid.P, cmap="viridis", interpolation="nearest")
    ax1.imshow(heightmap, cmap="viridis")
    plt.colorbar(imax, ax=ax)
    plt.show()

    for i in tqdm(range(int(1 / step * 3000 / 5000))):
        grid.run_step_gpu(step, 5000)
        dat = grid.P
        imax.set_data(dat)
        imax.set_clim(vmin=dat.min(), vmax=dat.max()) 
        plt.title(f"Timestep {step}")
        plt.pause(0.01)

    plt.ioff()
    plt.show(block=True)

if __name__ == "__main__":
    main()