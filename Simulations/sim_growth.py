from GlideOptimizer import growth_sim
from GlideOptimizer import sim_params

import matplotlib.pyplot as plt
from tqdm import tqdm
from GlideOptimizer import map_layers
from scipy.ndimage import zoom


def plot_growth_satellite_overlay(grid, params, save_path="growth_overlay.png"):
    """
    Fetch a satellite image of the simulated area and overlay the plant
    growth estimate (grid.P) on top of it, aligned by lat/lon.
    """
    satellite_img, sat_extent = map_layers.get_satellite_image(params)
    growth_extent = (params.lon_min, params.lon_max, params.lat_min, params.lat_max)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(satellite_img, extent=sat_extent)
    overlay = ax.imshow(
        grid.P,
        extent=growth_extent,
        cmap="viridis",
        alpha=0.55,
        vmin=grid.P.min(),
        vmax=grid.P.max(),
    )
    ax.set_xlim(sat_extent[0], sat_extent[1])
    ax.set_ylim(sat_extent[2], sat_extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Estimated Plant Growth Overlay")
    fig.colorbar(overlay, ax=ax, label="Plant density (P)")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved growth overlay to {save_path}")
    return fig


def main():
    plant_params = sim_params.generic_plant_params
    plant_params.R = 2.0
    plant_params.d = 0.38
    plant_params.v = 30.0
    params = sim_params.default_params
    map = map_layers.Map(params)
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
    plt.close(fig)

    plot_growth_satellite_overlay(grid, params)
    plt.show(block=True)

if __name__ == "__main__":
    main()