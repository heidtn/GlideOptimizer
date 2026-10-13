from GlideOptimizer import growth_sim
from GlideOptimizer import sim_params

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from GlideOptimizer import map_layers
from scipy.ndimage import zoom


def plot_growth_satellite_overlay(
    grid, params, save_path="growth_overlay.png", initial_alpha=0.8, vmin=0.0, vmax=None
):
    """
    Fetch a satellite image of the simulated area and overlay the plant
    growth estimate (grid.P) on top of it, aligned by lat/lon. Includes an
    interactive slider to adjust the overlay's opacity.

    Opacity is applied per pixel, scaled by plant density, so bare ground
    shows the satellite image through instead of being tinted uniformly.
    A uniform (scalar) alpha composites low-density cells just as opaquely
    as dense ones, which makes the same P value read as a different colour
    over dark tree canopy than over bright grass.

    The colour scale is fixed to [vmin, vmax] (default: 0 to the field's
    max) rather than autoscaled to [P.min(), P.max()] -- autoscaling
    stretches a near-uniform field across the whole colormap and visually
    amplifies trivial variation. Pin vmax explicitly to compare runs.
    """
    satellite_img, sat_extent = map_layers.get_satellite_image(params)
    growth_extent = (params.lon_min, params.lon_max, params.lat_min, params.lat_max)

    P = grid.P
    if vmax is None:
        vmax = float(P.max())
    # guard against a degenerate (all-zero / constant) field
    span = vmax - vmin
    if span <= 0:
        span = 1.0
    # normalized density in [0, 1], used to drive per-pixel opacity
    density_norm = np.clip((P - vmin) / span, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(bottom=0.15)
    ax.imshow(satellite_img, extent=sat_extent)
    overlay = ax.imshow(
        P,
        extent=growth_extent,
        cmap="viridis",
        alpha=density_norm * initial_alpha,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(sat_extent[0], sat_extent[1])
    ax.set_ylim(sat_extent[2], sat_extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Estimated Plant Growth Overlay")
    fig.colorbar(overlay, ax=ax, label="Plant density (P)")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved growth overlay to {save_path}")

    slider_ax = fig.add_axes([0.25, 0.04, 0.5, 0.03])
    alpha_slider = Slider(slider_ax, "Overlay opacity", 0.0, 1.0, valinit=initial_alpha)

    def _on_alpha_change(val):
        # rescale the whole per-pixel alpha field, preserving its shape
        overlay.set_alpha(density_norm * val)
        fig.canvas.draw_idle()

    alpha_slider.on_changed(_on_alpha_change)
    # keep a reference so the slider isn't garbage-collected and stops responding
    fig._alpha_slider = alpha_slider

    return fig


def main():
    plant_params = sim_params.generic_plant_params
    plant_params.R = 1.47
    plant_params.d = 0.15
    params = sim_params.default_params
    map = map_layers.Map(params)
    heightmap = map.elevation_array
    scale_factor = 8
    heightmap = zoom(heightmap, scale_factor, order=3)
    res = map.dx / scale_factor

    grid = growth_sim.GrowthGrid(heightmap.shape[0], heightmap.shape[1], heightmap, plant_params, resolution=res)
    grid.random_seed()
    step = 0.01
    im = None
    plt.ion()
    fig, (ax, ax1) = plt.subplots(2)
    imax = ax.imshow(grid.P, cmap="viridis", interpolation="nearest")
    ax1.imshow(heightmap, cmap="viridis")
    plt.colorbar(imax, ax=ax)
    plt.show()

    for i in tqdm(range(int(1 / step * 900 / 5000))):
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