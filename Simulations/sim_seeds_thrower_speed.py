from GlideOptimizer import simulator
from GlideOptimizer.map_layers import FlatMap
from GlideOptimizer.particles import ParticleManager
from GlideOptimizer.vehicle import ConeSprayer, StationarySprayer
from GlideOptimizer.simulator import Simulator
from GlideOptimizer.sim_params import test_params, chrysopogon_params

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def run_one_simulation(height: float, sprayer_speed: float):
    map = FlatMap(test_params, 1000.0, 1000.0)
    particles = ParticleManager(test_params)
    sprayer = ConeSprayer(90, chrysopogon_params)
    chrysopogon_params.particle_speed_mean = sprayer_speed
    vehicle = StationarySprayer(np.array((250.0, 250.0, height)), sprayer, 10)
    sim = Simulator(map, particles, test_params, vehicle, visualize=False, timestep=0.05)
    sim.run()
    cep = map.get_droplet_cep(95)
    hist = map.get_droplet_histogram()
    return cep, hist

def main():
    height_ranges = np.arange(10.0, 6000.0, step=600)
    speed_ranges = np.arange(3.0, 18.0, step=3)
    speed_plots = []
    hist = None
    for speed in tqdm(speed_ranges):
        spreads = []
        for height in tqdm(height_ranges, leave=False):
            cep, hist = run_one_simulation(height, speed)
            spreads.append(cep)
        speed_plots.append(spreads)

    nrows = int(np.ceil(len(speed_plots) / 2.0))
    fig, axs = plt.subplots(nrows=nrows, ncols=2)
    axs = axs.flatten()
    for i, speed in enumerate(speed_plots):
        axs[i].plot(height_ranges, speed_plots[i])
        axs[i].set_title(f"Spread CEP95 vs Altitude @ spreader speed {speed_ranges[i]}")
        axs[i].set_xlabel("Altitude (m)")
        axs[i].set_ylabel("CEP 95% (m)")

    hist_data, bin_edges = hist
    fig, ax = plt.subplots()
    ax.bar(bin_edges[:-1], hist_data, width=np.diff(bin_edges))
    ax.set_title("Histogram of seed landing distance from center")
    ax.set_xlabel("Distance from center (m)")
    ax.set_ylabel("Seed Count (#seeds)") 

    plt.show()

if __name__ == "__main__":
    main()