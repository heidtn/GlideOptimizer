from GlideOptimizer import simulator
from GlideOptimizer.map_layers import FlatMap
from GlideOptimizer.particles import ParticleManager
from GlideOptimizer.vehicle import ConeSprayer, StationarySprayer
from GlideOptimizer.simulator import Simulator
from GlideOptimizer.sim_params import test_params, chrysopogon_params

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def run_one_simulation(height: float, wind_speed: float):
    test_params.wind_speed = np.array([0, wind_speed, 0.])

    map = FlatMap(test_params, 5000.0, 5000.0)
    particles = ParticleManager(test_params)
    sprayer = ConeSprayer(90, chrysopogon_params)
    vehicle = StationarySprayer(np.array((2500.0, 2500.0, height)), sprayer, 10)
    sim = Simulator(map, particles, test_params, vehicle, visualize=False, timestep=0.05)
    sim.run()
    cep = map.get_droplet_cep(70)
    hist = map.get_droplet_histogram()
    center = map.get_droplet_centroid_distance()
    return cep, hist, center

def main():
    height_ranges = np.arange(10.0, 500.0, step=50.0)
    speed_ranges = np.arange(3.0, 18.0, step=3)

    speed_plots = []
    distances = [[] for speed in range(speed_ranges.size)]
    hist = None
    for i, speed in tqdm(enumerate(speed_ranges)):
        spreads = []
        for height in tqdm(height_ranges, leave=False):
            #print(f"Simming {speed} at height {height}")
            cep, hist, center = run_one_simulation(height, speed)
            spreads.append(cep)
            distances[i].append(np.linalg.norm(center))
        speed_plots.append(spreads)

    nrows = int(np.ceil(len(speed_plots) / 2.0))
    fig, axs = plt.subplots(nrows=nrows, ncols=2)
    axs = axs.flatten()
    for i, speed in enumerate(speed_plots):
        axs[i].plot(height_ranges, speed_plots[i])
        axs[i].set_title(f"Spread CEP70 vs Altitude @ wind speed {speed_ranges[i]}")
        axs[i].set_xlabel("Altitude (m)")
        axs[i].set_ylabel("CEP 70% (m)")

    hist_data, bin_edges = hist
    fig, ax = plt.subplots()
    ax.bar(bin_edges[:-1], hist_data, width=np.diff(bin_edges))
    ax.set_title("Histogram of seed landing distance from center")
    ax.set_xlabel("Distance from center (m)")
    ax.set_ylabel("Seed Count (#seeds)") 

    fig, ax = plt.subplots()
    X, Y = np.meshgrid(height_ranges, speed_ranges)
    plot = ax.pcolor(X, Y, np.array(distances))
    fig.colorbar(plot)
    
    ax.set_title("Centroid drift with wind and altitude")
    ax.set_ylabel("Wind Speed m/s")
    ax.set_xlabel("Drop Altitude m")


    plt.show()

if __name__ == "__main__":
    main()