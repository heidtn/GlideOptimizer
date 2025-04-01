from GlideOptimizer import simulator
from GlideOptimizer.map_layers import FlatMap
from GlideOptimizer.particles import ParticleManager
from GlideOptimizer.vehicle import ConeSprayer, StationarySprayer
from GlideOptimizer.simulator import Simulator
from GlideOptimizer.sim_params import default_params

if __name__ == "__main__":
    map = FlatMap(default_params)
    particles = ParticleManager(default_params)
    sprayer = ConeSprayer(90)
    vehicle = StationarySprayer()
    sim = Simulator(map, particles, default_params, vehicle)
    sim.run()