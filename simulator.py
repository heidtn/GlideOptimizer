from map_layers import Map
from sim_params import SimParams
from particles import ParticleManager
from visualization import Visualizer, ParticlePlotter

import numpy as np

class Simulator:
    def __init__(self, map: Map, particles: ParticleManager, params: SimParams):
        self.map = map
        self.params = params
        self.particle_manager = particles
        self.visualizer = ParticlePlotter(self.map, self.particle_manager)

    def run(self):
        RUNTIME = 100.0
        TIMESTEP = 0.05

        center = np.array([100., 100., 500]) 
        for i in range(1000):
            self.particle_manager.add_droplet(center, np.random.rand(3), 200e-6)
            center += np.array([1.0, 1., 0.0])

        from tqdm import tqdm
        for i in tqdm(range(int(RUNTIME / TIMESTEP))):
            self.particle_manager.step(TIMESTEP)
            self.particle_manager.check_droplet_collision(self.map)
            self.visualizer.update()
        
        self.visualizer.close()
        
if __name__ == "__main__":
    from sim_params import default_params
    map = Map(default_params)
    particles = ParticleManager(default_params)
    sim = Simulator(map, particles, default_params)
    sim.run()