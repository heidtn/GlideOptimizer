from map_layers import Map
from sim_params import SimParams
from particles import ParticleManager
from visualization import Visualizer, ParticlePlotter
from vehicle import ConeSprayer, LinearDrone, iDrone
from sim_params import default_params

import numpy as np

class Simulator:
    def __init__(self, map: Map, particles: ParticleManager, params: SimParams, vehicle: iDrone):
        self.map = map
        self.params = params
        self.particle_manager = particles
        self.visualizer = ParticlePlotter(self.map, self.particle_manager)
        self.drone = vehicle

    def run(self):
        RUNTIME = 100.0
        TIMESTEP = 0.1
        UPDATE_STEPS = 10

        from tqdm import tqdm
        for i in tqdm(range(int(RUNTIME / TIMESTEP))):
            self.particle_manager.step(TIMESTEP)
            self.particle_manager.check_droplet_collision(self.map)
            new_particles = self.drone.step(TIMESTEP)
            self.particle_manager.add_collection(new_particles)
            if i % UPDATE_STEPS == 0:
                self.visualizer.update()
        
        self.visualizer.close()
        
if __name__ == "__main__":
    map = Map(default_params)
    particles = ParticleManager(default_params)
    sprayer = ConeSprayer(40)

    trajectory = np.array([
        [100., 100., 500.],
        [100., 200., 500.],
        [200., 200., 500.],
        [200., 300., 500.],
        [300., 300., 500.],
    ])
    vehicle = LinearDrone(trajectory, 5.0, sprayer)
    sim = Simulator(map, particles, default_params, vehicle)
    sim.run()