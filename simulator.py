from map_layers import Map
from sim_params import SimParams
from particles import ParticleManager
from visualization import Visualizer, ParticlePlotter
from vehicle import ConeSprayer, LinearDrone, iDrone
from sim_params import default_params

import numpy as np
#np.seterr(all='raise')

class Simulator:
    def __init__(self, map: Map, particles: ParticleManager, params: SimParams, vehicle: iDrone):
        self.map = map
        self.params = params
        self.particle_manager = particles
        self.drone = vehicle

    def run(self):
        RUNTIME = 60.0*3.0
        TIMESTEP = 0.003
        UPDATE_STEPS = int(1/TIMESTEP)

        self.visualizer = ParticlePlotter(self.map, self.particle_manager)

        from tqdm import tqdm
        for i in tqdm(range(int(RUNTIME / TIMESTEP))):
            self.particle_manager.step(TIMESTEP)
            self.particle_manager.check_droplet_collision(self.map)
            new_particles = self.drone.step(TIMESTEP)
            self.particle_manager.add_collection(new_particles)
            if i % UPDATE_STEPS == 0:
                self.visualizer.update()

        i = 0
        #while np.any(self.particle_manager.pc.droplet_active):
        for i in tqdm(range(15000)):
            self.particle_manager.step(TIMESTEP)
            self.particle_manager.check_droplet_collision(self.map)
            if i % UPDATE_STEPS == 0:
                self.visualizer.update()
            i += 1
        
        self.visualizer.close()

def create_rectilinear_grid(start, length, spacing, count):
    grid = []
    pattern = np.array([
        np.array([length, 0.0, 0.0]),
        np.array([0.0, spacing, 0.0]),
        np.array([-length, 0.0, 0.0]),
        np.array([0.0, spacing, 0.0]),
    ])
    current_position = np.array(start)
    for i in range(count):
        for position in pattern:
            grid.append(np.array(current_position))
            current_position += position 
    return np.array(grid)
        
if __name__ == "__main__":
    map = Map(default_params)
    particles = ParticleManager(default_params)
    sprayer = ConeSprayer(90)

    trajectory = create_rectilinear_grid(np.array([200., 400., 500.]), 400., 200., 8)
    vehicle = LinearDrone(trajectory, 10.0, sprayer)
    sim = Simulator(map, particles, default_params, vehicle)
    sim.run()