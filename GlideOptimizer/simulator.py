from GlideOptimizer.map_layers import Map, FlatMap
from GlideOptimizer.sim_params import SimParams
from GlideOptimizer.particles import ParticleManager
from GlideOptimizer.visualization import Visualizer, ParticlePlotter
from GlideOptimizer.vehicle import ConeSprayer, LinearDrone, iDrone, StationarySprayer
from GlideOptimizer.sim_params import test_params, generic_amf_params, chrysopogon_params, default_params
from tqdm import tqdm

import numpy as np
#np.seterr(all='raise')

class Simulator:
    def __init__(self, map: Map, particles: ParticleManager, params: SimParams, vehicle: iDrone, visualize=True, timestep=0.003):
        self.map = map
        self.params = params
        self.particle_manager = particles
        self.drone = vehicle
        self.visualize = visualize
        self.timestep = timestep

    def run(self):
        # TODO(Heidt) we should move to RK4 or adaptive timesteps or something, but will need to unify updates
        UPDATE_STEPS = int(1/self.timestep)
        disable_visualization = not self.visualize
        if self.visualize: 
            self.visualizer = ParticlePlotter(self.map, self.particle_manager)

        i = 0
        with tqdm(total=100, disable=disable_visualization) as pbar:
            while not self.drone.is_done():
                self.particle_manager.step(self.timestep)
                self.particle_manager.check_droplet_collision(self.map)
                new_particles = self.drone.step(self.timestep)
                self.particle_manager.add_collection(new_particles)
                completion_ratio = self.drone.completion_ratio()
                i += 1
                ratio = (completion_ratio[0] / completion_ratio[1]) * 100
                pbar.update(ratio - pbar.n)
                if self.visualize:
                    if i % UPDATE_STEPS == 0:
                        self.visualizer.update()

        i = 0
        with tqdm(total=100, disable=disable_visualization) as pbar:
            while not self.particle_manager.particles_done():
                self.particle_manager.step(self.timestep)
                self.particle_manager.check_droplet_collision(self.map)
                if i % UPDATE_STEPS == 0:
                    droplet_ratio = self.particle_manager.get_completion_ratio()
                    pbar.update((float(droplet_ratio[0]) / float(droplet_ratio[1])) * 100.0 - pbar.n)
                    if self.visualize:
                        self.visualizer.update()
                i += 1
        if self.visualize: 
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


def test_sim_seeds():
    map = FlatMap(test_params, 500.0, 500.0)
    particles = ParticleManager(test_params)
    sprayer = ConeSprayer(90, chrysopogon_params)
    vehicle = StationarySprayer(np.array((250.0, 250.0, 1000.0)), sprayer, 10)
    sim = Simulator(map, particles, test_params, vehicle, timestep=0.05)
    sim.run()

def test_sim_spores():
    map = Map(default_params)
    particles = ParticleManager(default_params)
    sprayer = ConeSprayer(90, generic_amf_params)

    trajectory = create_rectilinear_grid(np.array([200., 400., 500.]), 400., 200., 8)
    vehicle = LinearDrone(trajectory, 10.0, sprayer)
    sim = Simulator(map, particles, default_params, vehicle, timestep=0.003)
    sim.run()
        
if __name__ == "__main__":
    test_sim_spores()