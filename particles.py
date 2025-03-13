import numpy as np
from sim_params import SimParams
from map_layers import Map

"""
    droplet models are from the following papers:
    "Influence of droplet collision modelling in Euler/Lagrange calculations of
spray evolution"
"A Simulation Model of the Spray Drift from Hydraulic Nozzles"
"""

AIR_DENSITY = 1.293 # kg/m^3
WATER_DENSITY = 977 # kg/m^3
# TODO(Heidt) this seems pretty dependent on temperature/humidity, might have to add that in
AIR_VISCOSITY = 3.178e-5 # N*s/m^2
LAMBDA = 0.01

class ParticleManager:
    def __init__(self, params: SimParams):
        # TODO(Heidt) need to vectorize these operations, how best to do with lifetime of droplets?
        self.droplet_positions = []
        self.droplet_velocities = []
        self.droplet_sizes = []
        self.droplet_active = []
        self.droplet_noise = []
        self.wind_speed = params.wind_speed
        self.wind_deviation = params.wind_deviation
        

    def add_droplet(self, initial_position: np.ndarray, initial_velocity: np.ndarray, initial_size: np.ndarray):
        self.droplet_positions.append(np.array(initial_position))
        self.droplet_velocities.append(np.array(initial_velocity))
        self.droplet_noise.append(np.array([0., 0., 0.]))
        self.droplet_sizes.append(initial_size)
        self.droplet_active.append(True)

    def step(self, delta_t: float):
        # Euler method step
        for i, drop in enumerate(self.droplet_positions):
            if not self.droplet_active[i]:
                continue
            velocity_fluctuation = np.random.normal(0, scale=self.wind_deviation, size=3)
            #TODO(Heidt) look at the paper that details the brownian motion
            self.droplet_noise[i] += -self.droplet_noise[i]*LAMBDA + velocity_fluctuation

            fluid_vel = self.wind_speed + self.droplet_noise[i]
            Re = AIR_DENSITY * self.droplet_sizes[i] * np.linalg.norm(fluid_vel - self.droplet_velocities[i]) / AIR_VISCOSITY
            if Re < 1000:
                Cd = 24 * (1 + 0.15 * Re**0.687) / Re
            else:
                Cd = 0.44

            velocity_diff = fluid_vel - self.droplet_velocities[i]
            air_accel = 3/4 * AIR_DENSITY * Cd * (velocity_diff) * np.linalg.norm(velocity_diff)
            grav_accel = np.array([0, 0, -9.8]) * (1 - AIR_DENSITY/WATER_DENSITY)


            self.droplet_positions[i] += self.droplet_velocities[i] * delta_t
            self.droplet_velocities[i] += (air_accel + grav_accel) * delta_t
            # TODO(Heidt) add in droplet evaporation term

    def check_droplet_collision(self, map: Map):
        for i, drop in enumerate(self.droplet_positions):
            if drop[2] < map.get_elevation_at(drop[0], drop[1]):
                map.add_droplet_at(drop[0], drop[1])
                self.droplet_active[i] = False

