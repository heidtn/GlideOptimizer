import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import convolve
from tqdm import tqdm
from PIL import Image

from numba import njit
from numba import cuda, float32
import numpy as np

#np.seterr(all='raise')

from GlideOptimizer.sim_params import PlantGrowthParams, generic_plant_params


class GrowthGrid:
    def __init__(
        self, width: int, height: int, params: PlantGrowthParams, resolution=1.0
    ):
        # https://pure.knaw.nl/ws/portalfiles/portal/478890/Rietkerk_ea_3482.pdf
        # see paper for variable references
        self.width = width
        self.height = height
        self.P = np.zeros((width, height))
        self.W = np.zeros_like(self.P)
        self.O = np.zeros_like(self.P)
        self.params = params
        self.res = resolution

    def random_seed(self):
        self.P = np.zeros((self.width, self.height))
        self.W = np.ones_like(self.P)
        self.O = np.ones_like(self.P)
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                if np.random.randint(0, 100) == 0:
                    self.P[i, j] = np.random.rand()

        self.W *= self.params.R/self.params.r_w
        self.O *= self.params.R/(self.params.alpha * self.params.W_0)

    def step(self, delta_t):
        p = self.params

        # dP/dt calcs
        water_uptake = p.gmax * (self.W / (self.W + p.k_1))
        growth_term = p.c * water_uptake * self.P
        death_term = p.d * self.P
        plant_dispersal = p.D_p * ndimage.laplace(self.P) / self.res**2.0
        dP = growth_term - death_term + plant_dispersal

        # dW/dt calcs
        infiltration = p.alpha * self.O * ((self.P + p.k_2 * p.W_0) / (self.P + p.k_2))
        water_intake = water_uptake * self.P
        water_loss = p.r_w * self.W
        water_diffusion = p.D_w * ndimage.laplace(self.W) / self.res**2.0
        dW = infiltration - water_intake - water_loss + water_diffusion

        # dO/dt calcs
        surface_diffusion = p.D_0 * ndimage.laplace(self.O) / self.res**2.0
        dO = p.R - infiltration + surface_diffusion

        self.P += dP * delta_t
        self.W += dW * delta_t
        self.O += dO * delta_t

        self.P = np.clip(self.P, 0, None)
        self.W = np.clip(self.W, 0, None)
        self.O = np.clip(self.O, 0, None)


    def run_step_gpu(self, delta_t, n_steps):
        params = self.params
        res = self.res
        # Copy to device
        P_d = cuda.to_device(self.P.astype(np.float64))
        W_d = cuda.to_device(self.W.astype(np.float64))
        O_d = cuda.to_device(self.O.astype(np.float64))

        nx, ny = self.P.shape
        P_out_d = cuda.device_array_like(P_d)
        W_out_d = cuda.device_array_like(W_d)
        O_out_d = cuda.device_array_like(O_d)

        threadsperblock = (16, 16)
        blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Unpack parameters
        c = params.c
        gmax = params.gmax
        k_1 = params.k_1
        d = params.d
        D_p = params.D_p
        alpha = params.alpha
        k_2 = params.k_2
        W_0 = params.W_0
        r_w = params.r_w
        D_w = params.D_w
        D_0 = params.D_0
        R = params.R

        # Main iteration loop
        for step in range(n_steps):
            update_rietkerk[blockspergrid, threadsperblock](
                P_d, W_d, O_d,
                P_out_d, W_out_d, O_out_d,
                c, gmax, k_1, d, D_p, alpha, k_2, W_0, r_w, D_w, D_0, R, res, delta_t
            )
            # Swap pointers for next iteration
            P_d, P_out_d = P_out_d, P_d
            W_d, W_out_d = W_out_d, W_d
            O_d, O_out_d = O_out_d, O_d

        # After all steps, bring results back to host
        self.P = P_d.copy_to_host()
        self.W = W_d.copy_to_host()
        self.O = O_d.copy_to_host()

import time
def main():
    grid = GrowthGrid(200, 200, generic_plant_params, resolution=2.0)
    grid.random_seed()
    images = []
    step = 0.001
    im = None
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots()
    imax = ax.imshow(grid.P, cmap='viridis', interpolation='nearest')
    plt.colorbar(imax, ax=ax)
    plt.show()

    for i in tqdm(range(int(1/step * 3000 / 5000))):
        grid.run_step_gpu(step, 5000)
        dat = grid.P
        im = Image.fromarray((dat - dat.min()) * 255/(dat.max() - dat.min()))
        arr = grid.P
        images.append(im)
        imax.set_data(arr)
        imax.set_clim(vmin=arr.min(), vmax=arr.max())  # Optional: autoscale color
        plt.title(f'Timestep {step}')
        plt.pause(0.01)  # Tiny pause to refresh plot

    plt.imshow(grid.P)
    plt.colorbar()
    plt.show() 

    images[0].save('plant_growth.gif', save_all=True, append_images=images[1:], duration=30)
    im = im.convert("RGB")
    dat = grid.P
    im = Image.fromarray((dat - dat.min()) * 255/(dat.max() - dat.min()))
    im = im.convert("RGB")
    im.save("test.png")


@cuda.jit
def update_rietkerk(
    P, W, O,
    P_out, W_out, O_out,
    c, gmax, k_1, d, D_p, alpha, k_2, W_0, r_w, D_w, D_0, R, res, delta_t
):
    i, j = cuda.grid(2)
    nx, ny = P.shape

    # Helper for Laplacian with reflecting boundary conditions
    def laplace(U, i, j):
        im1 = i-1 if i > 0 else 0         # reflect at 0
        ip1 = i+1 if i < nx-1 else nx-1   # reflect at nx-1
        jm1 = j-1 if j > 0 else 0         # reflect at 0
        jp1 = j+1 if j < ny-1 else ny-1   # reflect at ny-1

        return (
            U[im1, j] + U[ip1, j] +
            U[i, jm1] + U[i, jp1] -
            4.0 * U[i, j]
        )

    if i < nx and j < ny:
        # dP/dt
        growth_term = c * gmax * (W[i, j] / (W[i, j] + k_1)) * P[i, j]
        death_term = d * P[i, j]
        plant_dispersal = D_p * laplace(P, i, j) / (res * res)
        dP = growth_term - death_term + plant_dispersal

        # dW/dt
        infiltration = alpha * O[i, j] * ((P[i, j] + k_2 * W_0) / (P[i, j] + k_2))
        water_intake = gmax * (W[i, j] / (W[i, j] + k_1)) * P[i, j]
        water_loss = r_w * W[i, j]
        water_diffusion = D_w * laplace(W, i, j) / (res * res)
        dW = infiltration - water_intake - water_loss + water_diffusion

        # dO/dt
        surface_diffusion = D_0 * laplace(O, i, j) / (res * res)
        dO = R - infiltration + surface_diffusion

        # Euler update and clipping
        P_new = max(P[i, j] + dP * delta_t, 0.0)
        W_new = max(W[i, j] + dW * delta_t, 0.0)
        O_new = max(O[i, j] + dO * delta_t, 0.0)

        P_out[i, j] = P_new
        W_out[i, j] = W_new
        O_out[i, j] = O_new

            


if __name__ == "__main__":
    main()
