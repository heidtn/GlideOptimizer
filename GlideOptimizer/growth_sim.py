import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import convolve
from tqdm import tqdm
from PIL import Image

from numba import njit
from numba import cuda, float32
import numpy as np
import noise

# np.seterr(all='raise')

from GlideOptimizer.sim_params import PlantGrowthParams, generic_plant_params


def update_point(coords, seed, scale, map_size):
    return noise.snoise2(coords[0]/scale,
                          coords[1]/scale,
                          octaves=6,
                          persistence=0.5,
                          lacunarity=2,
                          repeatx=map_size[0],
                          repeaty=map_size[1],
                          base=seed
                         )

def expo(heightmap, heightmap_size, e):
    for x in range(heightmap_size[0]):
        for y in range(heightmap_size[1]):
            heightmap[x][y] = heightmap[x][y]**e
    return (heightmap / heightmap.max())

def generate_heightmap(map_size, max_height, scale, e=1.0):
    seed = int(np.random.random()*1000)
    heightmap = np.zeros(map_size)

    for x in range(map_size[0]):
        for y in range(map_size[1]):
            new_value = update_point((x, y), seed, scale, map_size)
            heightmap[x][y] = new_value
    heightmap = expo(heightmap, map_size, e)
    return (heightmap / heightmap.max()) * max_height

class GrowthGrid:
    def __init__(
        self,
        width: int,
        height: int,
        heightmap: np.ndarray,
        params: PlantGrowthParams,
        resolution=1.0,
    ):
        # https://pure.knaw.nl/ws/portalfiles/portal/478890/Rietkerk_ea_3482.pdf
        # see paper for variable references
        assert heightmap.shape[0] == width and heightmap.shape[1] == height
        self.width = width
        self.height = height
        self.res = resolution
        self.heightmap = heightmap
        self.params = params
        self.P = np.zeros((width, height))
        self.W = np.zeros_like(self.P)
        self.O = np.zeros_like(self.P)
        self.v_x, self.v_y = np.gradient(self.heightmap, self.res)
        self.v_x *= params.v
        self.v_y *= params.v
        

    def random_seed(self):
        self.P = np.zeros((self.width, self.height))
        self.W = np.ones_like(self.P)
        self.O = np.ones_like(self.P)
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                if np.random.randint(0, 100) == 0:
                    self.P[i, j] = np.random.rand() * 5.0 * self.res**2.0

        self.W *= self.params.R / self.params.r_w
        self.O *= self.params.R / (self.params.alpha * self.params.W_0)

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

    def run_step_gpu(self, delta_t, n_steps, dtype=np.float32):
        params = self.params
        res = self.res
        # Copy to device. dtype defaults to float32: this is a forward-Euler
        # reaction-diffusion system with values that stay in a small bounded
        # range, so float64 precision is unnecessary, and float32 roughly
        # halves memory traffic (the dominant cost for this kernel) while
        # also running at full speed on consumer GPUs, which throttle
        # float64 throughput heavily versus float32.
        P_d = cuda.to_device(self.P.astype(dtype))
        W_d = cuda.to_device(self.W.astype(dtype))
        O_d = cuda.to_device(self.O.astype(dtype))
        vx_d = cuda.to_device(self.v_x.astype(dtype))
        vy_d = cuda.to_device(self.v_y.astype(dtype))

        nx, ny = self.P.shape
        P_out_d = cuda.device_array_like(P_d)
        W_out_d = cuda.device_array_like(W_d)
        O_out_d = cuda.device_array_like(O_d)

        threadsperblock = (16, 16)
        blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Unpack parameters. Cast to the same dtype as the arrays: mixing a
        # float32 array with a float64 Python-float scalar would silently
        # promote every expression in the kernel back to float64.
        c = dtype(params.c)
        gmax = dtype(params.gmax)
        k_1 = dtype(params.k_1)
        d = dtype(params.d)
        D_p = dtype(params.D_p)
        alpha = dtype(params.alpha)
        k_2 = dtype(params.k_2)
        W_0 = dtype(params.W_0)
        r_w = dtype(params.r_w)
        D_w = dtype(params.D_w)
        D_0 = dtype(params.D_0)
        R = dtype(params.R)
        res = dtype(res)
        delta_t = dtype(delta_t)

        # Main iteration loop
        for step in range(n_steps):
            update_rietkerk[blockspergrid, threadsperblock](
                P_d,
                W_d,
                O_d,
                vx_d,
                vy_d,
                P_out_d,
                W_out_d,
                O_out_d,
                c,
                gmax,
                k_1,
                d,
                D_p,
                alpha,
                k_2,
                W_0,
                r_w,
                D_w,
                D_0,
                R,
                res,
                delta_t,
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
    heightmap = generate_heightmap((800, 800), 50.0, 160.0, e=2.0)
    plt.imshow(heightmap)
    plt.show()
    grid = GrowthGrid(800, 800, heightmap, generic_plant_params, resolution=2.0)
    grid.random_seed()
    images = []
    step = 0.001
    im = None
    plt.ion()  # Turn interactive mode on
    fig, (ax, ax1) = plt.subplots(2)
    imax = ax.imshow(grid.P, cmap="viridis", interpolation="nearest")
    ax1.imshow(heightmap, cmap="viridis")
    plt.colorbar(imax, ax=ax)
    plt.show()

    for i in tqdm(range(int(1 / step * 3000 / 5000))):
        grid.run_step_gpu(step, 5000)
        dat = grid.P
        im = Image.fromarray((dat - dat.min()) * 255 / (dat.max() - dat.min()))
        arr = grid.P
        images.append(im)
        imax.set_data(arr)
        imax.set_clim(vmin=arr.min(), vmax=arr.max())  # Optional: autoscale color
        plt.title(f"Timestep {step}")
        plt.pause(0.01)  # Tiny pause to refresh plot

    plt.imshow(grid.P)
    plt.colorbar()
    plt.show()

    images[0].save(
        "plant_growth.gif", save_all=True, append_images=images[1:], duration=30
    )
    im = im.convert("RGB")
    dat = grid.P
    im = Image.fromarray((dat - dat.min()) * 255 / (dat.max() - dat.min()))
    im = im.convert("RGB")
    im.save("test.png")


@cuda.jit
def update_rietkerk(
    P,
    W,
    O,
    v_x,
    v_y,
    P_out,
    W_out,
    O_out,
    c,
    gmax,
    k_1,
    d,
    D_p,
    alpha,
    k_2,
    W_0,
    r_w,
    D_w,
    D_0,
    R,
    res,
    delta_t,
):
    i, j = cuda.grid(2)
    nx, ny = P.shape

    if i < nx and j < ny:
        im1 = i - 1 if i > 0 else 0  # reflect at 0
        ip1 = i + 1 if i < nx - 1 else nx - 1  # reflect at nx-1
        jm1 = j - 1 if j > 0 else 0  # reflect at 0
        jp1 = j + 1 if j < ny - 1 else ny - 1  # reflect at ny-1

        # cache each field's value at this cell once instead of re-reading
        # it from global memory on every use
        Pc = P[i, j]
        Wc = W[i, j]
        Oc = O[i, j]

        inv_res2 = 1.0 / (res * res)
        P_lap = (P[im1, j] + P[ip1, j] + P[i, jm1] + P[i, jp1] - 4.0 * Pc) * inv_res2
        W_lap = (W[im1, j] + W[ip1, j] + W[i, jm1] + W[i, jp1] - 4.0 * Wc) * inv_res2
        O_lap = (O[im1, j] + O[ip1, j] + O[i, jm1] + O[i, jp1] - 4.0 * Oc) * inv_res2

        # dP/dt
        water_uptake = gmax * (Wc / (Wc + k_1))  # reused below instead of recomputed
        growth_term = c * water_uptake * Pc
        death_term = d * Pc
        dP = growth_term - death_term + D_p * P_lap

        # dW/dt
        infiltration = alpha * Oc * ((Pc + k_2 * W_0) / (Pc + k_2))
        water_intake = water_uptake * Pc
        water_loss = r_w * Wc
        dW = infiltration - water_intake - water_loss + D_w * W_lap

        # dO/dt: fluxes at each face, central difference for divergence
        F_x_ip1 = v_x[ip1, j] * O[ip1, j]
        F_x_im1 = v_x[im1, j] * O[im1, j]
        F_y_jp1 = v_y[i, jp1] * O[i, jp1]
        F_y_jm1 = v_y[i, jm1] * O[i, jm1]

        inv_2res = 1.0 / (2.0 * res)
        dF_x_dx = (F_x_ip1 - F_x_im1) * inv_2res
        dF_y_dy = (F_y_jp1 - F_y_jm1) * inv_2res
        surface_flux = dF_x_dx + dF_y_dy
        dO = R - infiltration + surface_flux + D_0 * O_lap

        # Euler update and clipping
        P_out[i, j] = max(Pc + dP * delta_t, 0.0)
        W_out[i, j] = max(Wc + dW * delta_t, 0.0)
        O_out[i, j] = max(Oc + dO * delta_t, 0.0)


if __name__ == "__main__":
    main()
