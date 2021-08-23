import taichi as ti
from mpm_base import MPMSimulationBase
ti.init(arch=ti.cuda)


@ti.data_oriented
class MlsMpmExplicitSolver(MPMSimulationBase):
    def __init__(self):
        super(MlsMpmExplicitSolver, self).__init__()
        self.quality = 1  # Use a larger value for higher-res simulations
        self.n_particles, self.n_grid = 9000 * self.quality ** 2, 128 * self.quality
        self.group_size = self.n_particles // 3
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho
        self.E, self.nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
                (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

        # Quantities defined on particles
        self.x = ti.Vector.field(2, dtype=float, shape=self.n_particles)  # position
        self.v = ti.Vector.field(2, dtype=float, shape=self.n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float,
                            shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float,
                            shape=self.n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=self.n_particles)  # material id
        self.Jp = ti.field(dtype=float, shape=self.n_particles)  # plastic deformation

        # Quantities defined on grid
        self.grid_v = ti.Vector.field(2, dtype=float,
                                 shape=(self.n_grid, self.n_grid))  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))  # grid node mass

    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // self.group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // self.group_size)
            ]
            self.material[i] = i // self.group_size  # 0: fluid 1: jelly 2: snow
            self.v[i] = ti.Matrix([0, 0])
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1

    @ti.kernel
    def reinitialize(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0

    @ti.kernel
    def particles_to_grid(self):
        for p in self.x:  # Particle state update and scatter to grid (P2G)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.F[p] = (ti.Matrix.identity(float, 2) +
                    self.dt * self.C[p]) @ self.F[p]  # deformation gradient update
            h = ti.exp(
                10 *
                (1.0 -
                 self.Jp[p]))  # Hardening coefficient: snow gets harder when compressed
            if self.material[p] == 1:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if self.material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
                self.F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.material[p] == 2:
                self.F[p] = U @ sig @ V.transpose(
                )  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
            ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(
                    3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_operations(self):
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:  # No need for epsilon here
                self.grid_v[i,
                       j] = (1 / self.grid_m[i, j]) * self.grid_v[i,
                                                        j]  # Momentum to velocity
                self.grid_v[i, j][1] -= self.dt * 50  # gravity
                if i < 3 and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0  # Boundary conditions
                if i > self.n_grid - 3 and self.grid_v[i, j][0] > 0: self.grid_v[i, j][0] = 0
                if j < 3 and self.grid_v[i, j][1] < 0: self.grid_v[i, j][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0: self.grid_v[i, j][1] = 0

    @ti.kernel
    def grid_to_particles(self):
        for p in self.x:  # grid to particle (G2P)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(
                    3, 3)):  # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection


if __name__ == '__main__':
    gui = ti.GUI("Taichi MLS-MPM-EXPLICIT", res=512, background_color=0x112F41)
    solver = MlsMpmExplicitSolver()
    solver.initialize()
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-3 // solver.dt)):
            solver.advance_one_time_step()
        gui.circles(solver.x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEEF0],
                    palette_indices=solver.material)
        gui.show(
        )  # Change to gui.show(f'{frame:06d}.png') to write images to disk