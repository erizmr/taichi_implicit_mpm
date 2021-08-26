import taichi as ti
from mpm_base import MPMSimulationBase
from gradient_descent import GradientDescentSolver
from diff_test import DiffTest
ti.init(arch=ti.cuda)


@ti.data_oriented
class MlsMpmSolver(MPMSimulationBase):
    def __init__(self, dt=1e-4, dim=2, gravity=9.8, gravity_dim=1, implicit=False, linear_solver=None):
        super(MlsMpmSolver, self).__init__(implicit=implicit)
        if linear_solver == 'gradient_descent':
            self.linear_solver = GradientDescentSolver(max_iterations=15, adaptive_step_size=False)
        self.dim = dim
        self.real = ti.f32
        self.quality = 1  # Use a larger value for higher-res simulations
        self.ignore_collision = True
        self.debug_mode = True
        self.n_particles, self.n_grid = 9000 * self.quality ** 2, 128 * self.quality
        self.n_nodes = self.n_grid ** self.dim
        self.bound = 3  # boundary of the simulation domain
        self.neighbour = (3,) * self.dim
        self.group_size = self.n_particles // 2
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.dt = dt / self.quality
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho
        self.E, self.nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
                (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

        self.gravity = ti.Vector([(-gravity if i == gravity_dim else 0) for i in range(dim)])

        # Quantities defined on particles
        self.x = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)  # position
        self.v = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)  # velocity
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                            shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                            shape=self.n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=self.n_particles)  # material id
        self.Jp = ti.field(dtype=self.real, shape=self.n_particles)  # plastic deformation

        # Quantities defined on grid
        self.grid_v = ti.Vector.field(self.dim, dtype=self.real,
                                 shape=(self.n_grid,)*self.dim)  # grid node momentum/velocity

        self.grid_v_new = ti.Vector.field(self.dim, dtype=self.real,
                                 shape=(self.n_grid,)*self.dim)  # new grid node momentum, also used for store forces temporally

        self.grid_m = ti.field(dtype=self.real, shape=(self.n_grid,)*self.dim)  # grid node mass

        if implicit:
            assert linear_solver
            self.old_F = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                            shape=self.n_particles)  # backup deformation gradient

            # # Quantities for linear solver
            # self.mass_matrix = ti.field(dtype=self.real, shape=self.n_nodes)
            # self.dv = ti.Vector.field(dim, dtype=self.real, shape=self.n_nodes)  # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
            # self.residual = ti.Vector.field(dim, dtype=self.real, shape=self.n_nodes)

            # Quantities for linear solver
            self.mass_matrix = ti.field(dtype=self.real, shape=(self.n_grid,)*self.dim)
            self.dv = ti.Vector.field(dim, dtype=self.real, shape=(self.n_grid,)*self.dim)  # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
            self.residual = ti.Vector.field(dim, dtype=self.real, shape=(self.n_grid,)*self.dim)

            # Define a diff test object
            self.diff_test = DiffTest(self.dim, self.dv, self.total_energy, self.compute_energy_gradient)

    def initialize(self):
        self.simulation_initialize()

    # TODO: currently 2D only
    @ti.kernel
    def simulation_initialize(self):
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // self.group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // self.group_size)
            ]
            self.material[i] = i // self.group_size  # 0: fluid 1: jelly 2: snow
            self.v[i] = ti.Matrix([0, 0])
            if self.material[i] == 0:
                self.v[i] = ti.Matrix([0, 2.0])
            if self.material[i] == 1:
                self.v[i] = ti.Matrix([0, -2.0])
            if self.material[i] == 2:
                self.v[i] = ti.Matrix([0, -3.0])
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.old_F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1

    @ti.kernel
    def reinitialize(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_v_new[i, j] = [0, 0]
            self.grid_m[i, j] = 0

    @ti.kernel
    def particles_to_grid(self):
        for p in self.x:  # Particle state update and scatter to grid (P2G)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.real)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.F[p] = (ti.Matrix.identity(self.real, 2) +
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
                self.F[p] = ti.Matrix.identity(self.real, 2) * ti.sqrt(J)
            elif self.material[p] == 2:
                self.F[p] = U @ sig @ V.transpose(
                )  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
            ) + ti.Matrix.identity(self.real, 2) * la * J * (J - 1)
            # stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            # affine = stress + self.p_mass * self.C[p]
            affine = self.p_mass * self.C[p]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):  # Loop over 3x3 grid node neighborhood
                dpos = (offset.cast(self.real) - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
                self.grid_v_new[base + offset] -= weight * (stress @ dpos)  # Store the stress

        if self.implicit:
            for I in ti.grouped(self.grid_m):
                if self.grid_m[I] > 0:
                    self.grid_v[I] /= self.grid_m[I]  # momentum to velocity

    @ti.kernel
    def grid_velocity_update_explicit(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                forces = self.grid_v_new[I] * (self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx)
                # Update using Symplectic Euler
                self.grid_v[I] = 1 / self.grid_m[I] * (forces + self.grid_v[I])  # Momentum to velocity
                # self.grid_v[I][1] -= self.dt * 50  # gravity
                self.grid_v[I] += self.dt * self.gravity
                if self.dim == 2:
                    i, j = I[0], I[1]
                    if i < 3 and self.grid_v[I][0] < 0:
                        self.grid_v[I][0] = 0  # Boundary conditions
                    if i > self.n_grid - self.bound and self.grid_v[I][0] > 0:
                        self.grid_v[I][0] = 0
                    if j < 3 and self.grid_v[I][1] < 0:
                        self.grid_v[I][1] = 0
                    if j > self.n_grid - self.bound and self.grid_v[I][1] > 0:
                        self.grid_v[I][1] = 0
                if self.dim == 3:
                    k = I[self.dim-1]
                    if k < 3 and self.grid_v[I][self.dim-1] < 0:
                        self.grid_v[I][self.dim-1] = 0
                    if k > self.n_grid - self.bound and self.grid_v[I][self.dim-1] > 0:
                        self.grid_v[I][self.dim-1] = 0

    def backward_euler(self):
        self.build_mass_matrix()
        self.build_initial_dv_for_newton()
        # Which should be called at the beginning of newton.
        self.backup_strain()

        # self.newton_solve()
        self.gradient_descent_solve()

        self.restore_strain()
        self.construct_new_velocity_from_newton_result()


    @ti.func
    def psi(self, F):  # strain energy density function Ψ(F)
        U, sig, V = ti.svd(F)
        # fixed corotated model, you can replace it with any constitutive model
        return self.mu_0 * (F - U @ V.transpose()).norm() ** 2 + self.lambda_0 / 2 * (F.determinant() - 1) ** 2

    @ti.func
    def dpsi_dF(self, F):  # first Piola-Kirchoff stress P(F), i.e. ∂Ψ/∂F
        U, sig, V = ti.svd(F)
        J = F.determinant()
        R = U @ V.transpose()
        return 2 * self.mu_0 * (F - R) + self.lambda_0 * (J - 1) * J * F.inverse().transpose()

    @ti.kernel
    def total_energy(self) -> ti.f32:
        result = ti.cast(0.0, self.real)
        
        # elastic potential energy
        # for p in self.F:
        #     result += self.psi(self.F[p]) * self.p_vol  # gathered from particles, psi defined in the rest space
        # print('after elastic', result)
        # inertia energy
        # for I in ti.grouped(self.dv):
        #     m = self.mass_matrix[I]
        #     dv = self.dv[I]
        #     result += m * dv.dot(dv) / 2
        # print('after inertia', result)
        # gravity potential
        for I in ti.grouped(self.dv):
            m = self.mass_matrix[I]
            # dv = self.dv[I]
            for i in ti.static(range(self.dim)):
                result -= self.dt * m * self.gravity[i]
        # print('after gravity', result)
        return result

    @ti.kernel
    def build_initial_dv_for_newton(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                # node_id = self.idx(I)
                node_id = I
                if ti.static(not self.ignore_collision):
                    cond = (I < self.bound and self.grid_v[I] < 0) or (
                                I > self.n_grid - self.bound and self.grid_v[I] > 0)
                    self.dv[node_id] = 0 if cond else self.gravity * self.dt
                else:
                    self.dv[node_id] = self.gravity * self.dt  # Newton initial guess for non-collided nodes

    @ti.kernel
    def construct_new_velocity_from_newton_result(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] += self.dv[I]   # self.dv[self.idx(I)]
                cond = (I < self.bound and self.grid_v[I] < 0) or (I > self.n_grid - self.bound and self.grid_v[I] > 0)
                self.grid_v[I] = 0 if cond else self.grid_v[I]

    # [i, j] or [i, j, k] -> id
    @ti.func
    def idx(self, I):
        return sum([I[i] * self.n_grid ** i for i in range(self.dim)])

    # id -> [i, j] or [i, j, k]
    @ti.func
    def node(self, p):
        return ti.Vector([(p % (self.n_grid ** (i + 1))) // (self.n_grid ** i) for i in range(self.dim)])

    @ti.kernel
    def backup_strain(self):
        for p in self.F:
            self.old_F[p] = self.F[p]

    @ti.kernel
    def restore_strain(self):
        for p in self.F:
            self.F[p] = self.old_F[p]

    @ti.kernel
    def build_mass_matrix(self):
        for I in ti.grouped(self.grid_m):
            mass = self.grid_m[I]
            if mass > 0:
                # self.mass_matrix[self.idx(I)] = mass
                self.mass_matrix[I] = mass

    @ti.kernel
    def incremental_update(self, dst: ti.template(), src: ti.template(), rate: ti.f32, step_direction: ti.template()):
        for I in ti.grouped(src):
            dst[I] = src[I] + rate * step_direction[I]

    @ti.kernel
    def compute_norm(self) -> ti.f32:
        norm_sq = ti.cast(0.0, self.real)
        for I in ti.grouped(self.dv):
            mass = self.mass_matrix[I]
            residual = self.residual[I]
            if mass > 0:
                norm_sq += residual.dot(residual) / mass
        return ti.sqrt(norm_sq)

    def gradient_descent_solve(self):
        self.diff_test.run(self.dv)
        self.linear_solver.solve(self.compute_energy_gradient, self.dv, self.residual)

        # for i in range(15):
        #     self.compute_energy_gradient()
        #     self.incremental_update(self.dv, self.dv, 10, self.residual)
        #     residual_norm = self.compute_norm()
        #     if ti.static(self.debug_mode):
        #         print('\033[1;31mgradient_descent_iter = ', i, ', residual_norm = ', residual_norm, '\033[0m')

    @ti.func
    def project(self, x: ti.template()):
        for p in ti.grouped(x):
            # I = self.node(p)
            I = p
            cond = any(I < self.bound and self.grid_v[I] < 0) or any(
                I > self.n_grid - self.bound and self.grid_v[I] > 0)
            if cond:
                x[p] = ti.zero(x[p])

    @ti.kernel
    def compute_energy_gradient(self, residual: ti.template()):
        # Compute the RHS (i.e. usually energy gradient) of the linear system
        for I in ti.grouped(self.dv):
            self.residual[I] = self.dt * self.mass_matrix[I] * self.gravity

        for I in ti.grouped(self.dv):
            self.residual[I] -= self.mass_matrix[I] * self.dv[I]

        # Compute dpsi/dF * F^T, P->G
        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                # g_v = self.grid_v[base + offset] + self.dv[self.idx(base + offset)]
                g_v = self.grid_v[base + offset] + self.dv[base + offset]
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            F = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]
            stress = (-self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(F) @ self.old_F[p].transpose()

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                force = weight * stress @ dpos
                # self.residual[self.idx(base + offset)] += self.dt * force
                self.residual[base + offset] += self.dt * force
        self.project(self.residual)
        
        # Copy to the residual holder
        for I in ti.grouped(residual):
            residual[I] = self.residual[I]


    @ti.kernel
    def grid_to_particles(self):
        for p in self.x:  # grid to particle (G2P)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.real)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.v[p])
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):  # loop over 3x3 grid node neighborhood
                dpos = offset.cast(self.real) - fx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                g_v = self.grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MPM')
    parser.add_argument('--implicit', action='store_true',
                        help='implicit or not')
    args = parser.parse_args()

    gui = ti.GUI("Taichi MLS-MPM", res=512, background_color=0x112F41)

    linear_solver = 'gradient_descent'
    dt = 1e-4
    if args.implicit:
        dt = 2e-3

    visualization_limit = dt

    solver = MlsMpmSolver(dt=dt, gravity=0.0, implicit=args.implicit, linear_solver=linear_solver)
    solver.initialize()
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(visualization_limit // solver.dt)):
            solver.advance_one_time_step()
        gui.circles(solver.x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEEF0],
                    palette_indices=solver.material)
        gui.show(
        )  # Change to gui.show(f'{frame:06d}.png') to write images to disk