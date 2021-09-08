import taichi as ti
import numpy as np
from mpm_base import MPMSimulationBase
from gradient_descent import GradientDescentSolver
from conjugate_gradient import ConjugateGradientSolver
from newton_optimization_solver import NewtonSolver
from diff_test import DiffTest

ti.init(arch=ti.cuda)


@ti.data_oriented
class MlsMpmSolver(MPMSimulationBase):
    def __init__(self, dt=1e-4, dim=2, gravity=9.8,
                 gravity_dim=1,
                 implicit=False,
                 optimization_solver=None,
                 diff_test=False):
        super(MlsMpmSolver, self).__init__(implicit=implicit)
        self.step = 0
        self.dim = dim
        self.real = ti.f32
        self.quality = 1  # Use a larger value for higher-res simulations
        self.ignore_collision = False
        self.debug_mode = True
        self.optimization_solver_name, self.optimization_solver = optimization_solver
        self.diff_test = diff_test
        self.n_particles, self.n_grid = 3000 * self.quality ** self.dim, 64 * self.quality
        print(f"Particle Number: {self.n_particles}, n_grid: {self.n_grid}")
        self.grid_shape = (self.n_grid,) * self.dim
        self.n_nodes = self.n_grid ** self.dim
        self.bound = 3  # boundary of the simulation domain
        self.neighbour = (3,) * self.dim
        self.group_size = self.n_particles // 2
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.dt = dt / self.quality
        self.cfl_limit = 0.6

        self.p_vol, self.p_rho = (self.dx * 0.5) ** self.dim, 1
        self.p_mass = self.p_vol * self.p_rho
        self.E, self.nu = 1e3, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
                (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

        self.gravity = ti.Vector([(-gravity if i == gravity_dim else 0) for i in range(dim)])

        # Max velocity for CFL checking
        self.max_velocity = ti.field(dtype=self.real, shape=())
        # Quantities defined on particles
        self.x = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)  # position
        self.v = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)  # velocity
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                                 shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                                 shape=self.n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=self.n_particles)  # material id
        self.colors = ti.Vector.field(3, dtype=float, shape=self.n_particles)
        self.Jp = ti.field(dtype=self.real, shape=self.n_particles)  # plastic deformation

        # Quantities defined on grid
        self.grid_v = ti.Vector.field(self.dim, dtype=self.real,
                                      shape=self.grid_shape)  # grid node momentum/velocity

        self.grid_v_new = ti.Vector.field(self.dim, dtype=self.real,
                                          shape=self.grid_shape)  # new grid node momentum, also used for store forces temporally

        self.grid_m = ti.field(dtype=self.real, shape=self.grid_shape)  # grid node mass

        if implicit:
            assert optimization_solver
            self.old_F = ti.Matrix.field(self.dim, self.dim, dtype=self.real,
                                         shape=self.n_particles)  # backup deformation gradient

            # # Quantities for linear solver
            # self.mass_matrix = ti.field(dtype=self.real, shape=self.n_nodes)
            # self.dv = ti.Vector.field(dim, dtype=self.real, shape=self.n_nodes)  # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
            # self.residual = ti.Vector.field(dim, dtype=self.real, shape=self.n_nodes)

            # Quantities for linear solver
            self.mass_matrix = ti.field(dtype=self.real, shape=self.grid_shape)
            self.dv = ti.Vector.field(dim, dtype=self.real,
                                      shape=self.grid_shape)  # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
            # self.residual = ti.Vector.field(dim, dtype=self.real, shape=self.grid_shape)

            if self.diff_test:
                # Define a diff test object
                self.diff_test = DiffTest(self.dim, self.dv, self.n_particles,
                                          self.total_energy,
                                          self.compute_energy_gradient,
                                          self.update_state,
                                          self.multiply,
                                          dtype=self.real,
                                          is_test_hessian=True)
            self.is_difftest_done = False

            # scratch data for calculate differential of F
            self.scratch_xp = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)
            self.scratch_vp = ti.Vector.field(self.dim, dtype=self.real, shape=self.n_particles)
            self.scratch_gradV = ti.Matrix.field(self.dim, self.dim, dtype=self.real, shape=self.n_particles)
            self.scratch_stress = ti.Matrix.field(self.dim, self.dim, dtype=self.real, shape=self.n_particles)

            # These should be updated everytime a new SVD is performed to F
            # if ti.static(dim == 2):
            self.psi0 = ti.field(dtype=self.real, shape=self.n_particles)  # d_PsiHat_d_sigma0
            self.psi1 = ti.field(dtype=self.real, shape=self.n_particles)  # d_PsiHat_d_sigma1
            self.psi00 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01 = ti.field(dtype=self.real,
                                shape=self.n_particles)  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01 = ti.field(dtype=self.real,
                                shape=self.n_particles)  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij = ti.Matrix.field(dim, dim, dtype=self.real, shape=self.n_particles)
            self.B01 = ti.Matrix.field(2, 2, dtype=self.real, shape=self.n_particles)
            if ti.static(dim == 3):
                self.psi2 = ti.field(dtype=self.real, shape=self.n_particles)  # d_PsiHat_d_sigma2
                self.psi22 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma2_d_sigma2
                self.psi02 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma0_d_sigma2
                self.psi12 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma1_d_sigma2

                self.m02 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
                self.p02 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
                self.m12 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
                self.p12 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
                # self.Aij = ti.Matrix.field(dim, dim, dtype=self.real, shape=self.n_particles)
                # self.B01 = ti.Matrix.field(2, 2, dtype=self.real, shape=self.n_particles)
                self.B12 = ti.Matrix.field(2, 2, dtype=self.real, shape=self.n_particles)
                self.B20 = ti.Matrix.field(2, 2, dtype=self.real, shape=self.n_particles)

    def initialize(self):
        if self.dim == 3:
            self.simulation_initialize_3D()
        if self.dim == 2:
            self.simulation_initialize_2D()
        functions_dict = {"multiply": self.multiply,
                          "compute_residual": self.compute_energy_gradient,
                          "update_simulation_state": self.update_state,
                          "project": self.project_kernel}
        self.optimization_solver.initialize(self.dim, self.grid_shape, functions_dict, dtype=self.real)
    
    def check_cfl_condition(self):
        # Check the CFL condition
        cfl_dt = self.cfl_limit / self.n_grid / self.max_velocity[None] / self.dim
        if self.dt > cfl_dt:
            self.dt = cfl_dt
            print(f"CFL required dt: {cfl_dt}, current dt: {self.dt}, max velocity: {self.max_velocity[None]} ")

    @ti.kernel
    def simulation_initialize_2D(self):
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * 0.2 + 0.3, #+ 0.10 * (i // self.group_size),
                ti.random() * 0.2 + 0.05 + 0.4 * (i // self.group_size)
            ]
            self.material[i] = i // self.group_size  # 0: fluid 1: jelly 2: snow
            self.v[i] = ti.Matrix([0 for _ in range(self.dim)])
            if self.material[i] == 0:
                self.v[i] = ti.Matrix([0, 2.0])
                self.colors[i] = (0, 0.5, 0.5)
            if self.material[i] == 1:
                self.v[i] = ti.Matrix([0, -2.0])
                self.colors[i] = (0.93, 0.33, 0.23)
            if self.material[i] == 2:
                self.v[i] = ti.Matrix([0, -3.0])
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            self.max_velocity[None] = 2.0

    @ti.kernel
    def simulation_initialize_3D(self):
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // self.group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // self.group_size),
                ti.random() * 0.2 + 0.05
                ]
            self.material[i] = i // self.group_size  # 0: fluid 1: jelly 2: snow
            self.v[i] = ti.Matrix([0 for _ in range(self.dim)])
            if self.material[i] == 0:
                self.v[i] = ti.Matrix([0, -2.0, 0.0])
                self.colors[i] = (0, 0.5, 0.5)
            if self.material[i] == 1:
                self.v[i] = ti.Matrix([0, -2.0, 0.0])
                self.colors[i] = (0.93, 0.33, 0.23)
            if self.material[i] == 2:
                self.v[i] = ti.Matrix([0, -2.0, 0.0])
            self.F[i] = ti.Matrix([[1 * 1.01, 0, 0], [0, 1 * 1.01, 0], [0, 0, 1 * 1.01]])
            self.Jp[i] = 1
            self.max_velocity[None] = 2.0

    @ti.kernel
    def reinitialize(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.zero(self.grid_v[I])
            self.grid_v_new[I] = ti.zero(self.grid_v[I])
            self.grid_m[I] = 0

    @ti.kernel
    def particles_to_grid(self):
        for p in self.x:  # Particle state update and scatter to grid (P2G)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.real)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.F[p] = (ti.Matrix.identity(self.real, self.dim) +
                         self.dt * self.C[p]) @ self.F[p]  # deformation gradient update
            h = ti.exp(
                10 *
                (1.0 -
                 self.Jp[p]))  # Hardening coefficient: snow gets harder when compressed

            # JELLY ONLY
            if self.material[p] == 1 or self.material[p] == 0:  # jelly, make it softer
                h = ti.cast(0.3, self.real)

            mu, la = self.mu_0 * h, self.lambda_0 * h

            # JELLY ONLY
            # if self.material[p] == 0:  # liquid
            #     mu = 0.0

            U, sig, V = ti.svd(self.F[p], self.real)
            J = ti.cast(1.0, self.real)
            for d in ti.static(range(self.dim)):
                new_sig = sig[d, d]
                if self.material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig

            # JELLY ONLY
            # if self.material[
            #     p] == 0:  # Reset deformation gradient to avoid numerical instability
            #     self.F[p] = ti.Matrix.identity(self.real, 2) * ti.sqrt(J)
            # elif self.material[p] == 2:
            #     self.F[p] = U @ sig @ V.transpose(
            #     )  # Reconstruct elastic deformation gradient after plasticity

            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
            ) + ti.Matrix.identity(self.real, self.dim) * la * J * (J - 1)
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
                    if i < self.bound and self.grid_v[I][0] < 0:
                        self.grid_v[I][0] = 0  # Boundary conditions
                    if i > self.n_grid - self.bound and self.grid_v[I][0] > 0:
                        self.grid_v[I][0] = 0
                    if j < self.bound and self.grid_v[I][1] < 0:
                        self.grid_v[I][1] = 0
                    if j > self.n_grid - self.bound and self.grid_v[I][1] > 0:
                        self.grid_v[I][1] = 0
                if self.dim == 3:
                    i, j, k = I[0], I[1], I[2]
                    if i < self.bound and self.grid_v[I][0] < 0:
                        self.grid_v[I][0] = 0  # Boundary conditions
                    if i > self.n_grid - self.bound and self.grid_v[I][0] > 0:
                        self.grid_v[I][0] = 0
                    if j < self.bound and self.grid_v[I][1] < 0:
                        self.grid_v[I][1] = 0
                    if j > self.n_grid - self.bound and self.grid_v[I][1] > 0:
                        self.grid_v[I][1] = 0
                    if k < self.bound and self.grid_v[I][2] < 0:
                        self.grid_v[I][2] = 0
                    if k > self.n_grid - self.bound and self.grid_v[I][2] > 0:
                        self.grid_v[I][2] = 0

    def backward_euler(self):
        self.build_mass_matrix()
        self.build_initial_dv_for_newton()
        # Which should be called at the beginning of newton.
        self.backup_strain()

        if self.optimization_solver_name == "gradient_descent":
            self.gradient_descent_solve()
        elif self.optimization_solver_name == "newton":
            self.newton_solve()

        self.restore_strain()
        self.construct_new_velocity_from_newton_result()
        self.step += 1

    @ti.func
    def psi(self, F):  # strain energy density function Ψ(F)
        # The Material Point Method for the Physics-based of Solids and Fluids, Page: 20, Eqn: 49
        U, sig, V = ti.svd(F, self.real)
        # fixed corotated model, you can replace it with any constitutive model
        return self.mu_0 * (F - U @ V.transpose()).norm() ** 2 + self.lambda_0 / 2 * (F.determinant() - 1) ** 2

    @ti.func
    def dpsi_dF(self, F):  # first Piola-Kirchoff stress P(F), i.e. ∂Ψ/∂F
        # The Material Point Method for the Physics-based of Solids and Fluids, Page: 20, Eqn: 52
        U, sig, V = ti.svd(F, self.real)
        J = F.determinant()
        R = U @ V.transpose()
        return 2 * self.mu_0 * (F - R) + self.lambda_0 * (J - 1) * J * F.inverse().transpose()

    @ti.func
    def first_piola_differential(self, p, F, dF):
        # The Material Point Method for the Physics-based of Solids and Fluids, Page: 22, Eqn: 69
        U, sig, V = ti.svd(F, self.real)
        D = U.transpose() @ dF @ V
        K = ti.Matrix.zero(self.real, self.dim, self.dim)
        self.dPdF_of_sigma_contract(p, D, K)
        return U @ K @ V.transpose()

    @ti.func
    def compute_stress_differential(self, p, grad_dv: ti.template(), dstress: ti.template(), dvp: ti.template()):
        Fn_local = self.old_F[p]
        dP = self.first_piola_differential(p, self.F[p], self.dt * grad_dv @ Fn_local)
        dstress += self.p_vol * dP @ Fn_local.transpose()

    # B = dPdF(Sigma) : A
    @ti.func
    def dPdF_of_sigma_contract(self, p, A, B: ti.template()):
        if ti.static(self.dim == 2):
            # B[0, 0] = self.psi00[p] * A[0, 0] + self.psi01[p] * A[1, 1]
            # B[1, 1] = self.psi01[p] * A[0, 0] + self.psi11[p] * A[1, 1]
            # B[0, 1] = ((self.m01[p] + self.p01[p]) * A[0, 1] + (self.m01[p] - self.p01[p]) * A[1, 0]) * 0.5
            # B[1, 0] = ((self.m01[p] - self.p01[p]) * A[0, 1] + (self.m01[p] + self.p01[p]) * A[1, 0]) * 0.5
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
        elif ti.static(self.dim == 3):
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1] + self.Aij[p][0, 2] * A[2, 2]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1] + self.Aij[p][1, 2] * A[2, 2]
            B[2, 2] = self.Aij[p][2, 0] * A[0, 0] + self.Aij[p][2, 1] * A[1, 1] + self.Aij[p][2, 2] * A[2, 2]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
            B[0, 2] = self.B20[p][0, 0] * A[0, 2] + self.B20[p][0, 1] * A[2, 0]
            B[2, 0] = self.B20[p][1, 0] * A[0, 2] + self.B20[p][1, 1] * A[2, 0]
            B[1, 2] = self.B12[p][0, 0] * A[1, 2] + self.B12[p][0, 1] * A[2, 1]
            B[2, 1] = self.B12[p][1, 0] * A[1, 2] + self.B12[p][1, 1] * A[2, 1]

    @ti.func
    def reinitialize_isotropic_helper(self, p):
        if ti.static(self.dim == 2):
            self.psi0[p] = 0.  # d_PsiHat_d_sigma0
            self.psi1[p] = 0.  # d_PsiHat_d_sigma1
            self.psi00[p] = 0.  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01[p] = 0.  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11[p] = 0.  # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01[p] = 0.  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0.  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
        if ti.static(self.dim == 3):
            self.psi0[p] = 0.  # d_PsiHat_d_sigma0
            self.psi1[p] = 0.  # d_PsiHat_d_sigma1
            self.psi2[p] = 0.  # d_PsiHat_d_sigma2
            self.psi00[p] = 0.  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi11[p] = 0.  # d^2_PsiHat_d_sigma1_d_sigma1
            self.psi22[p] = 0.  # d^2_PsiHat_d_sigma2_d_sigma2
            self.psi01[p] = 0.  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi02[p] = 0.  # d^2_PsiHat_d_sigma0_d_sigma2
            self.psi12[p] = 0.  # d^2_PsiHat_d_sigma1_d_sigma2

            self.m01[p] = 0.  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0.  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.m02[p] = 0.  # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
            self.p02[p] = 0.  # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
            self.m12[p] = 0.  # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
            self.p12[p] = 0.  # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
            self.B12[p] = ti.zero(self.B12[p])
            self.B20[p] = ti.zero(self.B20[p])

    @ti.func
    def clamp_small_magnitude(self, x, eps):
        result = ti.cast(0, self.real)
        if x < -eps:
            result = x
        elif x < 0:
            result = -eps
        elif x < eps:
            result = eps
        else:
            result = x
        return result

    @ti.func
    def update_isotropic_helper(self, p, F):
        self.reinitialize_isotropic_helper(p)
        if ti.static(self.dim == 2):
            U, sigma, V = ti.svd(F, self.real)
            J = sigma[0, 0] * sigma[1, 1]
            _2mu = self.mu_0 * 2
            _lambda = self.lambda_0 * (J - 1)
            Sprod = ti.Vector([sigma[1, 1], sigma[0, 0]])
            self.psi0[p] = _2mu * (sigma[0, 0] - 1) + _lambda * Sprod[0]
            self.psi1[p] = _2mu * (sigma[1, 1] - 1) + _lambda * Sprod[1]
            self.psi00[p] = _2mu + self.lambda_0 * Sprod[0] * Sprod[0]
            self.psi11[p] = _2mu + self.lambda_0 * Sprod[1] * Sprod[1]
            self.psi01[p] = _lambda + self.lambda_0 * Sprod[0] * Sprod[1]

            # (psi0-psi1)/(sigma0-sigma1)
            self.m01[p] = _2mu - _lambda

            # (psi0+psi1)/(sigma0+sigma1)
            self.p01[p] = (self.psi0[p] + self.psi1[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[1, 1], 1e-6)

            self.Aij[p] = ti.Matrix(
                [[self.psi00[p], self.psi01[p]],
                 [self.psi01[p], self.psi11[p]]])
            self.B01[p] = ti.Matrix(
                [[(self.m01[p] + self.p01[p]) * 0.5, (self.m01[p] - self.p01[p]) * 0.5],
                 [(self.m01[p] - self.p01[p]) * 0.5, (self.m01[p] + self.p01[p]) * 0.5]])

        elif ti.static(self.dim == 3):
            U, sigma, V = ti.svd(F)
            J = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]
            _2mu = self.mu_0 * 2
            _lambda = self.lambda_0 * (J - 1)
            Sprod = ti.Vector([sigma[1, 1] * sigma[2, 2], sigma[0, 0] * sigma[2, 2], sigma[0, 0] * sigma[1, 1]])
            self.psi0[p] = _2mu * (sigma[0, 0] - 1) + _lambda * Sprod[0]
            self.psi1[p] = _2mu * (sigma[1, 1] - 1) + _lambda * Sprod[1]
            self.psi2[p] = _2mu * (sigma[2, 2] - 1) + _lambda * Sprod[2]
            self.psi00[p] = _2mu + self.lambda_0 * Sprod[0] * Sprod[0]
            self.psi11[p] = _2mu + self.lambda_0 * Sprod[1] * Sprod[1]
            self.psi22[p] = _2mu + self.lambda_0 * Sprod[2] * Sprod[2]
            self.psi01[p] = _lambda * sigma[2, 2] + self.lambda_0 * Sprod[0] * Sprod[1]
            self.psi02[p] = _lambda * sigma[1, 1] + self.lambda_0 * Sprod[0] * Sprod[2]
            self.psi12[p] = _lambda * sigma[0, 0] + self.lambda_0 * Sprod[1] * Sprod[2]

            # (psiA-psiB)/(sigmaA-sigmaB)
            self.m01[p] = _2mu - _lambda * sigma[2, 2]  # i[p] = 0
            self.m02[p] = _2mu - _lambda * sigma[1, 1]  # i[p] = 2
            self.m12[p] = _2mu - _lambda * sigma[0, 0]  # i[p] = 1

            # (psiA+psiB)/(sigmaA+sigmaB)
            self.p01[p] = (self.psi0[p] + self.psi1[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[1, 1], 1e-6)
            self.p02[p] = (self.psi0[p] + self.psi2[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[2, 2], 1e-6)
            self.p12[p] = (self.psi1[p] + self.psi2[p]) / self.clamp_small_magnitude(sigma[1, 1] + sigma[2, 2], 1e-6)

            self.Aij[p] = ti.Matrix([
                [self.psi00[p], self.psi01[p], self.psi02[p]],
                [self.psi01[p], self.psi11[p], self.psi12[p]],
                [self.psi02[p], self.psi12[p], self.psi22[p]]])
            self.B01[p] = ti.Matrix([
                [(self.m01[p] + self.p01[p]) * 0.5, (self.m01[p] - self.p01[p]) * 0.5],
                [(self.m01[p] - self.p01[p]) * 0.5, (self.m01[p] + self.p01[p]) * 0.5]])
            self.B12[p] = ti.Matrix([
                [(self.m12[p] + self.p12[p]) * 0.5, (self.m12[p] - self.p12[p]) * 0.5],
                [(self.m12[p] - self.p12[p]) * 0.5, (self.m12[p] + self.p12[p]) * 0.5]])
            self.B20[p] = ti.Matrix([
                [(self.m02[p] + self.p02[p]) * 0.5, (self.m02[p] - self.p02[p]) * 0.5],
                [(self.m02[p] - self.p02[p]) * 0.5, (self.m02[p] + self.p02[p]) * 0.5]])


    @ti.kernel
    def total_energy(self) -> ti.f64:
        result = ti.cast(0.0, self.real)
        # elastic potential energy
        for p in self.F:
            result += self.psi(self.F[p]) * self.p_vol  # gathered from particles, psi defined in the rest space

        # inertia energy
        for I in ti.grouped(self.dv):
            m = self.mass_matrix[I]
            dv = self.dv[I]
            result += m * dv.dot(dv) / 2

        # gravity potential
        for I in ti.grouped(self.dv):
            m = self.mass_matrix[I]
            for i in ti.static(range(self.dim)):
                result -= self.dt * m * self.gravity[i]

        return result

    @ti.kernel
    def update_state(self, dv: ti.template()):
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
                g_v = self.grid_v[base + offset] + dv[base + offset]
                new_C += 4 * self.inv_dx * self.inv_dx * weight * g_v.outer_product(dpos)

            self.F[p] = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]
            self.update_isotropic_helper(p, self.F[p])

    @ti.kernel
    def build_initial_dv_for_newton(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                # node_id = self.idx(I)
                node_id = I
                if ti.static(not self.ignore_collision):
                    cond = (I < self.bound and self.grid_v[I] < 0) or (
                            I > self.n_grid - self.bound and self.grid_v[I] > 0)
                    self.dv[node_id] = -self.grid_v[I] if cond else self.gravity * self.dt
                else:
                    self.dv[node_id] = self.gravity * self.dt  # Newton initial guess for non-collided nodes

    @ti.kernel
    def construct_new_velocity_from_newton_result(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] += self.dv[I]
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

    def run_difftest(self):
        if self.diff_test and not self.is_difftest_done:
            self.diff_test.run(self.dv, self.F)
            self.is_difftest_done = True

    def gradient_descent_solve(self):
        self.optimization_solver.solve(self.compute_energy_gradient, self.dv)
        self.run_difftest()

    def newton_solve(self):
        self.optimization_solver.solve(self.dv, preconditioner=self.mass_matrix)
        self.run_difftest()

    @ti.func
    def compute_dv_and_grad_dv(self, dv: ti.template()):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            vp = ti.zero(self.scratch_vp[p])
            gradV = ti.zero(self.scratch_gradV[p])

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                dv0 = dv[base + offset]
                vp += weight * dv0
                gradV += 4 * self.inv_dx * self.inv_dx * weight * dv0.outer_product(dpos)

            self.scratch_vp[p] = vp
            self.scratch_gradV[p] = gradV

    @ti.kernel
    def multiply(self, x: ti.template(), b: ti.template()):
        for I in ti.grouped(b):
            b[I] = ti.zero(b[I])

        # Note the relationship H dx = - df, where H is the stiffness matrix
        # inertia part
        for I in ti.grouped(x):
            b[I] += self.mass_matrix[I] * x[I]

        self.compute_dv_and_grad_dv(x)

        # scratch_gradV is now temporaraly used for storing gradDV (evaluated at particles)
        # scratch_vp is now temporaraly used for storing DV (evaluated at particles)

        for p in self.x:
            self.scratch_stress[p] = ti.zero(self.scratch_stress[p])

        for p in self.x:
            self.compute_stress_differential(p, self.scratch_gradV[p], self.scratch_stress[p], self.scratch_vp[p])
            # scratch_stress is now V_p^0 dP (F_p^n)^T (dP is Ap in snow paper)

        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (
                    fx - 0.5) ** 2]  # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            stress = self.scratch_stress[p]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                b[base + offset] += 4 * self.inv_dx * self.inv_dx * self.dt * (weight * stress @ dpos)
                # fi -= \sum_p (Ap (xi-xp)  - fp )w_ip Dp_inv
                # fp: mesh force, not applied here; Ap: dp; Dp_inv: inv_dx

        # Handle boundary values
        self.project(b)

    @ti.func
    def project(self, x: ti.template()):
        for I in ti.grouped(x):
            # I = self.node(p)
            # I = p
            cond = any(I < self.bound and self.grid_v[I] < 0) or any(
                I > self.n_grid - self.bound and self.grid_v[I] > 0)
            if cond:
                x[I] = ti.zero(x[I])

    @ti.kernel
    def project_kernel(self, x: ti.template()):
        self.project(x)

    @ti.kernel
    def compute_energy_gradient(self, residual: ti.template()):

        for I in ti.grouped(self.dv):
            residual[I] = [0.0 for i in range(self.dim)]

        # Compute the RHS (i.e. usually energy gradient) of the linear system
        for I in ti.grouped(self.dv):
            residual[I] = self.dt * self.mass_matrix[I] * self.gravity
        #
        # inertia part
        for I in ti.grouped(self.dv):
            residual[I] -= self.mass_matrix[I] * self.dv[I]

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

                g_v = self.grid_v[base + offset] + self.dv[base + offset]
                new_C += 4 * self.inv_dx * self.inv_dx * weight * g_v.outer_product(dpos)

            F = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]
            stress = (-self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(F) @ self.old_F[p].transpose()

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                force = weight * stress @ dpos
                residual[base + offset] += self.dt * force
        # Handle boundary values
        self.project(residual)

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
            for idx_v in ti.static(range(self.dim)):
                if new_v[idx_v] > self.max_velocity[None]:
                    self.max_velocity[None] = new_v[idx_v]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MPM')
    parser.add_argument('--dim', type=int,
                        default=2,
                        help='dimension')
    parser.add_argument('--implicit', action='store_true',
                        help='implicit or not')
    parser.add_argument('--difftest', action='store_true',
                        help='do difftest or not')
    parser.add_argument('--gradient_descent', action='store_true',
                        help='do gradient descent or not')
    args = parser.parse_args()

    dim = args.dim
    assert dim == 2 or dim == 3

    # optimization_solver_type = 'gradient_descent'
    optimization_solver_type = 'newton'
    if args.gradient_descent:
        optimization_solver_type = 'gradient_descent'
    dt = 1e-4
    if args.implicit:
        dt = 1e-3

    visualization_limit = dt
    optimization_solver = None
    if optimization_solver_type == 'gradient_descent':
        optimization_solver = (
        optimization_solver_type, GradientDescentSolver(max_iterations=1000, adaptive_step_size=False))
    elif optimization_solver_type == 'newton':
        optimization_solver = (optimization_solver_type, NewtonSolver(max_iterations=20, tolerance=1e-6))

    solver = MlsMpmSolver(dt=dt,
                          dim=dim,
                          gravity=9.8,
                          implicit=args.implicit,
                          optimization_solver=optimization_solver,
                          diff_test=args.difftest)
    solver.initialize()

    # gui = ti.GUI("Taichi MLS-MPM", res=512, background_color=0x112F41)
    # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    #     for s in range(int(visualization_limit // solver.dt)):
    #         solver.advance_one_time_step()
    #     gui.circles(solver.x.to_numpy(),
    #                 radius=1.5,
    #                 palette=[0x068587, 0xED553B, 0xEEEEF0],
    #                 palette_indices=solver.material)
    #     gui.show(
    #     )  # Change to gui.show(f'{frame:06d}.png') to write images to disk

    if dim == 2:

        # gui = ti.GUI("Taichi MLS-MPM", res=512, background_color=0x112F41)
        # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        #     for s in range(int(visualization_limit // solver.dt)):
        #         solver.advance_one_time_step()
        #     gui.circles(solver.x.to_numpy(),
        #                 radius=1.5,
        #                 palette=[0x068587, 0xED553B, 0xEEEEF0],
        #                 palette_indices=solver.material)
        #     gui.show(f'{solver.step:06d}.png')  # Change to gui.show(f'{frame:06d}.png') to write images to disk
        #
        window = ti.ui.Window('Taichi MLS-MPM', (512, 512))
        canvas = window.get_canvas()

        while window.running:
            for s in range(int(visualization_limit // solver.dt)):
                solver.advance_one_time_step()
            canvas.set_background_color((0.067, 0.184, 0.255))
            canvas.circles(solver.x,
                           radius=0.0025,
                           per_vertex_color=solver.colors)
            window.show()
    elif dim == 3:
        res = (1024, 1024)
        window = ti.ui.Window("MLS-MPM 3D", res, vsync=True)

        # Set up
        frame_id = 0
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
        camera.position(0.5, 1.0, 1.95)
        camera.lookat(0.5, 0.3, 0.5)
        camera.fov(55)

        while window.running:
            for s in range(int(visualization_limit // solver.dt)):
                solver.advance_one_time_step()
            # Render
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            scene.ambient_light((0, 0, 0))

            scene.particles(solver.x, per_vertex_color=solver.colors, radius=0.002)
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

            canvas.scene(scene)

            window.show()

