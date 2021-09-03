import taichi as ti
from mpm_base import MPMSimulationBase
from gradient_descent import GradientDescentSolver
from conjugate_gradient import ConjugateGradientSolver
from diff_test import DiffTest
ti.init(arch=ti.cuda)


@ti.data_oriented
class MlsMpmSolver(MPMSimulationBase):
    def __init__(self, dt=1e-4, dim=2, gravity=9.8, gravity_dim=1,
                 implicit=False,
                 optimization_solver=None,
                 diff_test=False):
        super(MlsMpmSolver, self).__init__(implicit=implicit)
        self.dim = dim
        self.real = ti.f64
        self.quality = 1  # Use a larger value for higher-res simulations
        self.ignore_collision = True
        self.debug_mode = True
        self.optimization_solver = optimization_solver
        self.diff_test = diff_test
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
            assert optimization_solver
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

            if self.diff_test:
                # Define a diff test object
                self.diff_test = DiffTest(self.dim, self.dv, self.n_particles,
                                          self.total_energy,
                                          self.compute_energy_gradient,
                                          self.update_state)
            self.is_difftest_done = False

            # These should be updated everytime a new SVD is performed to F
            if ti.static(dim == 2):
                self.psi0 = ti.field(dtype=self.real, shape=self.n_particles)  # d_PsiHat_d_sigma0
                self.psi1 = ti.field(dtype=self.real, shape=self.n_particles)  # d_PsiHat_d_sigma1
                self.psi00 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma0_d_sigma0
                self.psi01 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma0_d_sigma1
                self.psi11 = ti.field(dtype=self.real, shape=self.n_particles)  # d^2_PsiHat_d_sigma1_d_sigma1
                self.m01 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
                self.p01 = ti.field(dtype=self.real,
                                    shape=self.n_particles)  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6

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
            # self.old_F[i] = ti.Matrix([[1, 0], [0, 1]])
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

            # JELLY ONLY
            if self.material[p] == 1 or self.material[p] == 0:  # jelly, make it softer
                h = ti.cast(0.3, self.real)

            mu, la = self.mu_0 * h, self.lambda_0 * h

            # JELLY ONLY
            # if self.material[p] == 0:  # liquid
            #     mu = 0.0

            U, sig, V = ti.svd(self.F[p], self.real)
            J = ti.cast(1.0, self.real)
            for d in ti.static(range(2)):
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
        # The Material Point Method for the Physics-based of Solids and Fluids, Page: 20, Eqn: 49
        U, sig, V = ti.svd(F, self.real)
        # fixed corotated model, you can replace it with any constitutive model
        return self.mu_0 * (F - U @ V.transpose()).norm() ** 2 + self.lambda_0 / 2 * (F.determinant() - 1) ** 2
        # return self.mu_0 * (sig - ti.Matrix([[1.0, 0.0], [0.0, 1.0]])).norm() ** 2 + self.lambda_0 / 2 * (F.determinant() - 1) ** 2

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
        dP = self.first_piola_differential(p, Fn_local, grad_dv @ Fn_local)
        dstress += self.p_vol * dP @ Fn_local.transpose()

    # B = dPdF(Sigma) : A
    @ti.func
    def dPdF_of_sigma_contract(self, p, A, B: ti.template()):
        if ti.static(self.dim == 2):
            B[0, 0] = self.psi00[p] * A[0, 0] + self.psi01[p] * A[1, 1]
            B[1, 1] = self.psi01[p] * A[0, 0] + self.psi11[p] * A[1, 1]
            B[0, 1] = ((self.m01[p] + self.p01[p]) * A[0, 1] + (self.m01[p] - self.p01[p]) * A[1, 0]) * 0.5
            B[1, 0] = ((self.m01[p] - self.p01[p]) * A[0, 1] + (self.m01[p] + self.p01[p]) * A[1, 0]) * 0.5
        # if ti.static(self.dim == 3):
        #     B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1] + self.Aij[p][0, 2] * A[2, 2]
        #     B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1] + self.Aij[p][1, 2] * A[2, 2]
        #     B[2, 2] = self.Aij[p][2, 0] * A[0, 0] + self.Aij[p][2, 1] * A[1, 1] + self.Aij[p][2, 2] * A[2, 2]
        #     B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
        #     B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
        #     B[0, 2] = self.B20[p][0, 0] * A[0, 2] + self.B20[p][0, 1] * A[2, 0]
        #     B[2, 0] = self.B20[p][1, 0] * A[0, 2] + self.B20[p][1, 1] * A[2, 0]
        #     B[1, 2] = self.B12[p][0, 0] * A[1, 2] + self.B12[p][0, 1] * A[2, 1]
        #     B[2, 1] = self.B12[p][1, 0] * A[1, 2] + self.B12[p][1, 1] * A[2, 1]

    @ti.func
    def reinitialize_isotropic_helper(self, p):
        if ti.static(self.dim == 2):
            self.psi0[p] = 0  # d_PsiHat_d_sigma0
            self.psi1[p] = 0  # d_PsiHat_d_sigma1
            self.psi00[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11[p] = 0  # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01[p] = 0  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])

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
                new_C += 4 * self.inv_dx * self.inv_dx  * weight * g_v.outer_product(dpos)

            self.F[p] = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]

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
    def incremental_update(self, dst: ti.template(), src: ti.template(), rate: ti.f64, step_direction: ti.template()):
        for I in ti.grouped(src):
            dst[I] = src[I] + rate * step_direction[I]

    @ti.kernel
    def compute_norm(self) -> ti.f64:
        norm_sq = ti.cast(0.0, self.real)
        for I in ti.grouped(self.dv):
            mass = self.mass_matrix[I]
            residual = self.residual[I]
            if mass > 0:
                norm_sq += residual.dot(residual) / mass
        return ti.sqrt(norm_sq)

    def gradient_descent_solve(self):
        self.optmization_solver.solve(self.compute_energy_gradient, self.dv, self.residual)
        if self.diff_test and not self.is_difftest_done:
            self.diff_test.run(self.dv, self.F)
            self.is_difftest_done = True

    def newton_solve(self):
        self.sovler.solve(self.step_direction, self.residual)
    
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

                dv0 = dv[self.idx(base + offset)]
                vp += weight * dv0
                gradV += 4 * self.inv_dx * weight * dv0.outer_product(dpos)

            self.scratch_vp[p] = vp
            self.scratch_gradV[p] = gradV

    @ti.kernel
    def multiply(self, x: ti.template(), b: ti.template()):
        for I in b:
            b[I] = ti.zero(b[I])

        # Note the relationship H dx = - df, where H is the stiffness matrix
        # inertia part
        for I in x:
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
                weight = self.real(1)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                b[base + offset] += self.dt * self.dt * (weight * stress @ dpos)
                # fi -= \sum_p (Ap (xi-xp)  - fp )w_ip Dp_inv
                # fp: mesh force, not applied here; Ap: dp; Dp_inv: inv_dx

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
        
        # for I in ti.grouped(self.dv):
        #     self.residual[I] = [0.0, 0.0]
        
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
    parser.add_argument('--difftest', action='store_true',
                        help='implicit or not')
    args = parser.parse_args()

    gui = ti.GUI("Taichi MLS-MPM", res=512, background_color=0x112F41)

    optimization_solver_type = 'gradient_descent'
    dt = 1e-4
    if args.implicit:
        dt = 4e-3

    visualization_limit = dt
    optimization_solver = None
    if optimization_solver_type == 'gradient_descent':
        optimization_solver = GradientDescentSolver(max_iterations=15, adaptive_step_size=False)
    elif optmization_solver_type == 'conjugate_gradient':
        optimization_solver = ConjugateGradientSolver
        
    solver = MlsMpmSolver(dt=dt, 
                          gravity=9.8, 
                          implicit=args.implicit, 
                          optimization_solver=linear_solver, 
                          diff_test=args.difftest)
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