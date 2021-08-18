import taichi as ti


@ti.data_oriented
class ImplicitMPMSovler:
    def __init__(self, dim, dt, n_particles, n_grid, p_rho, E, nu, gravity=9.8, gravity_dim=1, cfl=0.4, debug_mode=True):
        self.debug_mode = debug_mode
        # self.category = category
        self.dim = dim
        self.dt = dt
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1 / n_grid
        self.inv_dx = float(n_grid)
        self.p_rho = p_rho
        self.p_vol = (self.dx * 0.5) ** self.dim
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = ti.Vector([(-gravity if i == gravity_dim else 0) for i in range(dim)])

        self.E, self.nu = E, nu  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.cfl = cfl
        # self.ppc = ppc

        real = ti.f32
        self.real = real
        self.neighbour = (3,) * dim
        self.bound = 3
        self.n_nodes = self.n_grid ** dim

        # self.ignore_collision = ignore_collision

        self.x = ti.Vector.field(dim, dtype=real, shape=n_particles)  # position
        self.v = ti.Vector.field(dim, dtype=real, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=real, shape=n_particles)  # affine velocity matrix
        self.F = ti.Matrix.field(dim, dim, dtype=real, shape=n_particles)  # deformation gradient, i.e. strain
        self.old_F = ti.Matrix.field(dim, dim, dtype=real, shape=n_particles)  # for backup/restore F

        self.grid_v = ti.Vector.field(dim, dtype = real) # grid node momentum/velocity
        self.grid_m = ti.field(dtype = real) # grid node mass

        block_size = 16
        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.n_grid // block_size])
        self.grid.dense(
            indices, block_size).place(self.grid_v, self.grid_m)

    def reinitialize(self):
        pass

    @ti.kernel
    def particles_to_grid(self):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]  # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            affine = self.p_mass * self.C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I] # momentum to velocity

    def backward_euler_step(self):
        pass

    @ti.kernel
    def grid_to_particles(self):
        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            new_V = ti.zero(self.v[p])
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                # weight = self.real(1)
                weight = ti.cast(1.0, self.real)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                g_v = self.grid_v[base + offset]
                new_V += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p] = new_V
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(self.real, self.dim) + self.dt * self.C[p]) @ self.F[p] # F' = (I+dt * grad v)F
            # self.updateIsotropicHelper(p, self.F[p])
            self.x[p] += self.dt * self.v[p]


    def substep(self):
        self.reinitialize()
        self.particles_to_grid()
        self.backward_euler_step()
        self.grid_to_particles()

ti.init()
colors = ti.field(int, 1000)
@ti.kernel
def init(solver : ti.template()):
    '''
    for i in range(solver.n_particles / 2):
        solver.x[i] = ti.Vector([ti.random() for i in range(solver.dim)]) * 0.25 + 0.25
        solver.v[i] = ti.Vector([0, -6])
        solver.F[i] = ti.Matrix.identity(solver.real, solver.dim)
        colors[i] = 0x66ccff
    for i in range(solver.n_particles / 2, solver.n_particles):
        solver.x[i] = ti.Vector([ti.random() for i in range(solver.dim)]) * 0.25 + [0.45, 0.65]
        solver.v[i] = ti.Vector([0, -20])
        solver.F[i] = ti.Matrix.identity(solver.real, solver.dim)
        colors[i] = 0xED553B
    '''

    for i in range(solver.n_particles):
        solver.x[i] = ti.Vector([ti.random() for i in range(solver.dim)]) * 0.4 + 0.35
        solver.v[i] = ti.Vector([0, 0.0])
        solver.F[i] = ti.Matrix.identity(solver.real, solver.dim)
        colors[i] = 0xED553B


if __name__ == '__main__':

    solver = ImplicitMPMSovler(dim=2, dt=4e-5, n_particles=1000, n_grid=128, p_rho=1.0, E=50000, nu=0.4)
    init(solver)

    gui = ti.GUI("Taichi IMPLICIT-MPM", res=512, background_color=0x112F41)
    frame = 0
    while gui.running:
        for i in range(10):
            print('[new step], frame = ', frame, ', substep = ', i + 1)
            solver.substep()

        pos = solver.x.to_numpy()
        gui.circles(pos, radius=1.5, color=colors.to_numpy())
        # gui.show(f'{frame:06d}.png')
        gui.show()

        frame += 1