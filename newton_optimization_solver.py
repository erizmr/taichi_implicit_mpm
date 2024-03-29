import taichi as ti
from conjugate_gradient import ConjugateGradientSolver


@ti.data_oriented
class NewtonSolver:
    def __init__(self, line_search=False, line_search_steps=5, max_iterations=10, tolerance=1e-3):
        self.dtype = ti.f32
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.line_search = line_search
        self.line_search_step = line_search_steps
        self.dim = None
        self.step_direction = None
        self.residual = None
        self.linear_solver = None
        self.dv0 = None
        # Simulation specific functions
        self.total_energy = None
        self.multiply = None
        self.compute_residual = None
        self.update_simulation_state = None
        self.project = None
        self.ddv_checker = None

    def initialize(self, dim, shape, functions_dict, dtype=ti.f32):
        self.dtype = dtype
        self.dim = dim
        self.step_direction = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.residual = ti.Vector.field(dim, dtype=self.dtype, shape=shape)

        # Get the simulation specific functions
        self.total_energy = functions_dict["total_energy"]
        self.multiply = functions_dict["multiply"]
        self.compute_residual = functions_dict["compute_residual"]
        self.update_simulation_state = functions_dict["update_simulation_state"]
        self.project = functions_dict["project"]
        self.ddv_checker = functions_dict["ddv_checker"]

        cg_tolerance = 1e-6 if dim == 2 else 1e-8
        # Define a CG solver
        self.linear_solver = ConjugateGradientSolver(max_iterations=1000, relative_tolerance=cg_tolerance)
        self.linear_solver.initialize(dim=dim,
                                      shape=shape,
                                      functions_dict={"multiply": self.multiply,
                                                      "project": self.project})

        # Holder for line search
        if self.line_search:
            self.dv0 = ti.Vector.field(dim, dtype=dtype, shape=shape)

    @ti.kernel
    def compute_residual_norm(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].dot(x[I])
        return ti.sqrt(result)

    def linear_solve(self, x, b, precondtioner):
        self.linear_solver.solve(x, b, precondtioner)

    @ti.kernel
    def update_step(self, dst: ti.template(), src: ti.template(), scale: ti.f64):
        for I in ti.grouped(src):
            dst[I] += scale * src[I]

    @ti.kernel
    def update_general(self, dst: ti.template(), src1: ti.template(), src2: ti.template(), scale: ti.f64):
        for I in ti.grouped(dst):
            dst[I] = src1[I] + scale * src2[I]

    @ti.kernel
    def copy(self, dst: ti.template(), src: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = src[I]

    @ti.kernel
    def clear_step_direction(self):
        for I in ti.grouped(self.step_direction):
            for d in ti.static(range(self.dim)):
                self.step_direction[I][d] = 0.0

    def solve(self, x, preconditioner):
        assert self.multiply is not None
        assert self.compute_residual is not None
        assert self.update_simulation_state is not None

        E_0 = 0.0
        if self.line_search:
            self.copy(self.dv0, x)
            E_0 = self.total_energy()
        self.update_simulation_state(x)
        
        # TODO: check self.dv[I] == whether self.grid_v[I], it should be automatic satisfied if CG is correct
        # self.ddv_checker(0, 0)
        for n in range(self.max_iterations):
            # Compute RHS
            self.compute_residual(self.residual)
            
            residual_norm = self.compute_residual_norm(self.residual)
            if residual_norm < self.tolerance:
                print(f'\033[1;36m [Newton] Terminated at iter = {n}, Residual Norm = {residual_norm} \033[0m')
                break
            if n % 1 == 0:
                print(f'\033[1;36m [Newton] Iter = {n}, Residual Norm = {residual_norm} \033[0m')

            self.linear_solve(self.step_direction, self.residual, preconditioner)
            if self.line_search:
                step_size, E = 1.0, 0.0
                for l in range(self.line_search_step):
                    self.update_general(x, self.dv0, self.step_direction, step_size)
                    self.update_simulation_state(x)
                    E = self.total_energy()
                    # if ti.static(self.debug_mode):
                    print(f'\033[1;32m[line search] step={l}, E = {E},  E0 = {E_0} \033[0m')
                    step_size /= 2
                    if E < E_0:
                        break
                E_0 = E
                self.copy(self.dv0, x)
            else:
                self.update_step(x, self.step_direction, 1.0)
                # TODO: check self.dv[I] == whether self.grid_v[I], it should be automatic satisfied if CG is correct
                self.update_simulation_state(x)
            # self.ddv_checker(n, 1)

            # Set zero initial solution for linear solver
            self.clear_step_direction()
