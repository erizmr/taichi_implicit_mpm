import taichi as ti
from conjugate_gradient import ConjugateGradientSolver


@ti.data_oriented
class NewtonSolver:
    def __init__(self, max_iterations=10, tolerance=1e-3):
        self.dtype = ti.f32
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.dim = None
        self.step_direction = None
        self.residual = None
        self.linear_solver = None
        # Simulation specific functions
        self.multiply = None
        self.compute_residual = None
        self.update_simulation_state = None
        self.project = None

    def initialize(self, dim, shape, functions_dict, dtype=ti.f32):
        self.dtype = dtype
        self.dim = dim
        self.step_direction = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.residual = ti.Vector.field(dim, dtype=self.dtype, shape=shape)

        # Get the simulation specific functions
        self.multiply = functions_dict["multiply"]
        self.compute_residual = functions_dict["compute_residual"]
        self.update_simulation_state = functions_dict["update_simulation_state"]
        self.project = functions_dict["project"]

        # Define a CG solver
        self.linear_solver = ConjugateGradientSolver(max_iterations=1000, relative_tolerance=1e-3)
        self.linear_solver.initialize(dim=dim,
                                      shape=shape,
                                      functions_dict={"multiply": self.multiply,
                                                      "project": self.project})

    @ti.kernel
    def compute_residual_norm(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].dot(x[I])
        return ti.sqrt(result)

    def linear_solve(self, x, b, precondtioner):
        self.linear_solver.solve(x, b, precondtioner)

    @ti.kernel
    def update_step(self, dst: ti.template(), src: ti.template(), scale: ti.f32):
        for I in ti.grouped(src):
            dst[I] += scale * src[I]

    @ti.kernel
    def clear_step_direction(self):
        for I in ti.grouped(self.step_direction):
            for d in ti.static(range(self.dim)):
                self.step_direction[I][d] = 0.0

    def solve(self, x, preconditioner):
        assert self.multiply is not None
        assert self.compute_residual is not None
        assert self.update_simulation_state is not None

        self.update_simulation_state(x)
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
            self.update_step(x, self.step_direction, 1.0)
            self.update_simulation_state(x)

            # Set zero initial solution for linear solver
            self.clear_step_direction()
