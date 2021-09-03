import taichi as ti
from conjugate_gradient import ConjugateGradientSolver


class NewtonSolver:
    def __init__(self, dim, shape, dtype=ti.f32, max_iterations=10, tolerance=1e-3):
        self.dtype = dtype
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_direction = ti.Vector.field(dim, dtype=dtype, shape=shape)
        # Define a CG solver
        self.linear_solver = ConjugateGradientSolver(dim=dim, shape=shape)

        # Simulation specific functions
        self.multiply = None
        self.compute_residual = None
        self.update_simulation_state = None
        
    def initialize(self, multiply, compute_residual, update_simulation_state):
        self.multiply = multiply
        self.linear_solver.initialize(multiply)
        self.compute_residual = compute_residual
        self.update_simulation_state = update_simulation_state
    
    @ti.kernel
    def compute_residual_norm(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].dot(x[I])
        return ti.sqrt(result)

    def liner_solve(self, x, b):
        self.linear_solver.solve(x, b, self.multiply)

    @ti.kernel
    def update(self, dst: ti.template(), src: ti.template(), scale: ti.f32):
        for I in ti.grouped(src):
            dst[I] += scale * src[I]

    def solve(self, x, b):

        assert self.multiply is not None
        assert self.compute_residual is not None
        assert self.update_simulation_state is not None

        for n in range(self.max_iterations):
            # Compute RHS
            self.compute_residual(b)
            
            residual_norm = self.compute_residual_norm(b)
            if residual_norm < self.tolerance:
                print(f'\033[1;36m [Newton] Terminated at iter = {n}, Residual Norm = {residual_norm} \033[0m')
                break
            self.linear_sovle(self.step_direction, b)
            self.update_step(x, self.step_direction, 1.0)
            self.update_simulation_state()
