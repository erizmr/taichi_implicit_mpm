import taichi as ti


@ti.data_oriented
class GradientDescentSolver:
    def __init__(self, max_iterations=10, step_size=10.0, tolerance=1e-3, adaptive_step_size=False):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.adaptive_step_size = adaptive_step_size
        self.tolerance = tolerance
        self.x_last_step = None
        self.step_direction_last_step = None

    def initialize(self, dim, shape, functions_dict):
        pass

    def _initialize_buffer(self, dim, shape):
        assert self.adaptive_step_size
        self.step_direction_last_step = ti.Vector.field(dim, dtype=ti.f32)
        self.x_last_step = ti.Vector.field(dim, dtype=ti.f32)
        indices = ti.ijk if dim == 3 else ti.ij
        ti.root.dense(indices, shape).place(self.step_direction_last_step, self.x_last_step)

    @ti.kernel
    def copy_buffer(self, x: ti.template(), step_direction: ti.template()):
        for I in ti.grouped(x):
            self.x_last_step[I] = x[I]
            self.step_direction_last_step[I] = step_direction[I]

    def solve(self, compute_gradient, x, step_direction):
        for n in range(self.max_iterations):
            compute_gradient(step_direction)
            self.update(x, step_direction)
            residual_norm = self.compute_residual_norm(step_direction)
            if (n+1) % 50 == 0:
                print(f'\033[1;36m [Gradient Descent] Iter = {n}, Residual Norm = {residual_norm}, Step size = {self.step_size} \033[0m')
            if residual_norm < self.tolerance:
                print(f'\033[1;36m [Gradient Descent] Terminated at iter = {n}, Residual Norm = {residual_norm} \033[0m')
                break
            if self.adaptive_step_size:
                if self.step_direction_last_step is not None:
                    self.step_size = self._step_size_update(x, step_direction)
                    # print(self.step_size)
                else:
                    self._initialize_buffer(x.n, x.shape)
                self.copy_buffer(x, step_direction)

    @ti.kernel
    def compute_residual_norm(self, step_direction: ti.template()) -> ti.f32:
        residual = 0.0
        for I in ti.grouped(step_direction):
            residual += step_direction[I].dot(step_direction[I])
        residual = ti.sqrt(residual)
        return residual

    @ti.kernel
    def update(self, x: ti.template(), step_direction: ti.template()):
        for I in ti.grouped(x):
            x[I] += self.step_size * step_direction[I]

    @ti.kernel
    def _step_size_update(self, x: ti.template(), step_direction: ti.template()) -> ti.f32:
        # new_step_size = |(x_n - x_n-1)^T(s_n - s_n-1)| / |s_n - s_n-1|^2
        gradient_diff_sum = 0.0
        x_diff_dot_gradient_diff = 0.0
        for I in ti.grouped(x):
            step_direction_diff = ti.Vector([0.0 for _ in range(x.n)])
            for i in ti.static(range(x.n)):
                step_direction_diff[i] = step_direction[I][i] - self.step_direction_last_step[I][i]
            x_diff_dot_gradient_diff += (x[I] - self.x_last_step[I]).dot(step_direction_diff)
            gradient_diff_sum += step_direction_diff.dot(step_direction_diff)
        x_diff_dot_gradient_diff_abs = ti.abs(x_diff_dot_gradient_diff)
        gradient_diff_norm = ti.sqrt(gradient_diff_sum)
        return x_diff_dot_gradient_diff_abs / gradient_diff_norm
