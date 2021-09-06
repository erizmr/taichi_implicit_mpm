import taichi as ti


@ti.data_oriented
class ConjugateGradientSolver:
    def __init__(self, max_iterations=50, relative_tolerance=1e-3):
        self.max_iterations = max_iterations
        self.relative_tolerance = relative_tolerance
        self.r, self.p, self.q, self.Ap, self.sum = None, None, None, None, None
        self.dtype = ti.f32
        self.dim = None
        self.multiply = None
    
    def initialize(self, dim, shape, functions_dict, dtype=ti.f32):
        # Function to compute the Ap in CG
        self.multiply = functions_dict["multiply"]
        self.dim = dim
        
        self.r = ti.Vector.field(dim, shape=shape, dtype=dtype)
        self.p = ti.Vector.field(dim, shape=shape, dtype=dtype)
        self.q = ti.Vector.field(dim, shape=shape, dtype=dtype)
        self.Ap = ti.Vector.field(dim, shape=shape, dtype=dtype)
        self.sum = ti.field(dtype=dtype, shape=())
        
    @ti.kernel
    def reinitialize(self):
        for I in ti.grouped(self.r):
            for d in ti.static(range(self.dim)):
                self.r[I][d] = 0.
                self.p[I][d] = 0.
                self.q[I][d] = 0.
                self.Ap[I][d] = 0.

    @ti.kernel
    def reduction(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].dot(x[I])
        return result

    @ti.kernel
    def dot_product(self, a: ti.template(), b: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(a):
            result += a[I].dot(b[I])
        return result

    @ti.kernel
    def copy(self, dst: ti.template(), src: ti.template()):
        for I in ti.grouped(src):
            dst[I] = src[I]

    @ti.kernel
    def update(self, dst: ti.template(), src: ti.template(), scale: ti.f32):
        for I in ti.grouped(src):
            dst[I] += scale * src[I]

    @ti.kernel
    def update_general(self, dst: ti.template(), src_1: ti.template(), src_2: ti.template(), scale: ti.f32):
        for I in ti.grouped(src_1):
            dst[I] = src_1[I] + scale * src_2[I]

    @ti.kernel
    def compute_residual(self, b:ti.template()):
        for I in ti.grouped(b):
            self.r[I] = b[I] - self.Ap[I]

    @ti.kernel
    def compute_difference(self, dst:ti.template(), src1:ti.template(), src2:ti.template()):
        for I in ti.grouped(dst):
            dst[I] = src1[I] - src2[I]

    @ti.kernel
    def compute_norm(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].dot(x[I])
        return ti.sqrt(result)

    def solve(self, x, b):

        assert self.multiply is not None

        self.reinitialize()
        # CLear x
        self.multiply(x, self.Ap)
        self.compute_residual(b)
        self.compute_difference(self.r, b, self.Ap)

        rkTrk = abs(self.reduction(self.r))
        self.copy(self.p, self.r)
        # residual_norm = self.compute_norm(self.r)
        residual_norm = ti.sqrt(rkTrk)
        for i in range(self.max_iterations):
            if residual_norm < self.relative_tolerance * residual_norm:
                print(f"\033[1;31m CG terminated at {i}, Residual norm = {residual_norm} \033[0m")
                break
            if i % 10 == 0:
                print(f"\033[1;31m CG iteration: {i}, Residual norm = {residual_norm} \033[0m")
                pass
            self.multiply(self.p, self.Ap)
            alpha = rkTrk / self.dot_product(self.Ap, self.p)

            # Update x
            self.update(x, self.p, alpha)
            # Update r
            self.update(self.r, self.Ap, -alpha)
            residual_norm = self.compute_norm(self.r)

            rkTrk_last = rkTrk

            rkTrk = self.reduction(self.r)
            beta = rkTrk / rkTrk_last
            # print("rkTrk ", rkTrk, "alpha ", alpha, "beta ", beta)
            # Update p
            self.update_general(self.p, self.r, self.p, beta)
