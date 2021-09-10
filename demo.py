import taichi as ti
from mls_mpm import MlsMpmSolver
from gradient_descent import GradientDescentSolver
from newton_optimization_solver import NewtonSolver


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
    parser.add_argument('--line_search', action='store_true',
                        help='do line search in newton or not')
    parser.add_argument('--gradient_descent', action='store_true',
                        help='do gradient descent or not')
    args = parser.parse_args()

    dim = args.dim
    assert dim == 2 or dim == 3

    ti.init(arch=ti.cuda, kernel_profiler=True)
    
    newton_tolearance = 1e-6 if dim == 2 else 1e-12
    res = (512, 512) if dim == 2 else (1024, 1024, 1024)
    dtype = ti.f32
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
        optimization_solver_type, GradientDescentSolver(max_iterations=100,
                                                        adaptive_step_size=False))
    elif optimization_solver_type == 'newton':
        optimization_solver = (optimization_solver_type, NewtonSolver(line_search=args.line_search,
                                                                      max_iterations=10, 
                                                                      tolerance=newton_tolearance))

    solver = MlsMpmSolver(res,
                          dt=dt,
                          gravity=9.8,
                          dtype=dtype,
                          implicit=args.implicit,
                          optimization_solver=optimization_solver,
                          do_diff_test=args.difftest)
    solver.initialize()

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
        window = ti.ui.Window('Taichi MLS-MPM', res)
        canvas = window.get_canvas()

        while window.running:
            # for s in range(int(visualization_limit // solver.dt)):
            for s in range(int(5)):
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
