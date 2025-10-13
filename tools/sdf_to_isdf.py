import argparse

import numpy as np
import taichi as ti

import atom.direction
import atom.drawer3
import atom.fff3
import atom.phasor3
import atom.solid3


@ti.kernel
def copy(input: ti.template(), output: ti.template()):
    for i in ti.grouped(input):
        output[i] = input[i]


def sdf_to_isdf(bpn_path, sdf_path, isdf_path, no_gui=False):
    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)
    print(f"Domain size: {bpn.get_size()}")
    print(f"bounding box [xmin, xmax, ymin, ymax, zmin, zmax]: {bpn.bounding_box}")
    print(f"Point count: {bpn.point.shape[0]}")

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)

    input_sdf = ti.field(dtype=ti.f32, shape=sdf.grid.cell_3dcount)
    input_sdf.copy_from(sdf.sdf)

    offset = ti.math.vec3(0)
    offset_support = ti.math.vec3(0)
    infill_period = 8
    support_period = 4
    shell_thickness = 2
    compute_support = False
    compute_infill = True
    gyroid = True

    flag_field = ti.field(dtype=ti.uint8, shape=sdf.grid.cell_3dcount)

    def recompute():
        if compute_support:
            atom.fff3.generate_supports(
                sdf.sdf, input_sdf, flag_field, offset_support, support_period
            )
            if compute_infill:
                atom.fff3.sdf_generate_infill(
                    input_sdf,
                    sdf.sdf,
                    sdf.grid.cell_sides_length,
                    offset,
                    infill_period,
                    shell_thickness,
                    gyroid,
                )
        elif compute_infill:
            atom.fff3.sdf_generate_infill(
                input_sdf,
                sdf.sdf,
                sdf.grid.cell_sides_length,
                offset,
                infill_period,
                shell_thickness,
                gyroid,
            )
        else:
            copy(input_sdf, sdf.sdf)

    recompute()

    if no_gui:
        sdf.save(isdf_path)
        return

    normal_scale = 0.1
    see_boundary = False
    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    subdivisions = np.array([256, 256])
    grid_mesh2 = atom.drawer3.GridMesh2()
    grid_mesh2.create(subdivisions)
    grid_mesh2_orientation_theta = grid_mesh2.orientation[0]
    grid_mesh2_orientation_phi = grid_mesh2.orientation[1]
    origin_x = bpn.bounding_box[1] * 0.5
    origin_y = bpn.bounding_box[3] * 0.5
    origin_z = bpn.bounding_box[5] * 0.5
    infill_x = 0
    infill_y = 0
    infill_z = 0
    support_x = 0
    support_y = 0
    support_z = 0

    grid_mesh2.size = max(bpn.bounding_box)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )

    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()

    camera = ti.ui.Camera()
    camera.position(
        bpn.bounding_box[1] * 1.5, bpn.bounding_box[5] * 1.5, -bpn.bounding_box[3] * 1.5
    )
    camera.lookat(origin_x, origin_z, -origin_y)
    camera.up(0, 1, 0)

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "b":
                if see_boundary:
                    see_boundary = False
                else:
                    see_boundary = True
        with gui.sub_window("Parameters", 0.05, 0.05, 0.28, 0.63) as w:
            grid_mesh2.size = w.slider_float(
                "Grid mesh size",
                grid_mesh2.size,
                min(bpn.bounding_box),
                max(bpn.bounding_box),
            )
            grid_mesh2_orientation_theta = w.slider_float(
                "Theta", grid_mesh2_orientation_theta, 0.0, ti.math.pi
            )
            grid_mesh2_orientation_phi = w.slider_float(
                "Phi", grid_mesh2_orientation_phi, 0.0, ti.math.pi * 2.0
            )
            origin_x = w.slider_float("Origin X", origin_x, 0.0, bpn.bounding_box[1])
            origin_y = w.slider_float("Origin Y", origin_y, 0.0, bpn.bounding_box[3])
            origin_z = w.slider_float("Origin Z", origin_z, 0.0, bpn.bounding_box[5])

            compute_infillN = w.checkbox("Compute infill", compute_infill)
            if compute_infill != compute_infillN:
                compute_infill = compute_infillN
                recompute()
            gyroidN = w.checkbox("Gyroid infill", gyroid)
            if gyroid != gyroidN:
                gyroid = gyroidN
                recompute()
            infill_xN = w.slider_float("Infill X", infill_x, 0.0, bpn.bounding_box[1])
            if infill_xN != infill_x:
                infill_x = infill_xN
                offset.x = infill_xN
                recompute()
            infill_yN = w.slider_float("Infill Y", infill_y, 0.0, bpn.bounding_box[3])
            if infill_yN != infill_y:
                infill_y = infill_yN
                offset.y = infill_yN
                recompute()
            infill_zN = w.slider_float("Infill Z", infill_z, 0.0, bpn.bounding_box[5])
            if infill_zN != infill_z:
                infill_z = infill_zN
                offset.z = infill_zN
                recompute()
            infill_periodN = w.slider_int("Infill period", infill_period, 0, 32)
            if infill_periodN != infill_period:
                infill_period = infill_periodN
                recompute()
            infill_sN = w.slider_int("Shell thickness", shell_thickness, 0, 8)
            if shell_thickness != infill_sN:
                shell_thickness = infill_sN
                recompute()

            compute_supportN = w.checkbox("Compute supports", compute_support)
            if compute_support != compute_supportN:
                compute_support = compute_supportN
                recompute()
            support_xN = w.slider_float(
                "Support X", support_x, 0.0, bpn.bounding_box[1]
            )
            if support_xN != support_x:
                support_x = support_xN
                offset_support.x = support_xN
                recompute()
            support_yN = w.slider_float(
                "Support Y", support_y, 0.0, bpn.bounding_box[3]
            )
            if support_yN != support_y:
                support_y = support_yN
                offset_support.y = support_yN
                recompute()
            support_zN = w.slider_float(
                "Support Z", support_z, 0.0, bpn.bounding_box[5]
            )
            if support_zN != support_z:
                support_z = support_zN
                offset_support.z = support_zN
                recompute()
            support_periodN = w.slider_int("Support period", support_period, 0, 32)
            if support_periodN != support_period:
                support_period = support_periodN
                recompute()

            if w.button("Save"):
                sdf.save(isdf_path)

        grid_mesh2.orientation = np.array(
            [grid_mesh2_orientation_theta, grid_mesh2_orientation_phi]
        )
        grid_mesh2.origin = np.array([origin_x, origin_y, origin_z])
        grid_mesh2.update_vertex_normal()
        grid_mesh2.update_per_vertex_color_with_sdf(sdf)

        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((1.0, 1.0, 1.0))

        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )
        scene.mesh(
            grid_mesh2.vertex,
            grid_mesh2.index,
            grid_mesh2.normal,
            show_wireframe=False,
            per_vertex_color=grid_mesh2.per_vertex_color,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("bpn_path")
    parser.add_argument("sdf_path")
    parser.add_argument("isdf_path")
    parser.add_argument("no_gui", nargs="?", default="False")

    args = parser.parse_args()

    bpn_path = args.bpn_path
    sdf_path = args.sdf_path
    isdf_path = args.isdf_path
    no_gui = args.no_gui != "False" and args.no_gui != "0"

    sdf_to_isdf(bpn_path, sdf_path, isdf_path, no_gui)
