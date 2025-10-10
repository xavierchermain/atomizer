import argparse
import json
import os

import taichi as ti

import atom.fff3

ti.init(arch=ti.cpu, offline_cache_cleaning_policy="never")


class Parameters:
    def __init__(self):
        self.deposition_width: float = None
        self.solid_name: str = None
        self.max_slope: float = None
        self.degree_angle_max_diff: float = None
        self.smoothing_iter_count: int = None
        self.ortho_to_wall: bool = None
        self.infill: bool = None

        self.log_path: str = None
        self.stl_path: str = None
        self.obj_path: str = None
        self.bpn_path: str = None
        self.direction_path: str = None
        self.basis_path: str = None
        self.phasor_path: str = None
        self.triphasor_path: str = None
        self.frame_path: str = None
        self.toolpath_path: str = None
        self.smoothed_toolpath_path: str = None
        self.smoothed_tesselated_toolpath_path: str = None
        self.gcode_path: str = None

    def load(self, path: str):
        with open(path) as file:
            data = file.read()
            param_dict = json.loads(data)

        self.deposition_width = param_dict["deposition_width"]
        self.solid_name = param_dict["solid_name"]
        self.max_slope = param_dict["max_slope"]

        self.ortho_to_wall = param_dict.get("ortho_to_wall")
        if self.ortho_to_wall is None:
            self.ortho_to_wall = False

        self.all_up = param_dict.get("all_up")
        if self.all_up is None:
            self.all_up = False

        self.infill = param_dict.get("infill")
        if self.infill is None:
            self.infill = False

        self.top_lines = param_dict.get("top_lines")
        self.bottom_lines = param_dict.get("bottom_lines")

        self.log_path = f"data/log/{self.solid_name}.log"
        self.stl_path = f"data/mesh/{self.solid_name}.stl"
        self.obj_path = f"data/mesh/{self.solid_name}.obj"
        self.bpn_path = f"data/point_normal/{self.solid_name}.npz"
        self.sdf_path = f"data/sdf/{self.solid_name}.npz"
        self.direction_path = f"data/direction/{self.solid_name}.npz"
        self.basis_path = f"data/basis/{self.solid_name}.npz"
        self.phasor_path = f"data/phasor/{self.solid_name}.npz"
        self.triphasor_path = f"data/triphasor/{self.solid_name}.npz"
        self.frame_path = f"data/frame/{self.solid_name}.npz"
        self.toolpath_path = f"data/toolpath/{self.solid_name}.npz"
        self.smoothed_toolpath_path = f"data/toolpath/{self.solid_name}_smoothed.npz"
        self.platform_toolpath_path = f"data/toolpath/{self.solid_name}_platform.npz"
        self.craftware_gcode_path = f"data/gcode/{self.solid_name}_craftware.gcode"
        self.smoothed_tesselated_toolpath_path = (
            f"data/toolpath/{self.solid_name}_smoothed_tesselated.npz"
        )
        self.gcode_path = f"data/gcode/{self.solid_name}.gcode"

        self.smoothing_iter_count = 8
        self.degree_angle_max_diff = 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atomize and order atoms for fused filament fabrication. The input is a 3D solid and the output is a toolpath composed of 3D positions, each associated with a basis. The 3D solid path and the command parameters (deposition width, maximum tilting angle for the tool) are given with a json file. This command is composed of several sub-commands, first processing the mesh to convert it to a SDF, then atomizing the 3D solid, and finally ordering atoms to create the toolpath. The output toolpath is located in the `data/toolpath` folder, and a log file is generated in the `data/log` folder, giving the list of sub-commands that are runned, and the list of commands to visualize all the generated data. "
    )
    parser.add_argument(
        "json_path",
        help="The path to the JSON file contening the command parameters. The parameters are: - `solid_name`: the filename of the STL representing the 3D solid, without the extension. The STL has to be in the `data/mesh/` folder, and its extension has to be `.stl`; - `deposition_width`: the deposition width in millimeter. The layer height is half the deposition width; - `max_slope`: the maximum tilting angle for the tool in degrees; - `top_lines` and `bottom_lines` (optionals): the paths to a 1 channel, 8 bits per pixel, PNG file representing the target tangent directions, for the top and bottom surfaces, respectively. The mapping is: [0, 255] -> [-pi/2, pi/2]. The orientation represents a 2D line in the xy-plane. The 2D line field is on the upper face of the solid bounding box and then planarly projected onto the top and bottom surfaces along the z-axis; - `ortho_to_wall` (optional): if true, force the tool orientation to be parallel to the boundary. By default is false, as this feature creates a lot of tool orientation changes that are detrimental to the surface quality.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="To avoid including compilation time in the computation time, run each sub-command twice to cache the compilation.",
    )

    args = parser.parse_args()
    json_path = args.json_path
    warmup = args.warmup

    # DEBUG
    # json_path = "data/param/calibration_slope_1.json"
    # warmup = True

    params = Parameters()
    params.load(json_path)

    log_file = open(params.log_path, "w", encoding="utf-8")

    cell_sides_length = atom.fff3.cell_sides_length_from_deposition_width_kernel(
        params.deposition_width
    )
    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        cell_sides_length
    )

    str_to_print = "# Parameters"
    print(str_to_print)
    log_file.write(str_to_print + "\n\n")

    str_to_print = f"- Solid name: {params.solid_name}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- STL input: {params.stl_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- GCode output: {params.gcode_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Log file: {params.log_path}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Deposition width: {params.deposition_width:.2f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Layer height: {layer_height:.2f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Cell sides length: {cell_sides_length:.3f}"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    str_to_print = f"- Machine max slope angle: {params.max_slope:.1f} degrees"
    print(str_to_print)
    log_file.write(str_to_print + "\n")

    process_mesh_command = f"blender -b -P tools/process_for_atomizer.py -- {params.stl_path} {params.obj_path} {cell_sides_length * 0.5:.6f}"
    obj_to_bpn_cmd = f"python tools/obj_to_bpn.py {params.obj_path} {params.bpn_path}"
    bpn_to_sdf_cmd = f"python tools/bpn_to_sdf.py {params.bpn_path} {params.sdf_path} {params.deposition_width:.2f}"
    sdf_to_isdf_cmd = f"python tools/sdf_to_isdf.py {params.bpn_path} {params.sdf_path} {params.sdf_path} no_gui=True"

    compute_tool_orientations_cmd = f"python tools/compute_tool_orientations.py {params.sdf_path} {params.direction_path} --maxslope {params.max_slope}"
    if params.ortho_to_wall:
        compute_tool_orientations_cmd += " --ortho_to_wall"
    if params.all_up:
        compute_tool_orientations_cmd += " --allup"
    compute_tool_orientations_cmd += f" --logpath {params.log_path}"

    sdf_df_to_layers_cmd = f"python tools/sdf_df_to_layers.py {params.sdf_path} {params.direction_path} {params.phasor_path} --maxslope {params.max_slope}"
    if params.all_up:
        sdf_df_to_layers_cmd += " --allup"
    sdf_df_to_layers_cmd += f" --logpath {params.log_path}"

    compute_tangents_cmd = f"python tools/compute_tangents.py {params.sdf_path} {params.direction_path} {params.basis_path} --maxslope {params.max_slope}"
    if params.top_lines is not None:
        compute_tangents_cmd += f" --top_lines {params.top_lines}"
    if params.bottom_lines is not None:
        compute_tangents_cmd += f" --bottom_lines {params.bottom_lines}"
    compute_tangents_cmd += f" --logpath {params.log_path}"

    align_atoms_cmd = f"python tools/align_atoms.py {params.sdf_path} {params.phasor_path} {params.basis_path} {params.triphasor_path} --maxslope {params.max_slope} --logpath {params.log_path}"
    extract_explicit_atoms_cmd = f"python tools/extract_explicit_atoms.py {params.sdf_path} {params.triphasor_path} {params.frame_path} --logpath {params.log_path}"
    order_atoms_cmd = f"python tools/order_atoms.py {params.sdf_path} {params.frame_path} {params.toolpath_path} --logpath {params.log_path}"
    smooth_toolpath_point_cmd = f"python tools/smooth_toolpath_point.py {params.toolpath_path} {params.smoothed_toolpath_path} {params.smoothing_iter_count}"
    tesselate_toolpath_orientations_cmd = f"python tools/tesselate_toolpath_orientations.py {params.smoothed_toolpath_path} {params.smoothed_tesselated_toolpath_path} {params.degree_angle_max_diff}"
    add_platform_cmd = f"python tools/add_platform.py {params.smoothed_tesselated_toolpath_path} {params.platform_toolpath_path} {params.deposition_width} {layer_height}"
    toolpath_to_gcode_cmd = f"python tools/toolpath_to_gcode.py {params.platform_toolpath_path} {params.gcode_path}"
    ratrig_to_craftware_cmd = f"python tools/ratrig_to_craftware.py {params.gcode_path} {params.craftware_gcode_path}"

    visualize_bpn_cmd = f"python tools/visualize_bpn.py {params.bpn_path}"
    visualize_bpn_sdf_cmd = (
        f"python tools/visualize_bpn_sdf.py {params.bpn_path} {params.sdf_path}"
    )
    visualize_bpn_df_cmd = (
        f"python tools/visualize_bpn_df.py {params.bpn_path} {params.direction_path}"
    )
    visualize_bpn_layers_cmd = (
        f"python tools/visualize_bpn_layers.py {params.bpn_path} {params.phasor_path}"
    )
    visualize_bases_cmd = (
        f"python tools/visualize_bases.py {params.bpn_path} {params.basis_path}"
    )
    visualize_implicit_atoms_cmd = f"python tools/visualize_implicit_atoms.py {params.bpn_path} {params.triphasor_path}"
    visualize_explicit_atoms_cmd = f"python tools/visualize_explicit_atoms.py {params.frame_path} {layer_height:.3f}"
    visualize_toolpath_cmd = (
        f"python tools/visualize_toolpath.py {params.smoothed_toolpath_path}"
    )

    log_file.write("\n# Pipeline commands\n\n")
    log_file.write(process_mesh_command + "\n")
    log_file.write(obj_to_bpn_cmd + "\n")
    log_file.write(bpn_to_sdf_cmd + "\n")
    log_file.write(compute_tool_orientations_cmd + "\n")
    log_file.write(sdf_df_to_layers_cmd + "\n")
    log_file.write(compute_tangents_cmd + "\n")
    log_file.write(align_atoms_cmd + "\n")
    log_file.write(extract_explicit_atoms_cmd + "\n")
    log_file.write(order_atoms_cmd + "\n")
    log_file.write(smooth_toolpath_point_cmd + "\n")

    log_file.write("\n# Visualize commands\n\n")
    log_file.write(visualize_bpn_cmd + "\n")
    log_file.write(visualize_bpn_sdf_cmd + "\n")
    log_file.write(visualize_bpn_df_cmd + "\n")
    log_file.write(visualize_bpn_layers_cmd + "\n")
    log_file.write(visualize_bases_cmd + "\n")
    log_file.write(visualize_implicit_atoms_cmd + "\n")
    log_file.write(visualize_explicit_atoms_cmd + "\n")
    log_file.write(visualize_toolpath_cmd + "\n")

    str_to_print = "\nStep 1 Starting: Remesh\n"
    print(str_to_print)
    log_file.close()
    os.system(process_mesh_command)

    str_to_print = "\nStep 2 Starting: Mesh to Point Normal Cloud\n"
    print(str_to_print)
    os.system(obj_to_bpn_cmd)

    str_to_print = "\nStep 3 Starting: Point Normal Cloud to SDF\n"
    print(str_to_print)
    os.system(bpn_to_sdf_cmd)

    if params.infill:
        str_to_print = "\nStep 3 Bis Starting: SDF to SDF with Infill\n"
        print(str_to_print)
        os.system(sdf_to_isdf_cmd)

    log_file = open(params.log_path, "a", encoding="utf-8")
    print("\nStep 4 Starting: Direction Field Computation\n")
    log_file.write("\n# Direction Field Computation\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(compute_tool_orientations_cmd.partition(" --logpath")[0])
    os.system(compute_tool_orientations_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 5 Starting: Implicit Layers Computation\n")
    log_file.write("\n# Implicit Layers Computation\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(sdf_df_to_layers_cmd.partition(" --logpath")[0])
    os.system(sdf_df_to_layers_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 6 Starting: Tangents Computation\n")
    log_file.write("\n# Tangents Computation\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(compute_tangents_cmd.partition(" --logpath")[0])
    os.system(compute_tangents_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 7 Starting: Atoms alignment\n")
    log_file.write("\n# Atoms alignment\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(align_atoms_cmd.partition(" --logpath")[0])
    os.system(align_atoms_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 8 Starting: Implicit to Explicit Atoms\n")
    log_file.write("\n# Implicit to Explicit Atoms\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(extract_explicit_atoms_cmd.partition(" --logpath")[0])
    os.system(extract_explicit_atoms_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 9 Starting: Order Atoms\n")
    log_file.write("\n# Order Atoms\n\n")
    log_file.close()
    if warmup:
        print("Warmup")
        os.system(order_atoms_cmd.partition(" --logpath")[0])
    os.system(order_atoms_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 10 Starting: Smooth, Tesselate and Add a Platform\n")
    log_file.write("\n# Smooth, Tesselate and Add a Platform\n\n")
    log_file.close()
    os.system(smooth_toolpath_point_cmd)
    os.system(tesselate_toolpath_orientations_cmd)
    os.system(add_platform_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")

    print("\nStep 11 Starting: G-Code Generation\n")
    log_file.write("\n# G-Code Generation\n\n")
    log_file.close()
    os.system(toolpath_to_gcode_cmd)
    os.system(ratrig_to_craftware_cmd)
    log_file = open(params.log_path, "a", encoding="utf-8")
