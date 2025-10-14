import argparse
import time
from tqdm import tqdm
import taichi as ti
import numpy as np

import atom.toolpath3
import atom.kinematics3z

ti.init(arch=ti.gpu)

@ti.kernel
def calculate_extrusion(point: ti.types.ndarray(), width: ti.types.ndarray(), height: ti.types.ndarray(), travel_type: ti.types.ndarray(), filament_diameter: ti.f32, extrusion: ti.types.ndarray()):
    for i in range(1, point.shape[0]):
        if travel_type[i] == atom.toolpath3.TRAVEL_TYPE_DEPOSITION:
            cross_section = ti.math.pi*filament_diameter*filament_diameter/4.0
            p1 = ti.math.vec3(point[i, 0],point[i, 1],point[i, 2])
            p0 = ti.math.vec3(point[i-1, 0],point[i-1, 1],point[i-1, 2])
            volume = ti.math.length(p1-p0) * height[i] * width[i]
            extrusion[i] = volume/cross_section
        else:
            extrusion[i] = 0

@ti.kernel
def override_width_and_height(width: ti.types.ndarray(), height: ti.types.ndarray(), width_override: ti.f32, height_override: ti.f32):
    for i in range(width.shape[0]):
        width[i] = width_override
        height[i] = height_override

@ti.kernel
def calculate_feedrate(point: ti.types.ndarray(), point_machine: ti.types.ndarray(), travel_type: ti.types.ndarray(), extrusion: ti.types.ndarray(), feed_rate: ti.types.ndarray()):
    for i in range(point.shape[0]):
        if i == 0 or travel_type[i] == atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION:
            feed_rate[i] = atom.kinematics3z.TRAVEL_FEEDRATE
        else:
            x2 = (point_machine[i,0]-point_machine[i-1,0])**2
            y2 = (point_machine[i,1]-point_machine[i-1,1])**2
            z2 = (point_machine[i,2]-point_machine[i-1,2])**2
            u2 = (point_machine[i,3]-point_machine[i-1,3])**2
            v2 = (point_machine[i,4]-point_machine[i-1,4])**2
            e2 = extrusion[i]**2
            gcode_dist = ti.math.sqrt(x2+y2+z2+u2+v2+e2)
            real_dist = ti.math.length(ti.math.vec3(point[i, 0],point[i, 1],point[i, 2])-ti.math.vec3(point[i-1, 0],point[i-1, 1],point[i-1, 2]))
            feed_rate[i] = atom.kinematics3z.DEPOSITON_FEEDRATE * (1 if real_dist == 0 else gcode_dist/real_dist)

@ti.kernel
def center_toolpath(point: ti.types.ndarray(), x_offset: ti.f32, y_offset: ti.f32):
    for i in range(point.shape[0]):
        point[i, 0] += x_offset
        point[i, 1] += y_offset

@ti.kernel
def calculate_retract(point: ti.types.ndarray(), travel_type: ti.types.ndarray(), retract: ti.types.ndarray()):
    for i in range(1, point.shape[0]):
        if travel_type[i] == atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION:
            j = i
            dist = 0.0
            while j < point.shape[0] and travel_type[j] == atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION:
                dist += ti.math.length(ti.math.vec3(point[j, 0],point[j, 1],point[j, 2])-ti.math.vec3(point[j-1, 0],point[j-1, 1],point[j-1, 2]))
                if dist > atom.kinematics3z.RETRACT_THRESH:
                    retract[i] = 1
                    break
                j += 1
       
def toolpath_to_gcode(toolpath, gcode_path, width, height):
    if width > 0:
        if height == -1:
            height = 0.5*width
        override_width_and_height(toolpath.width, toolpath.height, width, height)

    toolpath_min, toolpath_max = toolpath.get_aabb()
    toolpath_offset = 0.5*(np.array((atom.kinematics3z.MAX_X_AXIS, atom.kinematics3z.MAX_Y_AXIS, 0)) + toolpath_min-toolpath_max) - toolpath_min
    center_toolpath(toolpath.point, toolpath_offset[0], toolpath_offset[1])

    machine_toolpath = atom.kinematics3z.Toolpath()
    valid = machine_toolpath.from_cartesian_toolpath(toolpath)
    if not valid:
        print("Fatal Error: collision found!")
        return

    extrusion = np.full(dtype=np.float32, shape=toolpath.point_count, fill_value=0)
    calculate_extrusion(toolpath.point, toolpath.width, toolpath.height, toolpath.travel_type, atom.kinematics3z.FILAMENT_DIAMETER, extrusion)

    feed_rate = np.full(dtype=np.float32, shape=toolpath.point_count, fill_value=600)
    calculate_feedrate(toolpath.point, machine_toolpath.point, toolpath.travel_type, extrusion, feed_rate)

    retract = np.full(dtype=np.int32, shape=toolpath.point_count, fill_value=0)
    calculate_retract(toolpath.point, toolpath.travel_type, retract)

    with open(gcode_path, "w") as file:
        file.write(atom.kinematics3z.HEADER)
        is_fan_on = False
        need_prime = True
        for i in tqdm(range(toolpath.point_count)):
            if need_prime and toolpath.travel_type[i] == atom.toolpath3.TRAVEL_TYPE_DEPOSITION:
                file.write(f"G1 E{atom.kinematics3z.RETRACT_LENGTH:.2f} F{atom.kinematics3z.RETRACT_SPEED:.0f} ; prime\n")
                need_prime = False
            elif not need_prime and retract[i]:
                file.write(f"G1 E-{atom.kinematics3z.RETRACT_LENGTH:.2f} F{atom.kinematics3z.RETRACT_SPEED:.0f} ; retract\n")
                need_prime = True

            x = machine_toolpath.point[i][0]
            y = machine_toolpath.point[i][1]
            z = machine_toolpath.point[i][2]
            u = machine_toolpath.point[i][3]
            v = machine_toolpath.point[i][4]
            e = extrusion[i]
            f = feed_rate[i]
            file.write(f"G1 X{x:.6f} Y{y:.6f} Z{z:.6f} U{u:.6f} V{v:.6f} E{e:.6f} F{f:.0f}\n")

            if is_fan_on and toolpath.point[i][2] < atom.kinematics3z.Z_FAN_ON:
                file.write("M106 S0 ; fan off\n")
                is_fan_on = False
            elif not is_fan_on and toolpath.point[i][2] > atom.kinematics3z.Z_FAN_ON:
                file.write("M106 S127 ; fan on\n")
                is_fan_on = True

        file.write(atom.kinematics3z.FOOTER)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the given toolpath to G-code for a 3Z-axis RatRig machine.")
    parser.add_argument("toolpath_path", help="The path to the input toolpath.")
    parser.add_argument("gcode_path", help="The path to the output G-code.")
    parser.add_argument("width_override", nargs="?", default="-1", help="The override for the deposition width.")
    parser.add_argument("height_override", nargs="?", default="-1", help="The override for the deposition height.")
    args = parser.parse_args()

    toolpath_path = args.toolpath_path
    toolpath = atom.toolpath3.Toolpath()
    toolpath.load(toolpath_path)
    toolpath.point = toolpath.point[: toolpath.point_count]
    gcode_path = args.gcode_path
    width = float(args.width_override)
    height = float(args.height_override)

    start = time.perf_counter()
    toolpath_to_gcode(toolpath, gcode_path, width, height)
    print(f"Toolpath to GCode took {time.perf_counter()-start:.1f} seconds.")
