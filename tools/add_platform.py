import argparse

import taichi as ti
import numpy as np
import atom.toolpath3
import atom.kinematics3z

ti.init(arch=ti.gpu)

def add_platform(input_path, output_path, nozzle_width, layer_height):
    toolpath = atom.toolpath3.Toolpath()
    toolpath.load(input_path)
    platform_size =  atom.kinematics3z.get_plaftorm_size(toolpath, nozzle_width, layer_height)
    toolpath_platform = atom.toolpath3.Toolpath()
    toolpath_platform.allocate(toolpath.point_count + 65536*int(platform_size[2]/layer_height))

    for i in range(int(platform_size[2]/layer_height)-1):
        z = (i+1)*layer_height

        # Perimenter
        toolpath_platform.insert(np.array((0, 0, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
        toolpath_platform.insert(np.array((0, platform_size[1], z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
        toolpath_platform.insert(np.array((platform_size[0], platform_size[1], z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
        toolpath_platform.insert(np.array((platform_size[0], 0, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
        toolpath_platform.insert(np.array((0, 0, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)

        # Top for easy detachment
        if i >= int(platform_size[2]/layer_height-3):
            infill_percentage = 0.2
            direction = 1 - (int(platform_size[2]/layer_height-6) % 2)
            if i % 2 == direction:
                for j in range(1, int(platform_size[1]/nozzle_width*infill_percentage)):
                    y = j*nozzle_width/infill_percentage
                    if j % 2 == 0:
                        toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
                    else:
                        toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
            else:
                for j in range(1, int(platform_size[0]/nozzle_width*infill_percentage)):
                    x = j*nozzle_width/infill_percentage
                    if j % 2 == 0:
                        toolpath_platform.insert(np.array((x, 0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((x, platform_size[1]-0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
                    else:
                        toolpath_platform.insert(np.array((x, platform_size[1]-0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((x, 0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)

        # Full layers
        elif i < 3 or i >= int(platform_size[2]/layer_height-6):
            direction = 1 - (int(platform_size[2]/layer_height-6) % 2)
            if i % 2 == direction:
                for j in range(1, int(platform_size[1]/nozzle_width)):
                    y = j*nozzle_width
                    if j % 2 == 0:
                        toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
                    else:
                        toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
            else:
                for j in range(1, int(platform_size[0]/nozzle_width)):
                    x = j*nozzle_width
                    if j % 2 == 0:
                        toolpath_platform.insert(np.array((x, 0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((x, platform_size[1]-0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
                    else:
                        toolpath_platform.insert(np.array((x, platform_size[1]-0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                        toolpath_platform.insert(np.array((x, 0.5*nozzle_width, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
        
        # Infill
        else:
            infill_percentage = 0.2
            for j in range(1, int(platform_size[1]/nozzle_width*infill_percentage)):
                y = j*nozzle_width/infill_percentage
                if j % 2 == 0:
                    toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                    toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
                else:
                    toolpath_platform.insert(np.array((platform_size[0]-0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_NO_DEPOSITION)
                    toolpath_platform.insert(np.array((0.5*nozzle_width, y, z)), np.array((0, 0)), atom.toolpath3.TRAVEL_TYPE_DEPOSITION)
    
    toolpath_output = atom.toolpath3.Toolpath()
    toolpath_output.allocate(toolpath.point_count+toolpath_platform.point_count)
    for i in range(toolpath_platform.point_count):
        toolpath_output.insert(toolpath_platform.point[i], toolpath_platform.tool_orientation[i], toolpath_platform.travel_type[i], nozzle_width, layer_height)
    for i in range(toolpath.point_count):
        toolpath_output.insert(toolpath.point[i]+np.array((0, 0, platform_size[2])), toolpath.tool_orientation[i], toolpath.travel_type[i], toolpath.width[i], toolpath.height[i])
        
    toolpath_output.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attempt to prevent collisions by lifting the object on a plaftorm.")
    parser.add_argument("input_path", help="The path to the input toolpath.")
    parser.add_argument("output_path", help="The path to the output toolpath with the added platform.")
    parser.add_argument("width", help="The deposition width for the platform.")
    parser.add_argument("height", help="The deposition height for the platform.")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    nozzle_width = float(args.width)
    layer_height = float(args.height)

    add_platform(input_path, output_path, nozzle_width, layer_height)
