from math import ceil, cos, pi

import numpy as np
import taichi as ti
import taichi.math as tm

from . import basis3, cone, direction, frame3, limits, math
from .bvh import BVH, STACK_SIZE_MAX
from .set import Set, set_find_u32, set_insert_u32, set_remove_u32

# Region thresholds Distance are relative to layer height for axis 0 (tangent)
# and 2 (normal), and relative to deposition width for axis 1 (bitangent)
THRESHOLD_T_0 = 2.0
THRESHOLD_T_1 = 0.5
THRESHOLD_T_2 = 0.5
THRESHOLD_B_0 = THRESHOLD_T_0 * 0.5
THRESHOLD_B_1 = 1.5
THRESHOLD_B_2 = THRESHOLD_T_2
THRESHOLD_N_0 = 0.5
THRESHOLD_N_1 = 0.5
THRESHOLD_N_2 = 1.5

NOZZLE_CONE_ANGLE = 80.0 * pi / 180.0
NOZZLE_COS_HALF_ANGLE = cos(NOZZLE_CONE_ANGLE * 0.5)

# Angle for cone supporting region: 130 degrees
SUPPORTING_REGION_CONE_ANGLE = 130.0 * pi / 180.0
SUPPORTING_REGION_COS_HALF_ANGLE = cos(SUPPORTING_REGION_CONE_ANGLE * 0.5)

# Travel types
TRAVEL_TYPE_DEPOSITION = 0
TRAVEL_TYPE_NO_DEPOSITION = 1

# For allocating arrays
TRAVEL_POINT_SIZE_MAX = 2**13
CYCLE_SIZE_MAX = 2**15


class Toolpath:
    def __init__(self) -> None:
        self.point = None
        self.travel_type = None
        self.tool_orientation = None
        self.width = None
        self.height = None

        self.point_count = None
        self.length_from_start = None
        self.platform_height = None

    def allocate(self, size: int):
        self.point = np.full(dtype=np.float32, shape=(size, 3), fill_value=ti.math.nan)
        self.travel_type = np.full(shape=size, dtype=np.int32, fill_value=-1)
        self.tool_orientation = np.full(
            shape=(size, 2), dtype=np.float32, fill_value=0.0
        )
        self.width = np.full(shape=size, dtype=np.float32, fill_value=0.0)
        self.height = np.full(shape=size, dtype=np.float32, fill_value=0.0)
        self.point_count = 0
        self.platform_height = 0

    def to_numpy(self):
        dict_array = {}
        dict_array["point"] = self.point
        dict_array["travel_type"] = self.travel_type
        dict_array["tool_orientation"] = self.tool_orientation
        dict_array["width"] = self.width
        dict_array["height"] = self.height
        dict_array["point_count"] = np.array(self.point_count)
        dict_array["platform_height"] = self.platform_height

        return dict_array

    def from_numpy(self, dict_array):
        self.point = dict_array["point"]
        self.travel_type = dict_array["travel_type"]
        self.tool_orientation = dict_array["tool_orientation"]
        self.width = dict_array["width"]
        self.height = dict_array["height"]
        self.point_count = int(dict_array["point_count"][()])
        self.platform_height = (
            dict_array["platform_height"]
            if "platform_height" in dict_array.keys()
            else 0
        )

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def insert(
        self,
        point: np.ndarray,
        tool_orientation: np.ndarray,
        travel_type: int,
        width: float = 0.0,
        height: float = 0.0,
    ):
        self.point[self.point_count] = point
        self.tool_orientation[self.point_count] = tool_orientation
        self.travel_type[self.point_count] = travel_type
        self.width[self.point_count] = width
        self.height[self.point_count] = height
        self.point_count += 1

    def flip(self):
        self.point = np.ascontiguousarray(
            np.flip(self.point[: self.point_count], axis=0)
        )
        self.tool_orientation = np.ascontiguousarray(
            np.flip(self.tool_orientation[: self.point_count], axis=0)
        )
        travel_type_flipped = np.flip(self.travel_type[: self.point_count])

        self.travel_type = np.ascontiguousarray(np.roll(travel_type_flipped, 1))
        self.width = np.ascontiguousarray(np.flip(self.width[: self.point_count]))
        self.height = np.ascontiguousarray(np.flip(self.height[: self.point_count]))

    def get_aabb(self):
        p_min = np.min(self.point, axis=0)
        p_max = np.max(self.point, axis=0)

        return p_min, p_max

    def length(self, count_travel=False):
        length = 0.0
        for i in range(1, self.point_count):
            if not count_travel and self.travel_type[i] == TRAVEL_TYPE_NO_DEPOSITION:
                continue
            length += np.linalg.norm(self.point[i] - self.point[i - 1])
        return length

    def compute_length_from_start(self):
        self.length_from_start = np.full(
            shape=self.point_count, dtype=np.float32, fill_value=0.0
        )
        length_from_start_i = 0.0
        self.length_from_start[0] = length_from_start_i
        for i in range(1, self.point_count):
            length_from_start_i += np.linalg.norm(self.point[i] - self.point[i - 1])
            self.length_from_start[i] = length_from_start_i

    def uplift(self, height):
        toolpath_uplift(self.point, self.tool_orientation, height)

    def smooth_orientations(self, iter_count: int):
        tool_orientation_buffer = np.full(
            shape=(self.point_count, 2), dtype=np.float32, fill_value=0.0
        )
        for _ in range(iter_count):
            toolpath_smooth_orientations(self.tool_orientation, tool_orientation_buffer)
            self.tool_orientation, tool_orientation_buffer = (
                tool_orientation_buffer,
                self.tool_orientation,
            )

    def max_diff_orientation(self):
        return toolpath_max_diff_orientation(self.tool_orientation)

    def smooth_points(self, iter_count: int):
        point_buffer = np.full(
            shape=(self.point_count, 3), dtype=np.float32, fill_value=0.0
        )
        for _ in range(iter_count):
            toolpath_smooth_points(self.point, self.travel_type, point_buffer)
            self.point, point_buffer = (
                point_buffer,
                self.point,
            )

    def tesselate_orientation(self, max_angle_degree: float):
        max_angle_radians = max_angle_degree / 180.0 * ti.math.pi
        MAX_POINT = self.point_count * 3
        point_new = np.full(shape=(MAX_POINT, 3), dtype=np.float32, fill_value=0.0)
        normal_new = np.full(shape=(MAX_POINT, 2), dtype=np.float32, fill_value=0.0)
        travel_type_new = np.full(shape=MAX_POINT, dtype=np.int32, fill_value=-1)
        width_new = np.full(shape=MAX_POINT, dtype=np.float32, fill_value=0.0)
        height_new = np.full(shape=MAX_POINT, dtype=np.float32, fill_value=0.0)
        self.point_count = toolpath_tesselate_orientation(
            self.point,
            self.tool_orientation,
            self.travel_type,
            self.width,
            self.height,
            point_new,
            normal_new,
            travel_type_new,
            width_new,
            height_new,
            max_angle_radians,
        )
        self.point = point_new[: self.point_count, :]
        self.tool_orientation = normal_new[: self.point_count, :]
        self.travel_type = travel_type_new[: self.point_count]
        self.width = width_new[:self.point_count]
        self.height = height_new[:self.point_count]

    
    
    def set_constant_deposition_width_and_height(self, width, height):
        self.width = np.full(shape=(self.point.shape[0],), fill_value=width, dtype=np.float32)
        self.height = np.full(shape=(self.point.shape[0],), fill_value=height, dtype=np.float32)


class ToolpathTaichi:
    def __init__(self):
        self.point = None
        self.travel_type = None
        self.tool_orientation = None
        self.width = None
        self.height = None

        self.point_count = None

    def allocate(self, size: int):
        self.point = ti.Vector.field(n=3, dtype=ti.f32, shape=size)
        self.travel_type = ti.field(dtype=ti.i32, shape=size)
        self.tool_orientation = ti.Vector.field(n=2, dtype=ti.f32, shape=size)
        self.width = ti.field(dtype=ti.f32, shape=size)
        self.height = ti.field(dtype=ti.f32, shape=size)
        self.point_count = 0

    def from_numpy(self, dict_array):
        shape = dict_array["point"].shape[0]

        self.point = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
        self.travel_type = ti.field(dtype=ti.i32, shape=shape)
        self.tool_orientation = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.width = ti.field(dtype=ti.f32, shape=shape)
        self.height = ti.field(dtype=ti.f32, shape=shape)

        self.point.from_numpy(dict_array["point"])
        self.travel_type.from_numpy(dict_array["travel_type"])
        self.tool_orientation.from_numpy(dict_array["tool_orientation"])
        self.width.from_numpy(dict_array["width"])
        self.height.from_numpy(dict_array["height"])
        self.point_count = int(dict_array["point_count"][()])

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def to_numpy(self):
        point_np = self.point.to_numpy()
        travel_type_np = self.travel_type.to_numpy()
        tool_orientation_np = self.tool_orientation.to_numpy()
        width_np = self.width.to_numpy()
        height_np = self.height.to_numpy()

        dict_array = {}
        dict_array["point"] = point_np
        dict_array["travel_type"] = travel_type_np
        dict_array["tool_orientation"] = tool_orientation_np
        dict_array["width"] = width_np
        dict_array["height"] = height_np
        dict_array["point_count"] = np.array(self.point_count)

        return dict_array


class ToolpathPlanner:
    def __init__(self) -> None:
        self.non_supporting_frames_set_size_max = 2**17

        self.frame_set: frame3.Set = None
        self.frame_data_set: FrameDataSet = None
        self.state_buffer = None
        self.toolpath: ToolpathTaichi = None
        self.travel_point = None
        self.travel_normal = None
        self.non_supporting_frame_set: Set = None
        self.triperiod: np.ndarray = None
        self.current_point: np.ndarray = None
        self.previous_point: np.ndarray = None
        self.current_normal_sph: np.ndarray = None
        self.p_0: np.ndarray = None
        self.domain_diag: float = None
        self.neighborhood_radius: float = None
        self.current_frame_index: int = None
        self.previous_frame_index: int = None
        self.current_travel_type: int = None
        self.iteration_number: int = None
        self.fast_track: int = None
        self.backpropagate: int = None
        self.all_reachable_by_constraint: int = None
        self.toolpath_point_count_previous: int = None
        self.bvh: BVH = None

    def init(
        self,
        frame_set: frame3.Set,
        triperiod: np.ndarray,
        neighborhood_radius: float,
        domain_diag: float,
        bvh: BVH,
        p_highest: np.ndarray,
    ):
        self.frame_set = frame_set

        self.frame_data_set = FrameDataSet()
        self.frame_data_set.init(self.frame_set)

        # Used for double buffering to update the state of the frames
        self.state_buffer = ti.field(dtype=ti.u32, shape=self.frame_set.phi_t.shape)

        self.toolpath = ToolpathTaichi()
        self.toolpath.allocate(int(self.frame_set.phi_t.shape[0] * 8))
        self.non_supporting_frame_set = Set()
        self.non_supporting_frame_set.create_u32(
            self.non_supporting_frames_set_size_max
        )

        self.bvh = bvh

        self.travel_point = ti.Vector.field(
            n=3, dtype=ti.f32, shape=TRAVEL_POINT_SIZE_MAX
        )
        self.travel_normal = ti.Vector.field(
            n=3, dtype=ti.f32, shape=TRAVEL_POINT_SIZE_MAX
        )

        self.cycle_is_visited = ti.field(dtype=ti.u32, shape=CYCLE_SIZE_MAX)
        self.cycle_index_stack = ti.field(dtype=ti.i32, shape=CYCLE_SIZE_MAX)
        self.cycle_p_jm1_stack = ti.Vector.field(
            n=3, dtype=ti.f32, shape=CYCLE_SIZE_MAX
        )

        self.triperiod = triperiod
        self.neighborhood_radius = neighborhood_radius
        self.domain_diag = domain_diag

        self.current_point = np.array([-1.0, -1.0, -1.0])
        self.previous_point = np.array([-1.0, -1.0, -1.0])

        self.current_frame_index = -1
        self.previous_frame_index = -1
        self.current_travel_type = TRAVEL_TYPE_NO_DEPOSITION

        self.p_0 = p_highest

        self.iteration_number = 0

        self.fast_track = False
        self.backpropagate = False
        self.all_reachable_by_constraint = False
        self.toolpath_point_count_previous = 0

    def to_numpy(self):
        frame_set_dict = self.frame_set.to_numpy()
        frame_data_set_dict = self.frame_data_set.to_numpy()
        toolpath_dict = self.toolpath.to_numpy()
        toolpath_dict["toolpath_point"] = toolpath_dict.pop("point")
        non_supporting_frame_set_dict = self.non_supporting_frame_set.to_numpy()

        dict_array = {}
        dict_array["triperiod"] = self.triperiod
        dict_array["current_point"] = self.current_point
        dict_array["previous_point"] = self.previous_point
        dict_array["current_normal"] = self.current_normal_sph
        dict_array["p_0"] = self.p_0
        dict_array["domain_diag"] = np.array(self.domain_diag)
        dict_array["neighborhood_radius"] = np.array(self.neighborhood_radius)
        dict_array["current_frame_index"] = np.array(self.current_frame_index)
        dict_array["previous_frame_index"] = np.array(self.previous_frame_index)
        dict_array["current_travel_type"] = np.array(self.current_travel_type)
        dict_array["iteration_number"] = np.array(self.iteration_number)
        dict_array["fast_track"] = np.array(self.fast_track)
        dict_array["backpropagate"] = np.array(self.backpropagate)
        dict_array["all_reachable_by_constraint"] = np.array(
            self.all_reachable_by_constraint
        )
        dict_array["toolpath_point_count_previous"] = np.array(
            self.toolpath_point_count_previous
        )

        dict_array = {
            **dict_array,
            **frame_set_dict,
            **frame_data_set_dict,
            **toolpath_dict,
            **non_supporting_frame_set_dict,
        }

        # BVH is not saved

        return dict_array

    def from_numpy(self, dict_array):
        self.frame_set = frame3.Set()
        frame_set_dict = {k: dict_array[k] for k in {"point", "normal", "phi_t"}}
        self.frame_set.from_numpy(frame_set_dict)

        self.frame_data_set = FrameDataSet()
        frame_data_set_dict = {k: dict_array[k] for k in {"state", "distance", "cost"}}
        self.frame_data_set.from_numpy(frame_data_set_dict)

        self.toolpath = ToolpathTaichi()
        toolpath_dict = {
            k: dict_array[k]
            for k in {
                "toolpath_point",
                "travel_type",
                "tool_orientation",
                "width",
                "height",
                "point_count",
            }
        }
        toolpath_dict["point"] = toolpath_dict.pop("toolpath_point")
        self.toolpath.from_numpy(toolpath_dict)

        self.non_supporting_frame_set = Set()
        non_supporting_frame_set_dict = {
            k: dict_array[k] for k in {"value", "size", "size_max"}
        }
        self.non_supporting_frame_set.from_numpy(non_supporting_frame_set_dict)

        self.triperiod = dict_array["triperiod"]
        self.current_point = dict_array["current_point"]
        self.previous_point = dict_array["previous_point"]
        self.current_normal_sph = dict_array["current_normal"]
        self.p_0 = dict_array["p_0"]
        self.domain_diag = float(dict_array["domain_diag"])
        self.neighborhood_radius = float(dict_array["neighborhood_radius"])
        self.current_frame_index = int(dict_array["current_frame_index"])
        self.previous_frame_index = int(dict_array["previous_frame_index"])
        self.current_travel_type = int(dict_array["current_travel_type"])
        self.iteration_number = int(dict_array["iteration_number"])
        self.fast_track = int(dict_array["fast_track"])
        self.backpropagate = int(dict_array["backpropagate"])
        self.all_reachable_by_constraint = int(
            dict_array["all_reachable_by_constraint"]
        )
        self.toolpath_point_count_previous = int(
            dict_array["toolpath_point_count_previous"]
        )

        self.state_buffer = ti.field(dtype=ti.u32, shape=self.frame_set.phi_t.shape)

        self.travel_point = ti.Vector.field(
            n=3, dtype=ti.f32, shape=TRAVEL_POINT_SIZE_MAX
        )
        self.travel_normal = ti.Vector.field(
            n=3, dtype=ti.f32, shape=TRAVEL_POINT_SIZE_MAX
        )

        self.cycle_is_visited = ti.field(dtype=ti.u32, shape=CYCLE_SIZE_MAX)
        self.cycle_index_stack = ti.field(dtype=ti.i32, shape=CYCLE_SIZE_MAX)
        self.cycle_p_jm1_stack = ti.Vector.field(
            n=3, dtype=ti.f32, shape=CYCLE_SIZE_MAX
        )

        self.bvh = BVH(auto_rebuild=True)
        self.bvh.from_frame_set(self.frame_set)

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def atom_i_is_adjacent_to_constraint(self, i):
        return toolpath_planner_atom_i_is_adjacent_to_constraint(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.frame_data_set.state,
            self.bvh,
            i,
            self.triperiod,
            self.neighborhood_radius,
        )

    def init_non_supporting_set_brute_force(self):
        self.non_supporting_frame_set.size = int(
            toolpath_planner_init_non_supporting_set_brute_force(
                self.frame_set.point,
                self.frame_set.normal,
                self.frame_set.phi_t,
                self.triperiod,
                self.neighborhood_radius,
                self.non_supporting_frame_set.value,
            )
        )

    def update_non_supporting_set_brute_force(self):
        self.non_supporting_frame_set.size = int(
            toolpath_planner_update_non_supporting_set_brute_force(
                self.frame_set.point,
                self.frame_set.normal,
                self.frame_set.phi_t,
                self.frame_data_set.state,
                self.non_supporting_frame_set.value,
                self.current_point,
                self.triperiod,
                self.neighborhood_radius,
                self.current_frame_index,
                self.non_supporting_frame_set.size,
            )
        )

    def update_non_supporting_set(self):
        self.non_supporting_frame_set.size = int(
            toolpath_planner_update_non_supporting_set(
                self.frame_set.point,
                self.frame_set.normal,
                self.frame_set.phi_t,
                self.frame_data_set.state,
                self.non_supporting_frame_set.value,
                self.bvh,
                self.current_point,
                self.triperiod,
                self.neighborhood_radius,
                self.current_frame_index,
                self.non_supporting_frame_set.size,
            )
        )

    def init_non_supporting_set(self):
        self.non_supporting_frame_set.size = int(
            toolpath_planner_init_non_supporting_set(
                self.frame_set.point,
                self.frame_set.normal,
                self.frame_set.phi_t,
                self.bvh,
                self.triperiod,
                self.neighborhood_radius,
                self.non_supporting_frame_set.value,
            )
        )

    def init_non_supporting_state_brute_force(self):
        toolpath_planner_init_non_supporting_state_brute_force(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.non_supporting_frame_set.value,
            self.triperiod,
            self.neighborhood_radius,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
        )

    def init_non_supporting_state(self):
        toolpath_planner_init_non_supporting_state(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.non_supporting_frame_set.value,
            self.bvh,
            self.triperiod,
            self.neighborhood_radius,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
        )

    def compute_cost(self):
        toolpath_planner_compute_cost(
            self.frame_data_set.state,
            self.frame_data_set.distance,
            self.non_supporting_frame_set.value,
            self.non_supporting_frame_set.size,
            self.current_frame_index,
            self.frame_data_set.cost,
        )

    def init_unaccessibility_brute_force(self):
        toolpath_planner_init_unaccessibility_brute_force(
            self.frame_set.point,
            self.frame_set.normal,
            self.non_supporting_frame_set.value,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
        )

    def init_unaccessibility(self):
        toolpath_planner_init_unaccessibility(
            self.frame_set.point,
            self.frame_set.normal,
            self.non_supporting_frame_set.value,
            self.bvh,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
        )

    def check_for_bidirectional_constraints(self):
        return toolpath_planner_check_for_bidirectional_constraints(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.bvh,
            self.triperiod,
            self.neighborhood_radius,
        )

    def init_atom_in_a_cycle(self):
        toolpath_planner_init_atom_in_a_cycle(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.cycle_is_visited,
            self.cycle_index_stack,
            self.cycle_p_jm1_stack,
            self.non_supporting_frame_set.value,
            self.bvh,
            self.triperiod,
            self.neighborhood_radius,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
        )

    def is_atom_i_reachable_by_a_constraint(self, i):
        return toolpath_planner_is_atom_i_reachable_by_a_constraint(
            self.frame_set.point,
            self.frame_set.normal,
            self.frame_set.phi_t,
            self.frame_data_set.state,
            self.cycle_is_visited,
            self.bvh,
            self.current_point,
            self.triperiod,
            self.neighborhood_radius,
            self.domain_diag,
            i,
        )

    def init_non_supporting_state_fast_track_option(self):
        if self.iteration_number == 0:
            toolpath_planner_init_non_supporting_state_iter_0(
                self.frame_set.point,
                self.frame_set.normal,
                self.frame_set.phi_t,
                self.non_supporting_frame_set.value,
                self.bvh,
                self.p_0,
                self.triperiod,
                self.neighborhood_radius,
                self.domain_diag,
                self.non_supporting_frame_set.size,
                self.frame_data_set.state,
                self.frame_data_set.distance,
            )
        elif self.iteration_number == 1:
            if self.fast_track:
                toolpath_planner_init_non_supporting_state_iter_1_fast_track(
                    self.frame_set.point,
                    self.frame_set.normal,
                    self.frame_set.phi_t,
                    self.non_supporting_frame_set.value,
                    self.bvh,
                    self.triperiod,
                    self.current_point,
                    self.neighborhood_radius,
                    self.domain_diag,
                    self.current_frame_index,
                    self.non_supporting_frame_set.size,
                    self.frame_data_set.state,
                    self.frame_data_set.distance,
                )
            else:
                toolpath_planner_init_non_supporting_state_iter_1(
                    self.frame_set.point,
                    self.frame_set.normal,
                    self.frame_set.phi_t,
                    self.non_supporting_frame_set.value,
                    self.bvh,
                    self.triperiod,
                    self.current_point,
                    self.neighborhood_radius,
                    self.domain_diag,
                    self.current_frame_index,
                    self.non_supporting_frame_set.size,
                    self.frame_data_set.state,
                    self.frame_data_set.distance,
                )
        else:
            if self.fast_track:
                toolpath_planner_init_non_supporting_state_iter_n_fast_track(
                    self.frame_set.point,
                    self.frame_set.normal,
                    self.frame_set.phi_t,
                    self.non_supporting_frame_set.value,
                    self.bvh,
                    self.triperiod,
                    self.current_point,
                    self.previous_point,
                    self.neighborhood_radius,
                    self.domain_diag,
                    self.current_frame_index,
                    self.non_supporting_frame_set.size,
                    self.frame_data_set.state,
                    self.frame_data_set.distance,
                )
            else:
                toolpath_planner_init_non_supporting_state_iter_n(
                    self.frame_set.point,
                    self.frame_set.normal,
                    self.frame_set.phi_t,
                    self.non_supporting_frame_set.value,
                    self.bvh,
                    self.triperiod,
                    self.current_point,
                    self.previous_point,
                    self.neighborhood_radius,
                    self.domain_diag,
                    self.current_frame_index,
                    self.non_supporting_frame_set.size,
                    self.frame_data_set.state,
                    self.frame_data_set.distance,
                )

    def find_best_next(self):
        ret_data = toolpath_planner_find_best_next(
            self.frame_data_set.state,
            self.frame_data_set.cost,
            self.non_supporting_frame_set.value,
            self.non_supporting_frame_set.size,
        ).to_numpy()
        next_index = int(ret_data[0])
        next_travel_type = int(ret_data[1])
        return next_index, next_travel_type

    def compute_cost_and_find_best_next(self):
        ret_data = toolpath_planner_compute_cost_and_find_best_next(
            self.frame_data_set.state,
            self.frame_data_set.distance,
            self.non_supporting_frame_set.value,
            self.non_supporting_frame_set.size,
            self.current_frame_index,
            self.frame_data_set.cost,
        ).to_numpy()
        next_index = int(ret_data[0])
        next_travel_type = int(ret_data[1])
        return next_index, next_travel_type

    def find_best_next_constrainer(self, constrained_index):
        ret_data = toolpath_planner_find_best_next_contrainer(
            self.frame_set.point,
            self.frame_set.normal,
            self.non_supporting_frame_set.value,
            constrained_index,
            self.non_supporting_frame_set.size,
            self.frame_data_set.state,
            self.frame_data_set.distance,
            self.frame_data_set.cost,
        )
        next_index = int(ret_data[0])
        next_travel_type = int(ret_data[1])
        return next_index, next_travel_type

    def update_toolpath(self, next_index: int, next_travel_type: int):
        ret_data = toolpath_planner_update_toolpath(
            self.frame_set.point,
            self.frame_set.normal,
            self.toolpath.point,
            self.toolpath.tool_orientation,
            self.toolpath.travel_type,
            self.bvh,
            self.travel_point,
            self.travel_normal,
            self.current_point,
            self.triperiod,
            self.iteration_number,
            self.current_frame_index,
            next_index,
            next_travel_type,
            self.toolpath.point_count,
        )
        self.previous_frame_index = int(ret_data[0])
        self.current_frame_index = int(ret_data[1])
        self.current_travel_type = int(ret_data[2])
        self.previous_point = np.array([ret_data[3], ret_data[4], ret_data[5]])
        self.current_point = np.array([ret_data[6], ret_data[7], ret_data[8]])
        self.current_normal_sph = np.array([ret_data[9], ret_data[10]])
        self.toolpath_point_count_previous = self.toolpath.point_count
        self.toolpath.point_count = int(ret_data[11])
        self.iteration_number = int(ret_data[12])


class FrameDataSet:
    def __init__(self):
        self.state = None
        self.distance = None
        self.cost = None

    def init(self, frame_set: frame3.Set):
        shape = frame_set.phi_t.shape
        self.state = ti.field(dtype=ti.u32, shape=shape)
        self.distance = ti.field(dtype=ti.f32, shape=shape)
        self.cost = ti.field(dtype=ti.f32, shape=shape)

    def to_numpy(self):
        state_numpy = self.state.to_numpy()
        distance_numpy = self.distance.to_numpy()
        cost_numpy = self.cost.to_numpy()

        dict_array = {}
        dict_array["state"] = state_numpy
        dict_array["distance"] = distance_numpy
        dict_array["cost"] = cost_numpy

        return dict_array

    def from_numpy(self, dict_array):
        self.state = ti.field(dtype=ti.u32, shape=dict_array["state"].shape)
        self.distance = ti.field(dtype=ti.f32, shape=dict_array["distance"].shape)
        self.cost = ti.field(dtype=ti.f32, shape=dict_array["cost"].shape)

        self.state.from_numpy(dict_array["state"])
        self.distance.from_numpy(dict_array["distance"])
        self.cost.from_numpy(dict_array["cost"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)


@ti.kernel
def toolpath_uplift(
    point: ti.types.ndarray(), normal: ti.types.ndarray(), height: float
):
    for i in range(point.shape[0]):
        normal_sph_i = ti.math.vec2(normal[i, 0], normal[i, 1])
        normal_i = direction.spherical_to_cartesian(normal_sph_i)
        point_i = ti.math.vec3(point[i, 0], point[i, 1], point[i, 2])
        point_i = point_i + normal_i * height
        for j in range(3):
            point[i, j] = point_i[j]


@ti.kernel
def toolpath_smooth_orientations(
    normal_in: ti.types.ndarray(), normal_out: ti.types.ndarray()
):
    # ti.loop_config(serialize=True)
    for i in range(normal_in.shape[0]):
        im1 = ti.max(i - 1, 0)
        ip1 = ti.min(i + 1, normal_in.shape[0] - 1)

        normal_sph_im1 = ti.math.vec2(normal_in[im1, 0], normal_in[im1, 1])
        normal_sph_i = ti.math.vec2(normal_in[i, 0], normal_in[i, 1])
        normal_sph_ip1 = ti.math.vec2(normal_in[ip1, 0], normal_in[ip1, 1])

        normal_im1 = direction.spherical_to_cartesian(normal_sph_im1)
        normal_i = direction.spherical_to_cartesian(normal_sph_i)
        normal_ip1 = direction.spherical_to_cartesian(normal_sph_ip1)

        normal_average = ti.math.normalize(normal_im1 + normal_i + normal_ip1)
        normal_average_sph = direction.cartesian_to_spherical(normal_average)
        normal_out[i, 0] = normal_average_sph[0]
        normal_out[i, 1] = normal_average_sph[1]


@ti.kernel
def toolpath_tesselate_orientation(
    point_in: ti.types.ndarray(),
    normal_in: ti.types.ndarray(),
    travel_type_in: ti.types.ndarray(),
    width_in: ti.types.ndarray(),
    height_in: ti.types.ndarray(),
    point_out: ti.types.ndarray(),
    normal_out: ti.types.ndarray(),
    travel_type_out: ti.types.ndarray(),
    width_out: ti.types.ndarray(),
    height_out: ti.types.ndarray(),
    max_angle_diff_radians: float,
) -> int:
    current_index = 0
    ti.loop_config(serialize=True)
    for i in range(normal_in.shape[0] - 1):
        ip1 = i + 1

        normal_sph_i = ti.math.vec2(normal_in[i, 0], normal_in[i, 1])
        normal_sph_ip1 = ti.math.vec2(normal_in[ip1, 0], normal_in[ip1, 1])

        normal_i = direction.spherical_to_cartesian(normal_sph_i)
        normal_ip1 = direction.spherical_to_cartesian(normal_sph_ip1)

        angle_i_ip1 = math.acos_safe(ti.math.dot(normal_i, normal_ip1))
        if angle_i_ip1 > max_angle_diff_radians:

            angle_subdivision = int(ti.ceil(angle_i_ip1 / max_angle_diff_radians))

            # Sample count is the number of points between the endpoints
            sample_point_count = angle_subdivision - 1

            point_i = ti.math.vec3(point_in[i, 0], point_in[i, 1], point_in[i, 2])
            point_ip1 = ti.math.vec3(
                point_in[ip1, 0], point_in[ip1, 1], point_in[ip1, 2]
            )
            vector = point_ip1 - point_i

            current_subindex = 0

            endpoint_subindex = sample_point_count + 1
            for j in range(endpoint_subindex):
                t = j / endpoint_subindex
                point_j = point_i + t * vector
                normal_j = direction.nlerp(normal_i, normal_ip1, t)
                width_j = ti.math.mix(width_in[i], width_in[ip1], t)
                height_j = ti.math.mix(height_in[i], height_in[ip1], t)

                global_index = current_index + current_subindex
                for k in ti.static(range(3)):
                    point_out[global_index, k] = point_j[k]
                for k in ti.static(range(2)):
                    normal_out[global_index, k] = direction.cartesian_to_spherical(
                        normal_j
                    )[k]
                if j == 0:
                    travel_type_out[global_index] = travel_type_in[i]
                else:
                    travel_type_out[global_index] = travel_type_in[ip1]
                width_out[global_index] = width_j
                height_out[global_index] = height_j

                current_subindex += 1

            # Update current index with the sub index
            current_index += current_subindex
        else:
            for k in ti.static(range(3)):
                point_out[current_index, k] = point_in[i, k]
            for k in ti.static(range(2)):
                normal_out[current_index, k] = normal_in[i, k]
            travel_type_out[current_index] = travel_type_in[i]
            width_out[current_index] = width_in[i]
            height_out[current_index] = height_in[i]
            current_index += 1

    for k in ti.static(range(3)):
        point_out[current_index, k] = point_in[normal_in.shape[0] - 1, k]
    for k in ti.static(range(2)):
        normal_out[current_index, k] = normal_in[normal_in.shape[0] - 1, k]
    travel_type_out[current_index] = travel_type_in[normal_in.shape[0] - 1]
    width_out[current_index] = width_in[normal_in.shape[0] - 1]
    height_out[current_index] = height_in[normal_in.shape[0] - 1]
    current_index += 1
    # return the number of points
    return current_index


@ti.kernel
def toolpath_max_diff_orientation(normal: ti.types.ndarray()) -> tuple[float, int]:
    angle_max = 0.0
    index_max = 0
    ti.loop_config(serialize=True)
    for i in range(normal.shape[0] - 1):
        ip1 = i + 1

        normal_sph_i = ti.math.vec2(normal[i, 0], normal[i, 1])
        normal_sph_ip1 = ti.math.vec2(normal[ip1, 0], normal[ip1, 1])

        normal_i = direction.spherical_to_cartesian(normal_sph_i)
        normal_ip1 = direction.spherical_to_cartesian(normal_sph_ip1)

        angle_i_ip1 = math.acos_safe(ti.math.dot(normal_i, normal_ip1))
        if angle_i_ip1 > angle_max:
            angle_max = angle_i_ip1
            index_max = i

    return angle_max, index_max


@ti.kernel
def toolpath_smooth_points(
    point_in: ti.types.ndarray(),
    travel_type: ti.types.ndarray(),
    point_out: ti.types.ndarray(),
):
    # ti.loop_config(serialize=True)
    for i in range(point_in.shape[0]):
        im1 = ti.max(i - 1, 0)
        ip1 = ti.min(i + 1, point_in.shape[0] - 1)

        smooth_point_i = (
            travel_type[i] == TRAVEL_TYPE_DEPOSITION
            and travel_type[ip1] == TRAVEL_TYPE_DEPOSITION
            and i != point_in.shape[0] - 1
            and i != 0
        )
        if not smooth_point_i:
            point_out[i, 0] = point_in[i, 0]
            point_out[i, 1] = point_in[i, 1]
            point_out[i, 2] = point_in[i, 2]
            continue

        point_im1 = ti.math.vec3(point_in[im1, 0], point_in[im1, 1], point_in[im1, 2])
        point_i = ti.math.vec3(point_in[i, 0], point_in[i, 1], point_in[i, 2])
        point_ip1 = ti.math.vec3(point_in[ip1, 0], point_in[ip1, 1], point_in[ip1, 2])

        average_point = (point_im1 + point_i + point_ip1) / 3.0

        point_out[i, 0] = average_point[0]
        point_out[i, 1] = average_point[1]
        point_out[i, 2] = average_point[2]


@ti.kernel
def toolpath_planner_init_non_supporting_set_brute_force(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    non_supporting_set: ti.template(),
) -> int:
    set_size = 0
    ti.loop_config(serialize=True)
    for i in range(phi_t.shape[0]):
        is_masked_i = ti.math.isnan(point[i]).any()
        if is_masked_i:
            continue
        is_non_supporting_i = frame_data_set_is_non_supporting_brute_force_i(
            point, normal, phi_t, i, triperiod, neighborhood_radius
        )
        if is_non_supporting_i:
            set_insert_u32(non_supporting_set, set_size, ti.u32(i))
            set_size += 1

    return set_size


@ti.kernel
def toolpath_planner_init_non_supporting_set(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    non_supporting_set: ti.template(),
) -> int:
    set_size = 0
    ti.loop_config(serialize=True)
    for i in range(phi_t.shape[0]):
        is_masked_i = ti.math.isnan(point[i]).any()
        if is_masked_i:
            continue

        is_non_supporting_i = frame_data_set_is_non_supporting_i(
            point, normal, phi_t, bvh, i, triperiod, neighborhood_radius
        )
        if is_non_supporting_i:
            set_insert_u32(non_supporting_set, set_size, ti.u32(i))
            set_size += 1

    return set_size


@ti.kernel
def toolpath_planner_init_non_supporting_state_brute_force(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    non_supporting_set_size: int,
    state: ti.template(),
):
    # Consider removing the serialization
    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        i = int(non_supporting_set[set_index])

        neighborhood_config_i = (
            frame_data_set_non_supporting_compute_neighborhood_config_brute_force_i(
                point, normal, phi_t, i, triperiod, neighborhood_radius
            )
        )

        state[i] = ti.u32(0)
        state[i] = atom_set_is_not_supporting(state[i])
        state[i] = state[i] | neighborhood_config_i


@ti.kernel
def toolpath_planner_update_non_supporting_set_brute_force(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    non_supporting_set: ti.template(),
    p_i: ti.math.vec3,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    i: int,
    non_supporting_set_size: int,
) -> int:
    set_size = non_supporting_set_size

    # Remove i from the non supporting set
    set_remove_u32(non_supporting_set, set_size, ti.u32(i))
    set_size = set_size - 1

    neighborhood_radius_sqr = neighborhood_radius * neighborhood_radius

    # In the neighborhood, update the states
    ti.loop_config(serialize=True)
    for j in range(point.shape[0]):
        if i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        p_ij = p_i - p_j
        p_ij_length_sqr = ti.math.dot(p_ij, p_ij)
        if p_ij_length_sqr > neighborhood_radius_sqr:
            continue

        is_non_supporting_j = frame_data_set_is_non_supporting_brute_force_i(
            point, normal, phi_t, j, triperiod, neighborhood_radius
        )
        is_non_supporting_j_tm1 = atom_is_not_supporting(state[j])
        if is_non_supporting_j and not is_non_supporting_j_tm1:
            # j is now a new non supporting atom
            set_insert_u32(non_supporting_set, set_size, ti.u32(j))
            set_size = set_size + 1

    return set_size


@ti.kernel
def toolpath_planner_update_non_supporting_set(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    p_i: ti.math.vec3,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    i: int,
    non_supporting_set_size: int,
) -> int:
    set_size = non_supporting_set_size

    # Remove i from the non supporting set
    set_remove_u32(non_supporting_set, set_size, ti.u32(i))
    set_size = set_size - 1

    # It is important to update the non supporting state correctly, so the neighborhood radius is increased
    stack = bvh.points_in_sphere_local_stack(p_i, neighborhood_radius * 1.2)

    # In the neighborhood, update the states
    ti.loop_config(serialize=True)
    for stack_index in range(STACK_SIZE_MAX):
        j = stack[stack_index]

        if j == -1 or i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        is_non_supporting_j = frame_data_set_is_non_supporting_i(
            point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
        )
        is_non_supporting_j_tm1 = atom_is_not_supporting(state[j])
        if is_non_supporting_j and not is_non_supporting_j_tm1:
            # j is now a new non supporting atom
            set_insert_u32(non_supporting_set, set_size, ti.u32(j))
            set_size = set_size + 1
            # By default, the new atom is set to unaccessible
            state[j] = atom_set_is_unaccessible(state[j])

    return set_size


@ti.kernel
def toolpath_planner_init_non_supporting_state(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    non_supporting_set_size: int,
    state: ti.template(),
):
    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        i = int(non_supporting_set[set_index])

        neighborhood_config_i = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, i, triperiod, neighborhood_radius
            )
        )

        state[i] = ti.u32(0)
        state[i] = atom_set_is_not_supporting(state[i])
        state[i] = state[i] | neighborhood_config_i


@ti.kernel
def toolpath_planner_init_non_supporting_state_iter_0(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    p_0: ti.math.vec3,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
):
    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])
        p_j = point[j]
        n_j = direction.spherical_to_cartesian(normal[j])

        neighborhood_config_j = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
            )
        )

        if atom_is_wall(state[j]):
            state[j] = atom_set_is_wall(ti.u32(0))
        else:
            state[j] = ti.u32(0)

        state[j] = atom_set_is_not_supporting(state[j])
        state[j] = state[j] | neighborhood_config_j

        # Update distance
        p_ij = p_j - p_0
        distance[j] = ti.math.length(p_ij) / domain_diag

        is_unaccessible_j = bvh.any_pt_in_cone(
            cone_origin=p_j + n_j * 0.001,
            cone_direction=n_j,
            cone_angle=NOZZLE_CONE_ANGLE * 0.5,
        )

        if is_unaccessible_j:
            state[j] = atom_set_is_unaccessible(state[j])


@ti.kernel
def toolpath_planner_init_non_supporting_state_iter_1(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    p_i: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    i: int,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
):
    spherical_basis_i = ti.math.vec3(normal[i], phi_t[i])

    a_i = math.vec6(p_i, spherical_basis_i)

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])
        p_j = point[j]
        n_j = direction.spherical_to_cartesian(normal[j])

        neighborhood_config_j = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
            )
        )

        if atom_is_wall(state[j]):
            state[j] = atom_set_is_wall(ti.u32(0))
        else:
            state[j] = ti.u32(0)

        state[j] = atom_set_is_not_supporting(state[j])
        state[j] = state[j] | neighborhood_config_j

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if region_number == 0:
            state[j] = atom_set_is_in_front(state[j])
        elif region_number == 1:
            state[j] = atom_set_is_behind(state[j])
        elif region_number == 5:
            state[j] = atom_set_is_below(state[j])
        elif region_number == 2 or region_number == 3:
            state[j] = atom_set_is_on_the_left_or_right(state[j])

        # Update distance
        p_ij = p_j - p_i
        distance[j] = ti.math.length(p_ij) / domain_diag

        is_unaccessible_j = bvh.any_pt_in_cone(
            cone_origin=p_j + n_j * 0.001,
            cone_direction=n_j,
            cone_angle=NOZZLE_CONE_ANGLE * 0.5,
        )

        if is_unaccessible_j:
            state[j] = atom_set_is_unaccessible(state[j])
        else:
            state[j] = atom_unset_is_unaccessible(state[j])


@ti.kernel
def toolpath_planner_init_non_supporting_state_iter_1_fast_track(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    p_i: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    i: int,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
):
    spherical_basis_i = ti.math.vec3(normal[i], phi_t[i])

    a_i = math.vec6(p_i, spherical_basis_i)

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])
        p_j = point[j]

        # Fast track means we keep the previous state, except those which are
        # recomputed during the fast track.
        state[j] = atom_unset_any_in_front(state[j])
        state[j] = atom_unset_any_behind(state[j])
        state[j] = atom_unset_any_left(state[j])
        state[j] = atom_unset_any_right(state[j])
        state[j] = atom_unset_is_in_front(state[j])
        state[j] = atom_unset_is_behind(state[j])
        state[j] = atom_unset_is_below(state[j])
        state[j] = atom_unset_is_on_the_left_or_right(state[j])

        # Update the local state
        state[j] = atom_set_is_not_supporting(state[j])
        neighborhood_config_j = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
            )
        )
        state[j] = state[j] | neighborhood_config_j

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if region_number == 0:
            state[j] = atom_set_is_in_front(state[j])
        elif region_number == 1:
            state[j] = atom_set_is_behind(state[j])
        elif region_number == 5:
            state[j] = atom_set_is_below(state[j])
        elif region_number == 2 or region_number == 3:
            state[j] = atom_set_is_on_the_left_or_right(state[j])

        # Update distance
        p_ij = p_j - p_i
        distance[j] = ti.math.length(p_ij) / domain_diag


@ti.kernel
def toolpath_planner_init_non_supporting_state_iter_n(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    p_i: ti.math.vec3,
    p_im1: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    i: int,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
):
    spherical_basis_i = ti.math.vec3(normal[i], phi_t[i])

    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))

    tbn_i = basis3.from_spherical(spherical_basis_i)
    tangent_i = ti.math.vec3(tbn_i[0, 0], tbn_i[1, 0], tbn_i[2, 0])

    path_direction = p_i - p_im1
    tangent_path_inversed = False
    if ti.math.dot(tangent_i, path_direction) < 0.0:
        tangent_path_inversed = True

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])
        p_j = point[j]
        n_j = direction.spherical_to_cartesian(normal[j])

        neighborhood_config_j = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
            )
        )

        if atom_is_wall(state[j]):
            state[j] = atom_set_is_wall(ti.u32(0))
        else:
            state[j] = ti.u32(0)

        state[j] = atom_set_is_not_supporting(state[j])
        state[j] = state[j] | neighborhood_config_j

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if (
            region_number == 0
            and not tangent_path_inversed
            or region_number == 1
            and tangent_path_inversed
        ):
            state[j] = atom_set_is_in_front(state[j])
        elif (
            region_number == 1
            and not tangent_path_inversed
            or region_number == 0
            and tangent_path_inversed
        ):
            state[j] = atom_set_is_behind(state[j])
        elif region_number == 5:
            state[j] = atom_set_is_below(state[j])
        elif region_number == 2 or region_number == 3:
            state[j] = atom_set_is_on_the_left_or_right(state[j])

        # Update distance
        p_ij = p_j - p_i
        distance[j] = ti.math.length(p_ij) / domain_diag

        is_unaccessible_j = bvh.any_pt_in_cone(
            cone_origin=p_j + n_j * 0.001,
            cone_direction=n_j,
            cone_angle=NOZZLE_CONE_ANGLE * 0.5,
        )

        if is_unaccessible_j:
            state[j] = atom_set_is_unaccessible(state[j])


@ti.kernel
def toolpath_planner_find_best_next_contrainer(
    point: ti.template(),
    normal: ti.template(),
    non_supporting_set: ti.template(),
    constrained_index: int,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
    cost: ti.template(),
) -> ti.math.ivec2:
    p_i = point[constrained_index]
    n_i = direction.spherical_to_cartesian(normal[constrained_index])

    COS_HALF_ANGLE = ti.math.min(
        SUPPORTING_REGION_COS_HALF_ANGLE, NOZZLE_COS_HALF_ANGLE
    )

    cost_min = limits.f32_max
    cost_min_arg = -1

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])

        if atom_is_unaccessible(state[j]):
            continue

        any_in_front_j = atom_any_in_front(state[j])
        any_behind_j = atom_any_behind(state[j])
        any_on_the_left_j = atom_any_left(state[j])
        any_on_the_right_j = atom_any_right(state[j])
        is_in_cycle_j = atom_is_in_cycle(state[j])

        tangent_neighborhood_config_j = 0
        if any_in_front_j or any_behind_j:
            tangent_neighborhood_config_j = 1
        if any_in_front_j and any_behind_j:
            tangent_neighborhood_config_j = 2

        bitangent_neighborhood_status_j = 0
        if any_on_the_left_j or any_on_the_right_j:
            bitangent_neighborhood_status_j = 1
        if any_on_the_left_j and any_on_the_right_j:
            bitangent_neighborhood_status_j = 2

        d_i = distance[j]

        is_constrainer = cone.is_point_inside(point[j], p_i, n_i, COS_HALF_ANGLE)

        cost[j] = d_i
        if tangent_neighborhood_config_j == 2 and not is_in_cycle_j:
            cost[j] = cost[j] + 1.0
        if bitangent_neighborhood_status_j == 2:
            cost[j] = cost[j] + 1.0
        if not is_constrainer:
            cost[j] = cost[j] + 3.0

        if cost[j] < cost_min:
            cost_min = cost[j]
            cost_min_arg = j

    return ti.math.ivec2(cost_min_arg, TRAVEL_TYPE_NO_DEPOSITION)


@ti.kernel
def toolpath_planner_init_non_supporting_state_iter_n_fast_track(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    p_i: ti.math.vec3,
    p_im1: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    i: int,
    non_supporting_set_size: int,
    state: ti.template(),
    distance: ti.template(),
):
    spherical_basis_i = ti.math.vec3(normal[i], phi_t[i])

    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))

    tbn_i = basis3.from_spherical(spherical_basis_i)
    tangent_i = ti.math.vec3(tbn_i[0, 0], tbn_i[1, 0], tbn_i[2, 0])

    path_direction = p_i - p_im1
    tangent_path_inversed = False
    if ti.math.dot(tangent_i, path_direction) < 0.0:
        tangent_path_inversed = True

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])
        p_j = point[j]

        # Fast track means we keep the previous state, except those which are
        # recomputed during the fast track.
        state[j] = atom_unset_any_in_front(state[j])
        state[j] = atom_unset_any_behind(state[j])
        state[j] = atom_unset_any_left(state[j])
        state[j] = atom_unset_any_right(state[j])
        state[j] = atom_unset_is_in_front(state[j])
        state[j] = atom_unset_is_behind(state[j])
        state[j] = atom_unset_is_below(state[j])
        state[j] = atom_unset_is_on_the_left_or_right(state[j])

        # Update the local state
        state[j] = atom_set_is_not_supporting(state[j])
        neighborhood_config_j = (
            frame_data_set_non_supporting_compute_neighborhood_config_i(
                point, normal, phi_t, bvh, j, triperiod, neighborhood_radius
            )
        )
        state[j] = state[j] | neighborhood_config_j

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if (
            region_number == 0
            and not tangent_path_inversed
            or region_number == 1
            and tangent_path_inversed
        ):
            state[j] = atom_set_is_in_front(state[j])
        elif (
            region_number == 1
            and not tangent_path_inversed
            or region_number == 0
            and tangent_path_inversed
        ):
            state[j] = atom_set_is_behind(state[j])
        elif region_number == 5:
            state[j] = atom_set_is_below(state[j])
        elif region_number == 2 or region_number == 3:
            state[j] = atom_set_is_on_the_left_or_right(state[j])

        # Update distance
        p_ij = p_j - p_i
        distance[j] = ti.math.length(p_ij) / domain_diag


@ti.kernel
def toolpath_planner_init_unaccessibility_brute_force(
    point: ti.template(),
    normal: ti.template(),
    non_supporting_set: ti.template(),
    non_supporting_set_size: int,
    state: ti.template(),
):
    for set_index in range(non_supporting_set_size):
        i = int(non_supporting_set[set_index])

        is_unaccessible_i = frame3.frame_set_any_point_inside_cone_i_brute_force(
            point, normal, i, NOZZLE_COS_HALF_ANGLE
        )

        if is_unaccessible_i:
            state[i] = atom_set_is_unaccessible(state[i])
        else:
            state[i] = atom_unset_is_unaccessible(state[i])


@ti.kernel
def toolpath_planner_init_unaccessibility(
    point: ti.template(),
    normal: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    non_supporting_set_size: int,
    state: ti.template(),
):
    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])

        p_j = point[j]
        n_j = direction.spherical_to_cartesian(normal[j])

        is_unaccessible_j = bvh.any_pt_in_cone(
            cone_origin=p_j + n_j * 0.001,
            cone_direction=n_j,
            cone_angle=NOZZLE_CONE_ANGLE * 0.5,
        )

        if is_unaccessible_j:
            state[j] = atom_set_is_unaccessible(state[j])


@ti.kernel
def toolpath_planner_compute_cost(
    state: ti.template(),
    distance: ti.template(),
    non_supporting_set: ti.template(),
    non_supporting_set_size: int,
    i: int,
    cost: ti.template(),
):

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])

        if atom_is_unaccessible(state[j]):
            continue

        is_in_front_j = atom_is_in_front(state[j])
        is_behind_j = atom_is_behind(state[j])
        any_in_front_j = atom_any_in_front(state[j])
        any_behind_j = atom_any_behind(state[j])
        any_on_the_left_j = atom_any_left(state[j])
        any_on_the_right_j = atom_any_right(state[j])
        is_in_cycle_j = atom_is_in_cycle(state[j])

        tangent_neighborhood_config_j = 0
        if any_in_front_j or any_behind_j:
            tangent_neighborhood_config_j = 1
        if any_in_front_j and any_behind_j:
            tangent_neighborhood_config_j = 2

        bitangent_neighborhood_status_j = 0
        if any_on_the_left_j or any_on_the_right_j:
            bitangent_neighborhood_status_j = 1
        if any_on_the_left_j and any_on_the_right_j:
            bitangent_neighborhood_status_j = 2

        d_i = distance[j]

        if is_in_front_j:
            cost[j] = d_i + 0.0
        elif is_behind_j:
            cost[j] = d_i + 1.0
        else:
            cost[j] = d_i + 2.0

            if tangent_neighborhood_config_j == 2 and not is_in_cycle_j:
                cost[j] = cost[j] + 1.0
            if bitangent_neighborhood_status_j == 2:
                cost[j] = cost[j] + 1.0


@ti.kernel
def toolpath_planner_compute_cost_and_find_best_next(
    state: ti.template(),
    distance: ti.template(),
    non_supporting_set: ti.template(),
    non_supporting_set_size: int,
    i: int,
    cost: ti.template(),
) -> ti.math.ivec2:
    cost_min = limits.f32_max
    cost_min_arg = -1

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        j = int(non_supporting_set[set_index])

        if atom_is_unaccessible(state[j]):
            continue

        is_in_front_j = atom_is_in_front(state[j])
        is_behind_j = atom_is_behind(state[j])
        is_on_the_left_or_right_j = atom_is_on_the_left_or_right(state[j])
        any_in_front_j = atom_any_in_front(state[j])
        any_behind_j = atom_any_behind(state[j])
        any_on_the_left_j = atom_any_left(state[j])
        any_on_the_right_j = atom_any_right(state[j])
        is_in_cycle_j = atom_is_in_cycle(state[j])

        tangent_neighborhood_config_j = 0
        if any_in_front_j or any_behind_j:
            tangent_neighborhood_config_j = 1
        if any_in_front_j and any_behind_j:
            tangent_neighborhood_config_j = 2

        bitangent_neighborhood_status_j = 0
        if any_on_the_left_j or any_on_the_right_j:
            bitangent_neighborhood_status_j = 1
        if any_on_the_left_j and any_on_the_right_j:
            bitangent_neighborhood_status_j = 2

        d_i = distance[j]

        if is_in_front_j:
            cost[j] = d_i + 0.0
        elif is_behind_j:
            cost[j] = d_i + 1.0
        else:
            cost[j] = d_i + 2.0

            if tangent_neighborhood_config_j == 2 and not is_in_cycle_j:
                cost[j] = cost[j] + 1.0
            if bitangent_neighborhood_status_j == 2:
                cost[j] = cost[j] + 1.0

        if cost[j] < cost_min:
            cost_min = cost[j]
            cost_min_arg = j

    travel_type = TRAVEL_TYPE_NO_DEPOSITION
    if cost_min < 2.0:
        travel_type = TRAVEL_TYPE_DEPOSITION
    return ti.math.ivec2(cost_min_arg, travel_type)


@ti.func
def toolpath_planner_is_atom_i_in_a_cycle(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    cycle_is_visited: ti.template(),
    cycle_index_stack: ti.template(),
    cycle_p_jm1_stack: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    i: int,
):
    """
    Assume i to be non masked and non supporting
    """
    cycle_is_visited_size = 0

    cycle_stack_size = 0
    cycle_index_stack[cycle_stack_size] = i
    cycle_p_jm1_stack[cycle_stack_size] = ti.math.vec3(-1.0)
    cycle_stack_size = cycle_stack_size + 1

    is_i_in_a_cycle = False

    while not is_i_in_a_cycle and cycle_stack_size != 0:
        # Take last element in the stack
        # Assume that atoms in the stack are unmasked and non supporting and accessible
        j = cycle_index_stack[cycle_stack_size - 1]
        p_jm1 = cycle_p_jm1_stack[cycle_stack_size - 1]
        cycle_stack_size = cycle_stack_size - 1

        if not set_find_u32(cycle_is_visited, cycle_is_visited_size, ti.u32(j)):
            set_insert_u32(cycle_is_visited, cycle_is_visited_size, ti.u32(j))
            cycle_is_visited_size = cycle_is_visited_size + 1

            spherical_basis_j = ti.math.vec3(normal[j], phi_t[j])
            a_j = math.vec6(point[j], ti.math.vec3(normal[j], phi_t[j]))
            tbn_j = basis3.from_spherical(spherical_basis_j)
            tangent_j = ti.math.vec3(tbn_j[0, 0], tbn_j[1, 0], tbn_j[2, 0])
            tangent_path_inversed = False
            if p_jm1.x != -1.0:
                path_direction = point[j] - p_jm1
                if ti.math.dot(tangent_j, path_direction) < 0.0:
                    tangent_path_inversed = True

            # Iterate through all adjacent atoms of j
            stack = bvh.points_in_sphere_local_stack(point[j], neighborhood_radius)

            ti.loop_config(serialize=True)
            for stack_index in range(STACK_SIZE_MAX):
                k = stack[stack_index]

                if k == -1 or j == k:
                    continue

                if ti.math.isnan(point[k]).any():
                    continue

                if atom_is_unaccessible(state[k]) or not atom_is_not_supporting(
                    state[k]
                ):
                    continue

                region_number_jk = atom_get_region_number(point[k], a_j, triperiod)

                is_k_in_front_of_j = (
                    region_number_jk == 0
                    and not tangent_path_inversed
                    or region_number_jk == 1
                    and tangent_path_inversed
                )

                if is_k_in_front_of_j:
                    if k == i:
                        is_i_in_a_cycle = True
                        break
                    else:
                        if not set_find_u32(
                            cycle_is_visited, cycle_is_visited_size, ti.u32(k)
                        ):
                            cycle_index_stack[cycle_stack_size] = k
                            cycle_p_jm1_stack[cycle_stack_size] = point[j]
                            cycle_stack_size = cycle_stack_size + 1

    return is_i_in_a_cycle


@ti.kernel
def toolpath_planner_is_atom_i_reachable_by_a_constraint(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    path_is_visited: ti.template(),
    bvh: ti.template(),
    p_jm1_p: ti.math.vec3,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    domain_diag: float,
    i: int,
) -> int:
    """
    Assume i to be non masked and non supporting
    """
    path_is_visited_size = 0

    constraint_reached = False
    i_reachable_by_constraint_index = -1

    jp1 = i
    p_jm1 = p_jm1_p
    set_insert_u32(path_is_visited, path_is_visited_size, ti.u32(i))
    path_is_visited_size = path_is_visited_size + 1

    while not constraint_reached and jp1 != -1:
        # Assume the next atom is unmasked, non supporting and accessible
        j = jp1
        p_j = point[j]

        cost_min_jk = limits.f32_max
        cost_min_arg_k = -1

        spherical_basis_j = ti.math.vec3(normal[j], phi_t[j])
        a_j = math.vec6(point[j], ti.math.vec3(normal[j], phi_t[j]))
        tbn_j = basis3.from_spherical(spherical_basis_j)
        tangent_j = ti.math.vec3(tbn_j[0, 0], tbn_j[1, 0], tbn_j[2, 0])

        tangent_path_inversed = False

        path_direction = point[j] - p_jm1
        if ti.math.dot(tangent_j, path_direction) < 0.0:
            tangent_path_inversed = True

        # Iterate through all adjacent atoms of j
        stack = bvh.points_in_sphere_local_stack(point[j], neighborhood_radius)

        ti.loop_config(serialize=True)
        for stack_index in range(STACK_SIZE_MAX):
            k = stack[stack_index]

            if k == -1 or j == k:
                continue

            if ti.math.isnan(point[k]).any():
                continue

            is_k_constrained = atom_is_unaccessible(
                state[k]
            ) or not atom_is_not_supporting(state[k])

            region_number_jk = atom_get_region_number(point[k], a_j, triperiod)

            is_k_in_front_of_j = (
                region_number_jk == 0
                and not tangent_path_inversed
                or region_number_jk == 1
                and tangent_path_inversed
            )
            is_k_behind_j = (
                region_number_jk == 1
                and not tangent_path_inversed
                or region_number_jk == 0
                and tangent_path_inversed
            )
            is_k_adjacent_to_j = (
                region_number_jk == 0
                or region_number_jk == 1
                or region_number_jk == 2
                or region_number_jk == 3
            )
            distance_jk = ti.math.length(point[k] - p_j) / domain_diag
            cost_jk = limits.f32_max
            if not is_k_constrained and (is_k_in_front_of_j or is_k_behind_j):
                is_k_visited = set_find_u32(
                    path_is_visited, path_is_visited_size, ti.u32(k)
                )
                if not is_k_visited:
                    if is_k_in_front_of_j:
                        cost_jk = distance_jk + 0.0
                    elif is_k_behind_j:
                        cost_jk = distance_jk + 1.0

            if cost_jk < cost_min_jk:
                cost_min_jk = cost_jk
                cost_min_arg_k = k

            if is_k_constrained and is_k_adjacent_to_j:
                constraint_reached = True
                i_reachable_by_constraint_index = k
                break

        jp1 = cost_min_arg_k
        p_jm1 = p_j
        set_insert_u32(path_is_visited, path_is_visited_size, ti.u32(jp1))
        path_is_visited_size = path_is_visited_size + 1

    return i_reachable_by_constraint_index


@ti.kernel
def toolpath_planner_init_atom_in_a_cycle(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    cycle_is_visited: ti.template(),
    cycle_index_stack: ti.template(),
    cycle_p_jm1_stack: ti.template(),
    non_supporting_set: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
    non_supporting_set_size: int,
    state: ti.template(),
):
    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        i = int(non_supporting_set[set_index])

        if atom_is_unaccessible(state[i]):
            continue

        if toolpath_planner_is_atom_i_in_a_cycle(
            point,
            normal,
            phi_t,
            state,
            cycle_is_visited,
            cycle_index_stack,
            cycle_p_jm1_stack,
            bvh,
            triperiod,
            neighborhood_radius,
            i,
        ):
            state[i] = atom_set_is_in_cycle(state[i])


@ti.kernel
def toolpath_planner_check_for_bidirectional_constraints(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    bvh: ti.template(),
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> int:
    any_bidirectional_constraint = False
    ti.loop_config(serialize=True)
    for i in range(phi_t.shape[0]):
        is_masked_i = ti.math.isnan(point[i]).any()
        if is_masked_i:
            continue

        p_i = point[i]
        a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))
        n_i = direction.spherical_to_cartesian(normal[i])

        stack = bvh.points_in_sphere_local_stack(p_i, neighborhood_radius)

        for stack_index in range(STACK_SIZE_MAX):
            j = stack[stack_index]

            if j == -1 or i == j:
                continue

            p_j = point[j]
            if ti.math.isnan(p_j).any():
                continue

            a_j = math.vec6(p_j, ti.math.vec3(normal[j], phi_t[j]))
            n_j = direction.spherical_to_cartesian(normal[j])

            region_number_ij = atom_get_region_number(p_j, a_i, triperiod)
            is_inside_cone_ij = cone.is_point_inside(
                p_j, p_i + n_i * 0.001, n_i, NOZZLE_COS_HALF_ANGLE
            )

            region_number_ji = atom_get_region_number(p_i, a_j, triperiod)
            is_inside_cone_ji = cone.is_point_inside(
                p_i, p_j + n_j * 0.001, n_j, NOZZLE_COS_HALF_ANGLE
            )

            j_constrains_i = region_number_ij == 4 or is_inside_cone_ij
            i_constrains_j = region_number_ji == 4 or is_inside_cone_ji

            if j_constrains_i and i_constrains_j:
                print(f"Atom {i} and {j} have a bidirectional constraint.")
                any_bidirectional_constraint = True

    return any_bidirectional_constraint


@ti.kernel
def toolpath_planner_find_best_next(
    state: ti.template(),
    cost: ti.template(),
    non_supporting_set: ti.template(),
    non_supporting_set_size: int,
) -> ti.math.ivec2:
    cost_min = limits.f32_max
    cost_min_arg = -1

    ti.loop_config(serialize=True)
    for set_index in range(non_supporting_set_size):
        i = int(non_supporting_set[set_index])

        if atom_is_unaccessible(state[i]):
            continue

        if cost[i] < cost_min:
            cost_min = cost[i]
            cost_min_arg = i

    # If cost in [0., 2.), travel type: deposition
    travel_type = TRAVEL_TYPE_NO_DEPOSITION
    if cost_min < 2.0:
        travel_type = TRAVEL_TYPE_DEPOSITION
    return ti.math.ivec2(cost_min_arg, travel_type)


@ti.kernel
def toolpath_planner_update_toolpath(
    frame_set_point: ti.template(),
    frame_set_normal: ti.template(),
    toolpath_point: ti.template(),
    toolpath_tool_orientation: ti.template(),
    toolpath_travel_type: ti.template(),
    bvh: ti.template(),
    travel_point: ti.template(),
    travel_normal: ti.template(),
    current_point_p: ti.math.vec3,
    triperiod: ti.math.vec3,
    iteration_number: int,
    current_frame_index_p: int,
    next_index: int,
    next_travel_type: int,
    toolpath_point_count_p: int,
) -> math.vec13:
    ret_data = math.vec13(0.0)
    toolpath_point_count = toolpath_point_count_p

    previous_frame_index = current_frame_index_p
    current_frame_index = next_index
    current_travel_type = next_travel_type
    ret_data[0] = previous_frame_index
    ret_data[1] = current_frame_index
    ret_data[2] = current_travel_type

    im1 = previous_frame_index
    i = current_frame_index

    previous_point = current_point_p
    current_point = frame_set_point[i]
    for comp_index in ti.static(range(3)):
        ret_data[3 + comp_index] = previous_point[comp_index]
        ret_data[6 + comp_index] = current_point[comp_index]

    previous_normal_sph = ti.math.vec2(-1.0)
    if iteration_number != 0:
        previous_normal_sph = frame_set_normal[im1]

    current_normal_sph = frame_set_normal[i]
    for comp_index in ti.static(range(2)):
        ret_data[9 + comp_index] = current_normal_sph[comp_index]

    # Travel collision avoidance
    if current_travel_type == TRAVEL_TYPE_NO_DEPOSITION and iteration_number != 0:
        deposition_width = triperiod[1]
        layer_height = triperiod[2]

        p_im1_i = current_point - previous_point
        travel_length = ti.math.length(p_im1_i)
        # For travel, we sample the space at the deposition width * 0.2
        space_sampling = triperiod[1] * 0.2
        travel_edge_count = int(ti.math.ceil(travel_length / space_sampling))
        current_normal = direction.spherical_to_cartesian(current_normal_sph)
        previous_normal = direction.spherical_to_cartesian(previous_normal_sph)

        # If travel length is greater than deposition width times two
        if travel_length > deposition_width * 2.0:
            travel_point_count = 0
            # It is true to always liftup, put false to not uplift when not necessary
            any_collision = True
            previous_point_in_travel = previous_point

            for j in range(travel_edge_count - 1):
                u = (j + 1.0) / travel_edge_count
                remaining_edge_count = travel_edge_count - j
                u_current = 1.0 / remaining_edge_count

                p_im1_i = current_point - previous_point_in_travel
                point_j = previous_point_in_travel + p_im1_i * u_current
                normal_j = direction.nlerp(previous_normal, current_normal, u)
                point_offset = normal_j * layer_height * 0.5

                # Check if point j is colliding with the part
                is_collinding_j = bvh.any_pt_in_cone(
                    point_j - point_offset,
                    normal_j,
                    NOZZLE_CONE_ANGLE * 0.5,
                )

                while is_collinding_j:
                    any_collision = True
                    point_j = point_j + point_offset
                    is_collinding_j = bvh.any_pt_in_cone(
                        point_j - point_offset,
                        normal_j,
                        NOZZLE_CONE_ANGLE * 0.5,
                    )

                previous_point_in_travel = point_j
                travel_point[travel_point_count] = point_j
                travel_normal[travel_point_count] = normal_j
                travel_point_count = travel_point_count + 1

            if any_collision:
                # Lift the points
                # It is better to have too much lift than not enough ;)
                for j in range(travel_point_count):
                    p_j = travel_point[j]
                    n_j = travel_normal[j]
                    point_offset = n_j * layer_height * 2.0

                    p_j = p_j + point_offset
                    travel_point[j] = p_j

                # Smooth the points
                for _ in range(2):
                    for j in range(travel_point_count):
                        p_j = travel_point[j]
                        n_j = travel_normal[j]

                        jp1 = ti.min(j + 1, travel_point_count - 1)
                        jm1 = ti.max(j - 1, 0)
                        p_jp1 = travel_point[jp1]
                        p_jm1 = travel_point[jm1]
                        travel_point[j].z = (p_j.z + p_jp1.z + p_jm1.z) / 3.0

                for j in range(travel_point_count):
                    p_j = travel_point[j]
                    n_j = travel_normal[j]
                    n_j_sph = direction.cartesian_to_spherical(n_j)
                    toolpath_point[toolpath_point_count] = p_j
                    toolpath_tool_orientation[toolpath_point_count] = n_j_sph
                    toolpath_travel_type[toolpath_point_count] = current_travel_type
                    toolpath_point_count = toolpath_point_count + 1

    toolpath_point[toolpath_point_count] = current_point
    toolpath_tool_orientation[toolpath_point_count] = current_normal_sph
    toolpath_travel_type[toolpath_point_count] = current_travel_type
    toolpath_point_count = toolpath_point_count + 1
    ret_data[11] = toolpath_point_count

    frame_set_point[i].x = ti.math.nan
    bvh.remove_point_ti_func(i)

    ret_data[12] = iteration_number + 1
    return ret_data


@ti.func
def frame_data_set_non_supporting_compute_neighborhood_config_brute_force_i(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    i: int,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> ti.u32:
    """
    Assume a valid index i and a non masked point[i]
    """

    p_i = point[i]
    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))
    neighborhood_radius_sqr = neighborhood_radius * neighborhood_radius

    any_in_front = 0
    any_behind = 0
    any_right = 0
    any_left = 0

    for j in range(point.shape[0]):
        if i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        p_ij = p_i - p_j
        p_ij_length_sqr = ti.math.dot(p_ij, p_ij)
        if p_ij_length_sqr > neighborhood_radius_sqr:
            continue

        region_number = atom_get_region_number(p_j, a_i, triperiod)

        if region_number == 0:
            any_in_front = 1
        if region_number == 1:
            any_behind = 1
        if region_number == 2:
            any_left = 1
        if region_number == 3:
            any_right = 1

    state = ti.u32(0)
    if any_in_front:
        state = atom_set_any_in_front(state)
    if any_behind:
        state = atom_set_any_behind(state)
    if any_left:
        state = atom_set_any_left(state)
    if any_right:
        state = atom_set_any_right(state)

    return state


@ti.func
def frame_data_set_non_supporting_compute_neighborhood_config_i(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    bvh: ti.template(),
    i: int,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> ti.u32:
    """
    Assume a valid index i and a non masked point[i]
    """

    p_i = point[i]
    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))

    any_in_front = 0
    any_behind = 0
    any_right = 0
    any_left = 0
    any_below = 0

    stack = bvh.points_in_sphere_local_stack(p_i, neighborhood_radius)

    for stack_index in range(STACK_SIZE_MAX):
        j = stack[stack_index]

        if j == -1 or i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        region_number = atom_get_region_number(p_j, a_i, triperiod)

        if region_number == 0:
            any_in_front = 1
        if region_number == 1:
            any_behind = 1
        if region_number == 2:
            any_left = 1
        if region_number == 3:
            any_right = 1
        if region_number == 5:
            any_below = 1

    state = ti.u32(0)
    if any_in_front:
        state = atom_set_any_in_front(state)
    if any_behind:
        state = atom_set_any_behind(state)
    if any_left:
        state = atom_set_any_left(state)
    if any_right:
        state = atom_set_any_right(state)
    if any_below:
        state = atom_set_any_below(state)

    return state


@ti.kernel
def toolpath_planner_atom_i_is_adjacent_to_constraint(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    bvh: ti.template(),
    i: int,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> int:
    """
    Assume a valid index i and a non masked point[i]
    """

    p_i = point[i]
    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))

    stack = bvh.points_in_sphere_local_stack(p_i, neighborhood_radius)
    is_adjacent_and_constrained_index = -1

    for stack_index in range(STACK_SIZE_MAX):
        j = stack[stack_index]

        if j == -1 or i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        region_number = atom_get_region_number(p_j, a_i, triperiod)

        is_j_adjacent = (
            region_number == 0
            or region_number == 1
            or region_number == 2
            or region_number == 3
        )
        is_j_constrained = not atom_is_not_supporting(state[j]) or atom_is_unaccessible(
            state[j]
        )

        if is_j_adjacent and is_j_constrained:
            is_adjacent_and_constrained_index = j

    return is_adjacent_and_constrained_index


@ti.func
def frame_data_set_is_non_supporting_brute_force_i(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    i: int,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> int:
    """
    Assume a valid index i and a non masked point[i]
    """
    p_i = point[i]
    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))
    neighborhood_radius_sqr = neighborhood_radius * neighborhood_radius

    any_supported = False

    for j in range(point.shape[0]):
        if i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        p_ij = p_i - p_j
        p_ij_length_sqr = ti.math.dot(p_ij, p_ij)
        if p_ij_length_sqr > neighborhood_radius_sqr:
            continue

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if region_number == 4:
            any_supported = True

    is_non_supporting = 1
    if any_supported:
        is_non_supporting = 0
    return is_non_supporting


@ti.func
def frame_data_set_is_non_supporting_i(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    bvh: ti.template(),
    i: int,
    triperiod: ti.math.vec3,
    neighborhood_radius: float,
) -> int:
    """
    Assume a valid index i and a non masked point[i]
    """
    p_i = point[i]
    a_i = math.vec6(p_i, ti.math.vec3(normal[i], phi_t[i]))

    any_supported = False

    stack = bvh.points_in_sphere_local_stack(p_i, neighborhood_radius)

    for stack_index in range(STACK_SIZE_MAX):
        j = stack[stack_index]

        if j == -1 or i == j:
            continue

        p_j = point[j]
        if ti.math.isnan(p_j).any():
            continue

        region_number = atom_get_region_number(p_j, a_i, triperiod)
        if region_number == 4:
            any_supported = True

    is_non_supporting = 1
    if any_supported:
        is_non_supporting = 0
    return is_non_supporting


@ti.func
def atom_get_region_number(
    p: ti.math.vec3,
    a: math.vec6,
    triperiod: ti.math.vec3,
):
    a_p = ti.math.vec3(a[0], a[1], a[2])
    a_basis = ti.math.vec3(a[3], a[4], a[5])
    a_tbn = basis3.from_spherical(a_basis)
    a_t = ti.math.vec3(a_tbn[0, 0], a_tbn[1, 0], a_tbn[2, 0])
    a_b = ti.math.vec3(a_tbn[0, 1], a_tbn[1, 1], a_tbn[2, 1])
    a_n = ti.math.vec3(a_tbn[0, 2], a_tbn[1, 2], a_tbn[2, 2])

    p_a_p = p - a_p

    dot_t = ti.math.dot(a_t, p_a_p)
    dot_t_abs = ti.abs(dot_t)
    dot_b = ti.math.dot(a_b, p_a_p)
    dot_b_abs = ti.abs(dot_b)
    dot_n = ti.math.dot(a_n, p_a_p)
    dot_n_abs = ti.abs(dot_n)

    threshold_t_0_s = THRESHOLD_T_0 * triperiod[0]
    threshold_t_1_s = THRESHOLD_T_1 * triperiod[1]
    threshold_t_2_s = THRESHOLD_T_2 * triperiod[2]
    threshold_b_0_s = THRESHOLD_B_0 * triperiod[0]
    threshold_b_1_s = THRESHOLD_B_1 * triperiod[1]
    threshold_b_2_s = THRESHOLD_B_2 * triperiod[2]
    threshold_n_0_s = THRESHOLD_N_0 * triperiod[0]
    threshold_n_1_s = THRESHOLD_N_1 * triperiod[1]
    threshold_n_2_s = THRESHOLD_N_2 * triperiod[2]

    space_number = 6
    if (
        dot_t_abs < threshold_t_0_s
        and dot_b_abs < threshold_t_1_s
        and dot_n_abs < threshold_t_2_s
    ):
        if dot_t >= 0:
            space_number = 0
        else:
            space_number = 1
    elif (
        dot_t_abs < threshold_b_0_s
        and dot_b_abs < threshold_b_1_s
        and dot_n_abs < threshold_b_2_s
    ):
        if dot_b >= 0:
            space_number = 2
        else:
            space_number = 3
    elif (
        dot_t_abs < threshold_n_0_s
        and dot_b_abs < threshold_n_1_s
        and dot_n_abs < threshold_n_2_s
    ):
        if dot_n < 0:
            space_number = 5

    if dot_n_abs >= threshold_t_2_s and dot_n_abs < threshold_n_2_s:
        is_inside_up_cone = cone.is_point_inside(
            p, a_p, a_n, SUPPORTING_REGION_COS_HALF_ANGLE
        )
        if is_inside_up_cone:
            space_number = 4

    return space_number


# States determined by the local neighborhood


@ti.func
def atom_set_any_in_front(state: ti.u32) -> ti.u32:
    state |= 0b00000000000001
    return state


@ti.func
def atom_unset_any_in_front(state: ti.u32) -> ti.u32:
    state &= 0b11111111111110
    return state


@ti.func
def atom_any_in_front(state: ti.u32) -> int:
    mask = 0b00000000000001
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_any_behind(state: ti.u32) -> ti.u32:
    state |= 0b00000000000010
    return state


@ti.func
def atom_unset_any_behind(state: ti.u32) -> ti.u32:
    state &= 0b11111111111101
    return state


@ti.func
def atom_any_behind(state: ti.u32) -> int:
    mask = 0b00000000000010
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_any_left(state: ti.u32) -> ti.u32:
    state |= 0b00000000000100
    return state


@ti.func
def atom_unset_any_left(state: ti.u32) -> ti.u32:
    state &= 0b11111111111011
    return state


@ti.func
def atom_any_left(state: ti.u32) -> int:
    mask = 0b00000000000100
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_any_right(state: ti.u32) -> ti.u32:
    state |= 0b00000000001000
    return state


@ti.func
def atom_unset_any_right(state: ti.u32) -> ti.u32:
    state &= 0b11111111110111
    return state


@ti.func
def atom_any_right(state: ti.u32) -> int:
    mask = 0b00000000001000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_not_supporting(state: ti.u32) -> ti.u32:
    state |= 0b00000000010000
    return state


@ti.func
def atom_unset_is_not_supporting(state: ti.u32) -> ti.u32:
    state &= 0b11111111101111
    return state


@ti.func
def atom_is_not_supporting(state: ti.u32) -> int:
    mask = 0b00000000010000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_any_below(state: ti.u32) -> ti.u32:
    state |= 0b00000000100000
    return state


@ti.func
def atom_unset_any_below(state: ti.u32) -> ti.u32:
    state &= 0b11111111011111
    return state


@ti.func
def atom_any_below(state: ti.u32) -> int:
    mask = 0b00000000100000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_unaccessible(state: ti.u32) -> ti.u32:
    state |= 0b00000001000000
    return state


@ti.func
def atom_unset_is_unaccessible(state: ti.u32) -> ti.u32:
    state &= 0b11111110111111
    return state


@ti.func
def atom_is_unaccessible(state: ti.u32) -> int:
    mask = 0b00000001000000
    masked = mask & state
    return masked == mask


# States determined relative to another frame


@ti.func
def atom_set_is_in_front(state: ti.u32) -> ti.u32:
    state |= 0b00000100000000
    return state


@ti.func
def atom_unset_is_in_front(state: ti.u32) -> ti.u32:
    state &= 0b11111011111111
    return state


@ti.func
def atom_is_in_front(state: ti.u32) -> int:
    mask = 0b00000100000000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_behind(state: ti.u32) -> ti.u32:
    state |= 0b00001000000000
    return state


@ti.func
def atom_unset_is_behind(state: ti.u32) -> ti.u32:
    state &= 0b11110111111111
    return state


@ti.func
def atom_is_behind(state: ti.u32) -> int:
    mask = 0b00001000000000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_below(state: ti.u32) -> ti.u32:
    state |= 0b00010000000000
    return state


@ti.func
def atom_unset_is_below(state: ti.u32) -> ti.u32:
    state &= 0b11101111111111
    return state


@ti.func
def atom_is_below(state: ti.u32) -> int:
    mask = 0b00010000000000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_on_the_left_or_right(state: ti.u32) -> ti.u32:
    state |= 0b00100000000000
    return state


@ti.func
def atom_unset_is_on_the_left_or_right(state: ti.u32) -> ti.u32:
    state &= 0b11011111111111
    return state


@ti.func
def atom_is_on_the_left_or_right(state: ti.u32) -> int:
    mask = 0b00100000000000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_in_cycle(state: ti.u32) -> ti.u32:
    state |= 0b01000000000000
    return state


@ti.func
def atom_unset_is_in_cycle(state: ti.u32) -> ti.u32:
    state &= 0b10111111111111
    return state


@ti.func
def atom_is_in_cycle(state: ti.u32) -> int:
    mask = 0b01000000000000
    masked = mask & state
    return masked == mask


@ti.func
def atom_set_is_wall(state: ti.u32) -> ti.u32:
    state |= 0b10000000000000
    return state


@ti.func
def atom_unset_is_wall(state: ti.u32) -> ti.u32:
    state &= 0b01111111111111
    return state


@ti.func
def atom_is_wall(state: ti.u32) -> int:
    mask = 0b10000000000000
    masked = mask & state
    return masked == mask
