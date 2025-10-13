import taichi as ti
import numpy as np

from . import direction, math, toolpath3

BALL_2DPOS_0 = np.array((-4.07, -12.16))
BALL_2DPOS_1 = np.array((304.93, -12.16))
BALL_2DPOS_2 = np.array((150.43, 296.84))
BALL_Z = -45.7
RAIL_ANGLE_0 = 29.89
RAIL_ANGLE_1 = -29.89
RAIL_ANGLE_2 = 90.0
Z_OFFSET = 75.0
MAX_TILT_ANGLE_DEG = 30.0
MAX_X_AXIS = 300
MAX_Y_AXIS = 293
MAX_Z_AXIS = 280
BALL_TO_CORNER = 10
NOZZLE_TO_GAUNTRY = 70
FILAMENT_DIAMETER = 1.75
DEPOSITON_FEEDRATE = 600
TRAVEL_FEEDRATE = 3000
Z_FAN_ON = 2.0
RETRACT_THRESH = 1.8
RETRACT_LENGTH = 2.0
RETRACT_SPEED = 2700

HEADER = \
f"""G21 ; set units to millimeters
G90 ; use absolute coordinates
M190 S55 ; wait for bed temperature to be reached
G32 ; homing and bed calibration
M104 S210 ; set temperature
M109 S210 ; wait for temperature to be reached
T0
M82 ; use absolute distances for extrusion
; switch to enable 3Z mode
M98 P"/macros/enable3Z.g"
M400 ; wait 
; purging line
G92 E0
G1 Z{1.3 + Z_OFFSET} U{1.3 + Z_OFFSET} V{1.3 + Z_OFFSET} F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z{0.3 + Z_OFFSET} U{0.3 + Z_OFFSET} V{0.3 + Z_OFFSET} F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z{0.3 + Z_OFFSET} U{0.3 + Z_OFFSET} V{0.3 + Z_OFFSET} F1000.0 E15 ; draw 1st line
G1 X0.4 Y200.0 Z{0.3 + Z_OFFSET} U{0.3 + Z_OFFSET} V{0.3 + Z_OFFSET} F1000.0 ; move to side a little
G1 X0.4 Y20 Z{0.3 + Z_OFFSET} U{0.3 + Z_OFFSET} V{0.3 + Z_OFFSET} F1000.0 E30 ; draw 2nd line
G1 E28.0 F2700 ; retract
; done purging extruder
M83 ; relative extrusion
"""
FOOTER = \
f"""M82 ; absolute extrusion
G92 E0
G1 E-2.0 F2700 ; retract
G92 E0
M104 S0 ; turn off temperature
M140 S0
M106 S0    ; fan off
; switch to disable 3Z mode
M98 P"/macros/disable3Z.g"
M400 ; wait 
"""

class Toolpath:
	def _init_(self) -> None:
		# 5D point: x, y, z0, z1, z2
		self.point = None
		self.travel_type = None
		self.width = None
		self.height = None

		self.point_count = None

	def from_cartesian_toolpath(self, cartesian_toolpath: toolpath3.Toolpath):
		point_count = cartesian_toolpath.point_count

		self.point = np.full(dtype=np.float32, shape=(point_count, 5), fill_value=0.0)
		self.travel_type = cartesian_toolpath.travel_type
		self.point_count = point_count

		self.width = cartesian_toolpath.width
		self.height = cartesian_toolpath.height

		return toolpath_from_cartesian_toolpath(
			cartesian_toolpath.point, cartesian_toolpath.tool_orientation, self.point
		) != 0

	def to_cartesian_toolpath(self) -> toolpath3.Toolpath:
		point_count = self.point_count

		toolpath_cartesian = toolpath3.Toolpath()
		toolpath_cartesian.point_count = self.point_count
		toolpath_cartesian.point = np.full(
			dtype=np.float32, shape=(point_count, 3), fill_value=0.0
		)
		toolpath_cartesian.tool_orientation = np.full(
			shape=(point_count, 2), dtype=np.float32, fill_value=0.0
		)
		toolpath_cartesian.travel_type = self.travel_type
		toolpath_cartesian.width = self.width
		toolpath_cartesian.height = self.height
		toolpath_to_cartesian_toolpath(
			self.point, toolpath_cartesian.point, toolpath_cartesian.tool_orientation
		)
		return toolpath_cartesian

	def to_numpy(self):
		dict_array = {}
		dict_array["point"] = self.point
		dict_array["travel_type"] = self.travel_type
		dict_array["width"] = self.width
		dict_array["height"] = self.height
		dict_array["point_count"] = np.array(self.point_count)

		return dict_array

	def from_numpy(self, dict_array):
		self.point = dict_array["point"]
		self.travel_type = dict_array["travel_type"]
		self.width = dict_array["width"]
		self.height = dict_array["height"]
		self.point_count = int(dict_array["point_count"][()])

	def save(self, filename: str):
		dict_array = self.to_numpy()
		np.savez(filename, **dict_array)

	def load(self, filename: str):
		dict_array = np.load(filename)
		self.from_numpy(dict_array)

	def smooth_points(self, iter_count: int):
		point_buffer = np.full(
			shape=(self.point_count, 5), dtype=np.float32, fill_value=0.0
		)
		for _ in range(iter_count):
			toolpath_smooth_points(self.point, self.travel_type, point_buffer)
			self.point, point_buffer = (point_buffer, self.point)


def KinematicTaichi(
	ball_0_position,
	ball_1_position,
	ball_2_position,
	balls_z_position,
	slot_0_angle,
	slot_1_angle,
	slot_2_angle,
	z_offset=0,
):
	# Initialize constraints for the inverse kinematic
	slot_0_normal = ti.math.vec2(
		-np.sin(slot_0_angle / 180 * ti.math.pi),
		np.cos(slot_0_angle / 180 * ti.math.pi),
	)
	slot_1_normal = ti.math.vec2(
		-np.sin(slot_1_angle / 180 * ti.math.pi),
		np.cos(slot_1_angle / 180 * ti.math.pi),
	)
	slot_2_normal = ti.math.vec2(
		-np.sin(slot_2_angle / 180 * ti.math.pi),
		np.cos(slot_2_angle / 180 * ti.math.pi),
	)
	constraint_0 = ti.math.vec4(slot_0_normal[0], slot_0_normal[1], 0, 0)
	constraint_1 = ti.math.vec4(
		slot_1_normal[0],
		slot_1_normal[1],
		0,
		-np.dot(ball_1_position - ball_0_position, slot_1_normal),
	)
	constraint_2 = ti.math.vec4(
		slot_2_normal[0],
		slot_2_normal[1],
		0,
		-np.dot(ball_2_position - ball_0_position, slot_2_normal),
	)
	l1 = ti.math.vec2(ball_1_position - ball_0_position)
	l2 = ti.math.vec2(ball_2_position - ball_0_position)
	l1_rot = ti.math.vec2(-l1[1], l1[0])
	l2_rot = ti.math.vec2(-l2[1], l2[0])
	ball_0_3d_position = ti.math.vec3(
		ball_0_position[0], ball_0_position[1], balls_z_position
	)

	# Initialize additional constraints for the forward kinematic
	slot_0_direction = ti.math.vec2(-slot_0_normal[1], slot_0_normal[0])
	s0d_s1n = np.dot(slot_0_direction, slot_1_normal)
	s0d_s2n = np.dot(slot_0_direction, slot_2_normal)
	Cf = np.dot(
		constraint_2[3] * slot_1_normal - constraint_1[3] * slot_2_normal,
		slot_0_direction,
	)

	@ti.func
	def solve_weierstrass(A, B, C):
		# Solve A*cos(x) + B*sin(x) + C = 0 and return the solution with the smaller absoute value
		return 0 if A + C == 0 else 2*ti.math.atan2((C + A)/(-B - ti.math.sign(B)*ti.math.sqrt(B**2 + (A + C)*(A - C))), 1.0)

	@ti.func
	def rotate(vector, axis, angle):
		return vector*ti.math.cos(angle) + ti.math.cross(axis, vector)*ti.math.sin(angle)+axis*ti.math.dot(axis, vector)*(1-ti.math.cos(angle))

	@ti.func
	def get_angle_to_match_z(vector, axis, target_z):
		C = axis[2]*ti.math.dot(vector, axis)
		A = vector[2]-C
		C -= target_z
		B = ti.math.cross(axis, vector)[2]
		return solve_weierstrass(A, B, C)

	@ti.func
	def inverse(position, normal):
		offset = 0.0
		# Nozzle-bed collisition
		if position.z < 0:
			offset = ti.math.nan

		# Exceeded max angle
		if ti.math.acos(normal.z)/ti.math.pi*180 > MAX_TILT_ANGLE_DEG + 1e-4:
			offset = ti.math.nan

		# Define a tangent space
		normal = ti.math.normalize(ti.math.vec3(-normal[0], -normal[1], normal[2]))
		bitanget = ti.math.normalize(ti.math.cross(normal, ti.math.vec3(1, 0, 0)))
		tangent = ti.math.normalize(ti.math.cross(bitanget, normal))
		TBN = ti.math.mat3((tangent, bitanget, normal))

		# Initialize the 2D lines contraints
		n0 = (TBN @ constraint_0[:3])[:2]
		n1 = (TBN @ constraint_1[:3])[:2]
		k1 = constraint_1[3]
		n2 = (TBN @ constraint_2[:3])[:2]
		k2 = constraint_2[3]

		# Solve for theta and t
		d = ti.math.vec2(-n0[1], n0[0])
		d_n1 = ti.math.dot(d, n1)
		d_n2 = ti.math.dot(d, n2)
		l1_n1 = ti.math.dot(l1, n1)
		l1_rot_n1 = ti.math.dot(l1_rot, n1)
		A = d_n1*ti.math.dot(l2, n2) - d_n2*l1_n1
		B = d_n1*ti.math.dot(l2_rot, n2) - d_n2*l1_rot_n1
		C = ti.math.dot(k2*n1 - k1*n2, d)
		theta = solve_weierstrass(A, B, C)
		ct = ti.math.cos(theta)
		st = ti.math.sin(theta)
		t = -(l1_n1*ct + l1_rot_n1*st + k1)/d_n1

		# Compute the frame rotation
		R = ti.math.mat2(((ct, -st), (st, ct)))
		R_3d = ti.math.mat3(((R[0, 0], R[0, 1], 0), (R[1, 0], R[1, 1], 0), (0, 0, 1)))

		# Compute the ball 0 position and the deltas on Z
		b0_2d = t*d
		b1_2d = b0_2d + R @ l1
		b2_2d = b0_2d + R @ l2
		TBN_T = TBN.transpose()
		b0 = TBN_T @ ti.math.vec3(b0_2d[0], b0_2d[1], 0)
		Z = ti.math.vec3(TBN_T[2, 0], TBN_T[2, 1], TBN_T[2, 2])
		b1_z = ti.math.dot(Z, ti.math.vec3(b1_2d[0], b1_2d[1], 0))
		b2_z = ti.math.dot(Z, ti.math.vec3(b2_2d[0], b2_2d[1], 0))
		delta_z1 = b1_z - b0[2]
		delta_z2 = b2_z - b0[2]

		# Apply the transform and return
		pos = TBN_T @ R_3d @ (position - ball_0_3d_position) + ball_0_3d_position
		x, y, z0, z1, z2 = pos[0] + b0[0], pos[1] + b0[1], pos[2]+z_offset, pos[2]-delta_z1+z_offset, pos[2]-delta_z2+z_offset

		# Endstop collision
		minZ = ti.min(ti.min(z0, z1), z2)
		if minZ < 0 and not ti.math.isnan(offset):
			offset = -minZ

		# Exceeded axes range
		if x < 0 or x > MAX_X_AXIS or y < 0 or y > MAX_Y_AXIS or z0 > MAX_Z_AXIS or z1 > MAX_Z_AXIS or z2 > MAX_Z_AXIS:
			offset = ti.math.nan

		# Bed-gauntry collision
		corner0 = ti.math.vec3(ball_0_position[0]-BALL_TO_CORNER, ball_0_position[1]-BALL_TO_CORNER, 0)
		corner1 = ti.math.vec3(ball_1_position[0]+BALL_TO_CORNER, ball_1_position[1]-BALL_TO_CORNER, 0)
		corner2 = ti.math.vec3(ball_0_position[0]-BALL_TO_CORNER, ball_2_position[1]+BALL_TO_CORNER, 0)
		corner3 = ti.math.vec3(ball_1_position[0]+BALL_TO_CORNER, ball_2_position[1]+BALL_TO_CORNER, 0)
		corner0 = TBN_T @ R_3d @ (corner0 - ball_0_3d_position) + ball_0_3d_position + ti.math.vec3(b0[0], b0[1], -pos[2])
		corner1 = TBN_T @ R_3d @ (corner1 - ball_0_3d_position) + ball_0_3d_position + ti.math.vec3(b0[0], b0[1], -pos[2])
		corner2 = TBN_T @ R_3d @ (corner2 - ball_0_3d_position) + ball_0_3d_position + ti.math.vec3(b0[0], b0[1], -pos[2])
		corner3 = TBN_T @ R_3d @ (corner3 - ball_0_3d_position) + ball_0_3d_position + ti.math.vec3(b0[0], b0[1], -pos[2])

		maxCorner = ti.max(ti.max(corner0[2], corner1[2]), ti.max(corner2[2], corner3[2]))
		if maxCorner > NOZZLE_TO_GAUNTRY and not ti.math.isnan(offset) and offset < maxCorner - NOZZLE_TO_GAUNTRY:
		 	offset = maxCorner - NOZZLE_TO_GAUNTRY

		return x, y, z0, z1, z2, offset
	
	@ti.func
	def forward(x, y, z0, z1, z2):
		# Solve for the rotation to match the Z values
		vector1 = ti.math.vec3(l1[0], l1[1], 0)
		axis1 = ti.math.normalize(ti.math.vec3(-l1[1], l1[0], 0))
		angle1 = get_angle_to_match_z(vector1, axis1, z0-z1)

		vector2 = rotate(ti.math.vec3(l2[0], l2[1], 0), axis1, angle1)
		vector1 = rotate(vector1, axis1, angle1)
		axis2 = ti.math.normalize(vector1)
		angle2 = get_angle_to_match_z(vector2, axis2, z0-z2)

		# Get the projected 11 and l2
		l1p = rotate(vector1, axis2, angle2)[0:2]
		l2p = rotate(vector2, axis2, angle2)[0:2]
		l1p_rot = ti.math.vec2(-l1p[1], l1p[0])
		l2p_rot = ti.math.vec2(-l2p[1], l2p[0])

		# Solve for theta and t
		l1_n1 = ti.math.dot(l1p, slot_1_normal)
		l1_rot_n1 = ti.math.dot(l1p_rot, slot_1_normal)
		A = s0d_s1n*ti.math.dot(l2p, slot_2_normal) - s0d_s2n*l1_n1
		B = s0d_s1n*ti.math.dot(l2p_rot, slot_2_normal) - s0d_s2n*l1_rot_n1
		theta = solve_weierstrass(A, B, Cf)
		t = -(l1_n1*ti.math.cos(theta) + l1_rot_n1*ti.math.sin(theta) + constraint_1[3])/s0d_s1n
		b0 = t*slot_0_direction

		# Get the normal
		up = ti.math.vec3(0, 0, 1)
		normal = rotate(rotate(rotate(up, axis1, angle1), axis2, angle2), up, theta)
		normal = ti.math.vec3(-normal[0], -normal[1], normal[2])

		# Get the position
		pos = ti.math.vec3(x-b0[0], y-b0[1], z0-z_offset) - ball_0_3d_position
		position = rotate(rotate(rotate(pos, up, -theta), axis2, -angle2), axis1, -angle1) + ball_0_3d_position

		return position, normal

	# Return the kinematics
	return inverse, forward


inverse, forward = KinematicTaichi(
	BALL_2DPOS_0,
	BALL_2DPOS_1,
	BALL_2DPOS_2,
	BALL_Z,
	RAIL_ANGLE_0,
	RAIL_ANGLE_1,
	RAIL_ANGLE_2,
	Z_OFFSET,
)

@ti.kernel
def get_vertical_offset_kernel(point_in: ti.types.ndarray(), normal_in: ti.types.ndarray(), x_shift: ti.f32, y_shift: ti.f32, z_shift: ti.f32) -> ti.f32:
	max_offset: ti.f32 = 0
	error: ti.i32 = 0

	for i in range(point_in.shape[0]):
		point_i = ti.math.vec3(point_in[i, 0] + x_shift, point_in[i, 1] + y_shift, point_in[i, 2] + z_shift)
		normal_sph_i = ti.math.vec2(normal_in[i, 0], normal_in[i, 1])
		normal_i = direction.spherical_to_cartesian(normal_sph_i)
		_, _, _, _, _, offset = inverse(point_i, normal_i)
		if ti.math.isnan(offset):
			ti.atomic_max(error, 1)
		ti.atomic_max(max_offset, offset)
	return ti.math.nan if error else max_offset

def get_plaftorm_size(toolpath, nozzle_width, layer_height):
	toolpath_min, toolpath_max = toolpath.get_aabb()
	toolpath_offset = 0.5*(np.array((MAX_X_AXIS, MAX_Y_AXIS, 0)) + toolpath_min-toolpath_max) - toolpath_min
	z_shift = 0.0
	
	while True:	
		offset = get_vertical_offset_kernel(toolpath.point, toolpath.tool_orientation, toolpath_offset[0], toolpath_offset[1], z_shift)
		if np.isnan(offset):
			print("Fatal Error: No platform could solve the collision.")
			assert(False) # Non printable model
		elif offset > 0:
			z_shift += offset
		else:
			break

	return np.ceil(toolpath_max[0]/nozzle_width)*nozzle_width, np.ceil(toolpath_max[1]/nozzle_width)*nozzle_width, np.ceil(z_shift/layer_height)*layer_height

@ti.kernel
def toolpath_from_cartesian_toolpath(
	point_in: ti.types.ndarray(),
	normal_in: ti.types.ndarray(),
	point_machine: ti.types.ndarray(),
) -> ti.i32:
	valid: ti.i32 = 1
	for i in range(point_in.shape[0]):

		point_i = ti.math.vec3(point_in[i, 0], point_in[i, 1], point_in[i, 2])
		normal_sph_i = ti.math.vec2(normal_in[i, 0], normal_in[i, 1])
		normal_i = direction.spherical_to_cartesian(normal_sph_i)

		x, y, z0, z1, z2, collision = inverse(point_i, normal_i)

		if ti.math.isnan(collision) or collision != 0:
			ti.atomic_min(valid, 0)
		
		point_machine[i, 0] = x
		point_machine[i, 1] = y
		point_machine[i, 2] = z0
		point_machine[i, 3] = z1
		point_machine[i, 4] = z2
	return valid


@ti.kernel
def toolpath_to_cartesian_toolpath(
	point_machine: ti.types.ndarray(),
	point_out: ti.types.ndarray(),
	normal_out: ti.types.ndarray(),
):
	for i in range(point_machine.shape[0]):
		point_i, normal_i = forward(
			point_machine[i, 0],
			point_machine[i, 1],
			point_machine[i, 2],
			point_machine[i, 3],
			point_machine[i, 4],
		)
		normal_sph_i = direction.cartesian_to_spherical(normal_i)

		for j in ti.static(range(3)):
			point_out[i, j] = point_i[j]
		for j in ti.static(range(2)):
			normal_out[i, j] = normal_sph_i[j]


@ti.kernel
def toolpath_smooth_points(
	point_in: ti.types.ndarray(),
	travel_type: ti.types.ndarray(),
	point_out: ti.types.ndarray(),
):
	for i in range(point_in.shape[0]):
		im1 = ti.max(i - 1, 0)
		ip1 = ti.min(i + 1, point_in.shape[0] - 1)

		smooth_point_i = (
			i != point_in.shape[0] - 1
			and i != 0
			and travel_type[im1] == toolpath3.TRAVEL_TYPE_DEPOSITION
			and travel_type[i] == toolpath3.TRAVEL_TYPE_DEPOSITION
			and travel_type[ip1] == toolpath3.TRAVEL_TYPE_DEPOSITION
		)
		if not smooth_point_i:
			for j in ti.static(range(5)):
				point_out[i, j] = point_in[i, j]
			continue

		point_im1 = math.vec5(
			point_in[im1, 0],
			point_in[im1, 1],
			point_in[im1, 2],
			point_in[im1, 3],
			point_in[im1, 4],
		)
		point_i = math.vec5(
			point_in[i, 0],
			point_in[i, 1],
			point_in[i, 2],
			point_in[i, 3],
			point_in[i, 4],
		)
		point_ip1 = math.vec5(
			point_in[ip1, 0],
			point_in[ip1, 1],
			point_in[ip1, 2],
			point_in[ip1, 3],
			point_in[ip1, 4],
		)

		average_point = (point_im1 + point_i + point_ip1) / 3.0

		for j in ti.static(range(5)):
			point_out[i, j] = average_point[j]
