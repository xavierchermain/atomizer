import numpy as np
import argparse
from tqdm import tqdm

def Kinematic(ball_0_position, ball_1_position, ball_2_position, balls_z_position, slot_0_angle, slot_1_angle, slot_2_angle, z_offset = 0):
	# Initialize constraints for the inverse kinematic
	slot_0_normal = np.array((-np.sin(np.radians(slot_0_angle)), np.cos(np.radians(slot_0_angle))))
	slot_1_normal = np.array((-np.sin(np.radians(slot_1_angle)), np.cos(np.radians(slot_1_angle))))
	slot_2_normal = np.array((-np.sin(np.radians(slot_2_angle)), np.cos(np.radians(slot_2_angle))))
	constraint_0 = np.array((*slot_0_normal, 0, 0))
	constraint_1 = np.array((*slot_1_normal, 0, -np.dot(ball_1_position-ball_0_position, slot_1_normal)))
	constraint_2 = np.array((*slot_2_normal, 0, -np.dot(ball_2_position-ball_0_position, slot_2_normal)))
	l1 = ball_1_position - ball_0_position
	l2 = ball_2_position - ball_0_position
	l1_rot = np.array((-l1[1], l1[0]))
	l2_rot = np.array((-l2[1], l2[0]))
	ball_0_3d_position = np.array((ball_0_position[0], ball_0_position[1], balls_z_position))

	# Initialize additional constraints for the forward kinematic
	slot_0_direction = np.array((-slot_0_normal[1], slot_0_normal[0]))
	s0d_s1n = np.dot(slot_0_direction, slot_1_normal)
	s0d_s2n = np.dot(slot_0_direction, slot_2_normal)
	Cf = np.dot(constraint_2[-1]*slot_1_normal-constraint_1[-1]*slot_2_normal, slot_0_direction)
	
	def normalize(v):
		return v/np.linalg.norm(v)
	
	def solve_weierstrass(A, B, C):
		# Solve A*cos(x) + B*sin(x) + C = 0 and return the solution with the smaller absoute value
		return 0 if A + C == 0 else 2*np.arctan((C + A)/(-B - np.sign(B)*np.sqrt(B**2 + (A + C)*(A - C))))
	
	def rotate(vector, axis, angle):
		return vector*np.cos(angle) + np.cross(axis, vector)*np.sin(angle)+axis*np.dot(axis, vector)*(1-np.cos(angle))

	def get_angle_to_match_z(vector, axis, target_z):
		C = axis[2]*np.dot(vector, axis)
		A = vector[2]-C
		C -= target_z
		B = np.cross(axis, vector)[2]
		return solve_weierstrass(A, B, C)
	
	def inverse(position, normal):
		# Define a tangent space
		normal = normalize(np.array((-normal[0], -normal[1], normal[2])))
		bitanget = normalize(np.cross(normal, np.array((1, 0, 0))))
		tangent = normalize(np.cross(bitanget, normal))
		TBN = np.array((tangent, bitanget, normal))

		# Initialize the 2D lines contraints
		n0 = (TBN @ constraint_0[:-1])[:-1]
		n1 = (TBN @ constraint_1[:-1])[:-1]
		k1 = constraint_1[-1]
		n2 = (TBN @ constraint_2[:-1])[:-1]
		k2 = constraint_2[-1]

		# Solve for theta and t
		d = np.array((-n0[1], n0[0]))
		d_n1 = np.dot(d, n1)
		d_n2 = np.dot(d, n2)
		l1_n1 = np.dot(l1, n1)
		l1_rot_n1 = np.dot(l1_rot, n1)
		A = d_n1*np.dot(l2, n2) - d_n2*l1_n1
		B = d_n1*np.dot(l2_rot, n2) - d_n2*l1_rot_n1
		C = np.dot(k2*n1 - k1*n2, d)
		theta = solve_weierstrass(A, B, C)
		ct = np.cos(theta)
		st = np.sin(theta)
		t = -(l1_n1*ct + l1_rot_n1*st + k1)/d_n1

		# Compute the frame rotation
		R = np.array(((ct, -st), (st, ct)))
		R_3d = np.array(((R[0][0], R[0][1], 0), (R[1][0], R[1][1], 0), (0, 0, 1)))

		# Compute the ball 0 position and the deltas on Z
		b0_2d = t*d
		b1_2d = b0_2d + R @ l1
		b2_2d = b0_2d + R @ l2
		TBN_T = TBN.T
		b0 = TBN_T @ np.array((*b0_2d, 0))
		Z = TBN_T[2]
		b1_z = np.dot(Z, np.array((*b1_2d, 0)))
		b2_z = np.dot(Z, np.array((*b2_2d, 0)))
		delta_z1 = b1_z - b0[2]
		delta_z2 = b2_z - b0[2]

		# Apply the transform and return
		pos = TBN_T @ R_3d @ (position - ball_0_3d_position) + ball_0_3d_position
		return pos[0] + b0[0], pos[1] + b0[1], pos[2]+z_offset, pos[2]-delta_z1+z_offset, pos[2]-delta_z2+z_offset

	def forward(x, y, z0, z1, z2):
		# Solve for the rotation to match the Z values
		vector1 = np.array((*l1, 0))
		axis1 = normalize(np.array((-l1[1], l1[0], 0)))
		angle1 = get_angle_to_match_z(vector1, axis1, z0-z1)

		vector2 = rotate(np.array((*l2, 0)), axis1, angle1)
		vector1 = rotate(vector1, axis1, angle1)
		axis2 = normalize(vector1)
		angle2 = get_angle_to_match_z(vector2, axis2, z0-z2)

		# Get the projected 11 and l2
		l1p = rotate(vector1, axis2, angle2)[0:2]
		l2p = rotate(vector2, axis2, angle2)[0:2]
		l1p_rot = np.array((-l1p[1], l1p[0]))
		l2p_rot = np.array((-l2p[1], l2p[0]))

		# Solve for theta and t
		l1_n1 = np.dot(l1p, slot_1_normal)
		l1_rot_n1 = np.dot(l1p_rot, slot_1_normal)
		A = s0d_s1n*np.dot(l2p, slot_2_normal) - s0d_s2n*l1_n1
		B = s0d_s1n*np.dot(l2p_rot, slot_2_normal) - s0d_s2n*l1_rot_n1
		theta = solve_weierstrass(A, B, Cf)
		t = -(l1_n1*np.cos(theta) + l1_rot_n1*np.sin(theta) + constraint_1[-1])/s0d_s1n
		b0 = t*slot_0_direction

		# Get the normal
		up = np.array((0, 0, 1))
		normal = rotate(rotate(rotate(up, axis1, angle1), axis2, angle2), up, theta)
		normal = np.array((-normal[0], -normal[1], normal[2]))

		# Get the position
		pos = np.array((x-b0[0], y-b0[1], z0-z_offset)) - ball_0_3d_position
		position = rotate(rotate(rotate(pos, up, -theta), axis2, -angle2), axis1, -angle1) + ball_0_3d_position

		return position, normal

	# Return a kinematic object
	return type('Kinematic', (), {'inverse': lambda *x: inverse(*x[1:]), 'forward': lambda *x: forward(*x[1:])})()

def translate_gcode(input_path, output_path, disable_normal, kinematic):
	def parse(line, symbol, prev):
			sIndex = line.find(symbol)
			eIndex = line[sIndex:].find(' ')
			if eIndex == -1:
				eIndex = line[sIndex:].find('\n')
			if line[sIndex+1 : sIndex+eIndex] == '':
				sIndex = -1
			return float(line[sIndex+1 : sIndex+eIndex]) if sIndex >= 0 else prev
	
	with open(input_path, 'r') as input_file:
		line_count = len(input_file.readlines())

	with open(input_path, 'r') as input_file:
		with open(output_path, 'w') as output_file:
			x, y, z, u, v, e, f, position = 0, 0, 0, 0, 0, 0, 0, np.array((0, 0, 0))
			x_prev, y_prev, z_prev, u_prev, v_prev, position_prev = x, y, z, u, v, position
			for line in tqdm(input_file, total=line_count):		
				if line[0:3] == 'G1 ' and line[3] != 'E':
					x = parse(line, 'X', x)
					y = parse(line, 'Y', y)
					z = parse(line, 'Z', z)
					u = parse(line, 'U', u)
					v = parse(line, 'V', v)
					e = parse(line, 'E', e)
					f = parse(line, 'F', f)

					position, normal = kinematic.forward(x, y, z, u, v)
					gcode_distance = np.linalg.norm(np.array((x, y, z, u, v))-np.array((x_prev, y_prev, z_prev, u_prev, v_prev)))
					real_distance = np.linalg.norm(position-position_prev)

					z_dist = max(max(np.abs(z-z_prev), np.abs(u-u_prev)), np.abs(v-v_prev))
					z_speed = f * z_dist / gcode_distance if gcode_distance > 0 else 1
					rateo = 1900/z_speed if z_speed > 1900 else 1

					if gcode_distance == 0:
						real_distance, gcode_distance = 1, 1
					output_file.write(f'G1 X{position[0]:.5f} Y{position[1]:.5f} Z{position[2]:.5f} E{e:.5f} F{f*rateo*real_distance/gcode_distance:.5f}; ({normal[0]:.5f}, {normal[1]:.5f}, {normal[2]:.5f})\n')
					
					if not disable_normal:
						output_file.write(f'G1 X{position[0]+normal[0]:.5f} Y{position[1]+normal[1]:.5f} Z{position[2]+normal[2]:.5f} F100000\n')
						output_file.write(f'G1 X{position[0]:.5f} Y{position[1]:.5f} Z{position[2]:.5f} F100000\n')

					x_prev, y_prev, z_prev, u_prev, v_prev, position_prev = x, y, z, u, v, position
				else:
					output_file.write(line)

def get_kinematic():
	return Kinematic(np.array((-4.07, -12.16)), np.array((304.93, -12.16)), np.array((150.43, 296.84)), -45.7, 29.89, -29.89, 90, 75)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert RatRig G-code to a format that can be visualized using Craftware Legacy. The normals at each point are shown as travels.")
	parser.add_argument("input_path", help="The path to the input RatRig G-code.")
	parser.add_argument("output_path", nargs="?", default="-1", help="The path to the output G-code for visualization in Craftware.")
	parser.add_argument("disable_normal", nargs="?", default="False", help="If True, do not output the normal as travels at each point.")
	args = parser.parse_args()
	input_path = args.input_path
	output_path = args.output_path if args.output_path != "-1" else input_path[:input_path.index('.gcode')]+'_craftware.gcode'
	disable_normal = args.disable_normal != "False"

	kinematic = get_kinematic()
	translate_gcode(input_path, output_path, disable_normal, kinematic)    
