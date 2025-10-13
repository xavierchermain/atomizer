from math import pi

import taichi as ti

from . import (
    basis3,
    direction,
    grid3,
    line,
    math,
    phasor3,
    random,
    solid3,
    toolpath3,
    transform3,
    triphasor3,
)

MACHINE_MAX_SLOPE_ANGLE = 7.0 * ti.math.pi / 180.0
MAX_SLOPE_ANGLE = min(MACHINE_MAX_SLOPE_ANGLE, (pi - toolpath3.NOZZLE_CONE_ANGLE) * 0.5)
CEIL_MAX_ANGLE = MAX_SLOPE_ANGLE
# FLOOR_MAX_ANGLE = MAX_SLOPE_ANGLE
FLOOR_MAX_ANGLE = 1.0 * ti.math.pi / 180.0
WALL_MAX_ANGLE = 45.0 * ti.math.pi / 180.0

LAYER_HEIGHT_WRT_NOZZLE = 0.5
SPACE_SAMPLING_WRT_LAYER_HEIGHT = 0.5
CELL_SIDES_LENGTH_WRT_SPACE_SAMPLING = 0.75
# CELL_SIDES_LENGTH_WRT_SPACE_SAMPLING = 1. / ti.sqrt(3.0)

CURVATURE_THRESHOLD = 0.1

ALIGNMENT_ITERATION_COUNT = 64
# Probability to ignore a neighbor
P_IGNORE_NEIGHBOR_START = 0.5
STOP_IGNORING_NEIGHBOR = ALIGNMENT_ITERATION_COUNT // 2
SEED = 1
SPATIAL_WEIGHT_EXPONENT = 1
SPATIAL_FILTER_RADIUS = 2.0


def generate_supports(
    sdf: solid3.SDF,
    input_sdf,
    flag_field,
    offset: ti.math.vec3,
    period: float,
):
    flag_supports(sdf.sdf, sdf.grid.cell_3dcount, flag_field)
    process_flagged_support(
        input_sdf,
        sdf.sdf,
        flag_field,
        sdf.grid.cell_sides_length,
        offset,
        period,
    )


@ti.kernel
def init_spherical_direction_field_from_sdf(
    sdf: ti.template(),
    cell_sides_length: float,
    all_up: int,
    spherical_direction: ti.template(),
    state: ti.template(),
):
    """
    Notes
    -----
    Chermain et al. (2025), Section 4.1, paragraph Tool orientation constraints
    """
    origin = ti.math.vec3(0.0, 0.0, 0.0)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)

    for i in ti.grouped(ti.ndrange(*state.shape)):
        cell_center_i = grid3.cell_center_point(i, origin, cell_sides_length)

        is_boundary_region_i = is_boundary_region(sdf[i], layer_height)
        is_outside_i = is_after_boundary(sdf[i])

        if is_boundary_region_i:
            n_i = solid3.sdf_compute_closest_normal_central(sdf, i, cell_sides_length)
            is_pointing_down = n_i.z < 0.0
            n_i_pointing_up = n_i
            if is_pointing_down:
                n_i_pointing_up = -n_i

            n_i_pointing_up_sph = direction.cartesian_to_spherical(n_i_pointing_up)
            closest_normal_angle = n_i_pointing_up_sph[0]
            is_ceiling = closest_normal_angle < CEIL_MAX_ANGLE and not is_pointing_down
            # is_floor = closest_normal_angle < FLOOR_MAX_ANGLE and is_pointing_down

            curvature = solid3.sdf_compute_closest_curvature_central(
                sdf, i, cell_sides_length
            )
            if cell_center_i.z < layer_height or all_up:
                spherical_direction[i] = ti.math.vec2(0.0, 0.0)
                state[i] = direction.constrain(state[i])
            elif curvature < CURVATURE_THRESHOLD:
                # if is_ceiling or is_floor:
                if is_ceiling:
                    spherical_direction[i] = n_i_pointing_up_sph
                    state[i] = direction.constrain(state[i])
        elif is_outside_i:
            spherical_direction[i] = ti.math.vec2(ti.math.nan)


@ti.kernel
def orthogonolize_direction_wall_region(
    sdf: ti.template(),
    cell_sides_length: float,
    spherical_direction: ti.template(),
    state: ti.template(),
):
    """
    Notes
    -----
    Chermain et al. (2025), Section 6, paragraph Strata orthogonal to walls
    """
    for i in ti.grouped(ti.ndrange(*state.shape)):
        sd_to_closest_point = sdf[i]

        deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
        is_boundary_region_i = is_boundary_region(sd_to_closest_point, deposition_width)

        if is_boundary_region_i:
            n_i = solid3.sdf_compute_closest_normal_central(sdf, i, cell_sides_length)
            bn_is_pointing_down = n_i.z < 0.0
            n_i_pointing_up = n_i
            if bn_is_pointing_down:
                n_i_pointing_up = -n_i

            boundary_normal_i_sph = direction.cartesian_to_spherical(n_i_pointing_up)
            normal_polar_angle = boundary_normal_i_sph[0]

            is_equator = ti.abs((normal_polar_angle - ti.math.pi * 0.5)) < ti.min(
                WALL_MAX_ANGLE, MAX_SLOPE_ANGLE
            )

            is_wall_region_i = (
                is_boundary_region(sdf[i], deposition_width) and is_equator
            )

            curvature = solid3.sdf_compute_closest_curvature_central(
                sdf, i, cell_sides_length
            )
            if curvature < CURVATURE_THRESHOLD and is_wall_region_i:
                up_tilted = direction.orthogonolize(
                    direction.spherical_to_cartesian(spherical_direction[i]), n_i
                )
                up_tilted_sph = direction.cartesian_to_spherical(up_tilted)
                up_tilted_normal_angle = up_tilted_sph[0]
                if up_tilted_normal_angle < MAX_SLOPE_ANGLE:
                    spherical_direction[i] = up_tilted_sph
                    state[i] = direction.constrain(state[i])


@ti.kernel
def spherical_field_constrain_fisrt_layer_up(
    spherical_direction: ti.template(),
    cell_sides_length: float,
):
    origin = ti.math.vec3(0.0)
    for i in ti.grouped(ti.ndrange(*spherical_direction.shape)):

        layer_height = layer_height_from_cell_sides_length(cell_sides_length)
        p_i = grid3.cell_center_point(i, origin, cell_sides_length)

        # If masked
        if ti.math.isnan(spherical_direction[i][0]):
            continue

        if p_i.z <= layer_height:
            spherical_direction[i] = ti.math.vec2(0.0, 0.0)


@ti.kernel
def init_deposition_tangent_field(
    sdf: ti.template(),
    normal: ti.template(),
    top_line: ti.template(),
    bottom_line: ti.template(),
    cell_sides_length: float,
    phi_t: ti.template(),
    state: ti.template(),
):
    """
    Initialization of the tangents of the toolpaths on the boundary

    Notes
    -----
    Chermain et al. (2025), Section 4.2, paragraph Deposition tangent computation
    """
    no_top_line = False
    if ti.math.isnan(top_line[0, 0]):
        no_top_line = True

    no_bottom_line = False
    if ti.math.isnan(bottom_line[0, 0]):
        no_bottom_line = True

    grid_shape = state.shape
    grid_size_xy = ti.math.ivec2(grid_shape[0], grid_shape[1]) * cell_sides_length
    origin = ti.math.vec3(0.0, 0.0, 0.0)
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)

    for i in ti.grouped(ti.ndrange(*grid_shape)):
        # If outside
        if ti.math.isnan(normal[i][0]):
            continue

        curvature = solid3.sdf_compute_closest_curvature_central(
            sdf, i, cell_sides_length
        )
        if curvature > CURVATURE_THRESHOLD:
            continue

        cell_center_i = grid3.cell_center_point(i, origin, cell_sides_length)
        cell_center_xy_normalized = cell_center_i.xy / grid_size_xy
        field_normal_i = direction.spherical_to_cartesian(normal[i])
        boundary_normal_i = solid3.sdf_compute_closest_normal_central(
            sdf, i, cell_sides_length
        )

        bn_is_pointing_down = boundary_normal_i.z < 0.0
        if bn_is_pointing_down:
            boundary_normal_i = -boundary_normal_i

        boundary_normal_i_sph = direction.cartesian_to_spherical(boundary_normal_i)
        normal_polar_angle = boundary_normal_i_sph[0]

        is_north_pole = normal_polar_angle < CEIL_MAX_ANGLE and not bn_is_pointing_down
        is_south_pole = normal_polar_angle < FLOOR_MAX_ANGLE and bn_is_pointing_down
        is_equator = ti.abs((normal_polar_angle - ti.math.pi * 0.5)) < WALL_MAX_ANGLE

        is_wall_region_i = is_boundary_region(sdf[i], deposition_width) and is_equator
        is_top_region_i = (
            is_boundary_region(sdf[i], layer_height * 2.0) and is_north_pole
        )
        is_bottom_region_i = (
            is_boundary_region(sdf[i], layer_height * 2.0) and is_south_pole
        )

        if is_wall_region_i:
            state[i] = tangent_boundary_constrain(state[i])

            # Thank you Cédric
            tangent_i = ti.math.cross(field_normal_i, boundary_normal_i)

            t_from_n = basis3.tangent_from_normal(field_normal_i)
            b_from_n = ti.math.cross(field_normal_i, t_from_n)
            world_to_tangent = transform3.compute_frame_to_canonical_matrix(
                t_from_n, b_from_n, field_normal_i, ti.math.vec3(0.0, 0.0, 0.0)
            ).transpose()
            tangent_i_tangent_space = transform3.apply_to_point(
                world_to_tangent, tangent_i
            )
            phi_t[i] = ti.math.atan2(
                tangent_i_tangent_space.y, tangent_i_tangent_space.x
            )

        if is_top_region_i and not no_top_line:
            state[i] = tangent_boundary_constrain(state[i])
            phi_t[i] = line.line_field2_eval_nearest(
                cell_center_xy_normalized, top_line
            )

        if is_bottom_region_i and not no_bottom_line:
            state[i] = tangent_boundary_constrain(state[i])
            phi_t[i] = line.line_field2_eval_nearest(
                cell_center_xy_normalized, bottom_line
            )


@ti.kernel
def init_phasor3_field_from_sdf(
    sdf: ti.template(),
    cell_sides_length: float,
    all_up: int,
    phase: ti.template(),
    state: ti.template(),
):
    """
    Initialization of the positions of the strata on the boundary

    Notes
    -----
    Chermain et al. (2025), Section 4.3, paragraph Atom positions constraints
    """
    ceil_max_angle = CEIL_MAX_ANGLE
    floor_max_angle = FLOOR_MAX_ANGLE
    if all_up:
        ceil_max_angle = 1.0 * ti.math.pi / 180.0
        floor_max_angle = 1.0 * ti.math.pi / 180.0

    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    cell_diagonal_length = grid3.cell_diagonal_length(cell_sides_length)
    boundary_width = cell_diagonal_length
    for i in ti.grouped(ti.ndrange(*sdf.shape)):
        is_boundary_area_i = is_boundary_region(sdf[i], boundary_width)

        if is_boundary_area_i:
            n_i = solid3.sdf_compute_closest_normal_central(sdf, i, cell_sides_length)
            is_pointing_down = n_i.z < 0.0
            if is_pointing_down:
                n_i = -n_i

            n_i_sph = direction.cartesian_to_spherical(n_i)
            closest_normal_angle = n_i_sph[0]
            is_ceil = closest_normal_angle < ceil_max_angle and not is_pointing_down
            is_floor = closest_normal_angle < floor_max_angle and is_pointing_down

            if is_ceil or is_floor:
                curvature = solid3.sdf_compute_closest_curvature_central(
                    sdf, i, cell_sides_length
                )
                if curvature < CURVATURE_THRESHOLD:
                    phase[i] = distance_to_phase_1extraction_per_cos_along_normal(
                        sdf[i], layer_height
                    )
                    if is_pointing_down:
                        phase[i] = -phase[i]
                    state[i] = phasor3.constrain_phase(state[i])


@ti.kernel
def init_triphasor3_field(
    sdf: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    cell_sides_length: float,
    phase_b: ti.template(),
    state: ti.template(),
):
    """
    Initialization of the position of the toolpaths on the boundary

    Notes
    -----
    Chermain et al. (2025), Section 4.3, paragraph Atom positions constraints
    """
    grid_shape = state.shape

    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)

    for i in ti.grouped(ti.ndrange(*grid_shape)):
        # If outside
        if ti.math.isnan(normal[i][0]):
            continue

        curvature = solid3.sdf_compute_closest_curvature_central(
            sdf, i, cell_sides_length
        )
        high_curvature_i = False
        if curvature >= CURVATURE_THRESHOLD:
            high_curvature_i = True

        boundary_normal_i = solid3.sdf_compute_closest_normal_central(
            sdf, i, cell_sides_length
        )
        bn_is_pointing_down = boundary_normal_i.z < 0.0
        boundary_normal_i_pointing_up = boundary_normal_i
        if bn_is_pointing_down:
            boundary_normal_i_pointing_up = -boundary_normal_i

        boundary_normal_i_pointing_up_sph = direction.cartesian_to_spherical(
            boundary_normal_i_pointing_up
        )
        normal_polar_angle = boundary_normal_i_pointing_up_sph[0]

        is_equator = ti.abs((normal_polar_angle - ti.math.pi * 0.5)) < WALL_MAX_ANGLE

        boundary_width = grid3.cell_diagonal_length(cell_sides_length)
        is_boundary_region_i = is_boundary_region(sdf[i], boundary_width)

        if is_equator and is_boundary_region_i and not high_curvature_i:
            state[i] = triphasor3.constrain_phase_b(state[i])
            phase_b[i] = distance_to_phase_bitangent_1ex(sdf[i], deposition_width)

            tbn = basis3.from_spherical(ti.math.vec3(normal[i], phi_t[i]))
            bitangent = ti.math.vec3(tbn[0, 1], tbn[1, 1], tbn[2, 1])
            if ti.math.dot(boundary_normal_i, bitangent) < 0:
                phase_b[i] = -phase_b[i]


@ti.kernel
def init_frame_set_wall_state(
    sdf: ti.template(),
    frame_point: ti.template(),
    cell_sides_length: float,
    frame_state: ti.template(),
):
    origin = ti.math.vec3(0.0, 0.0, 0.0)
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)

    for frame_index in frame_state:
        point_i = frame_point[frame_index]
        sdf_cell_3dindex = grid3.cell_3dindex_from_point(
            point_i, origin, cell_sides_length
        )
        boundary_normal_i = solid3.sdf_compute_closest_normal_central(
            sdf, sdf_cell_3dindex, cell_sides_length
        )

        bn_is_pointing_down = boundary_normal_i.z < 0.0
        if bn_is_pointing_down:
            boundary_normal_i = -boundary_normal_i

        boundary_normal_i_sph = direction.cartesian_to_spherical(boundary_normal_i)
        normal_polar_angle = boundary_normal_i_sph[0]

        is_equator = ti.abs((normal_polar_angle - ti.math.pi * 0.5)) < WALL_MAX_ANGLE

        is_wall_region_i = (
            is_boundary_region(sdf[sdf_cell_3dindex], deposition_width) and is_equator
        )

        if is_wall_region_i:
            frame_state[frame_index] = toolpath3.atom_set_is_wall(
                frame_state[frame_index]
            )


@ti.kernel
def basis3_field_restrict(
    normal: ti.template(),
    phi_t: ti.template(),
    state: ti.template(),
    normal_restricted: ti.template(),
    phi_t_restricted: ti.template(),
    state_restricted: ti.template(),
):
    for i in ti.grouped(normal_restricted):
        j_block_origin = i * 2
        n_i = ti.math.vec3(0.0)
        constraint_count = 0
        masked_count = 0

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal.shape)
            if is_invalid_j:
                masked_count = masked_count + 1
                continue

            if ti.math.isnan(normal[j][0]):
                masked_count = masked_count + 1
                continue

            if tangent_boundary_is_constrained(state[j]):
                constraint_count = constraint_count + 1

            n_j = direction.spherical_to_cartesian(normal[j])
            n_i = n_i + n_j

        if masked_count == 8:
            normal_restricted[i] = ti.math.vec2(ti.math.nan)
            continue

        n_i = math.normalize_safe(n_i)
        normal_restricted[i] = direction.cartesian_to_spherical(n_i)

        if constraint_count == 0:
            continue

        state_restricted[i] = tangent_boundary_constrain(state_restricted[i])

        t_from_n_i = basis3.tangent_from_normal(n_i)
        b_from_n_i = ti.math.cross(n_i, t_from_n_i)
        world_to_tangent_i = transform3.compute_frame_to_canonical_matrix(
            t_from_n_i, b_from_n_i, n_i, ti.math.vec3(0.0, 0.0, 0.0)
        ).transpose()

        t_i_cov_00 = 0.0
        t_i_cov_01 = 0.0
        t_i_cov_11 = 0.0
        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal.shape)
            if is_invalid_j:
                continue

            if ti.math.isnan(normal[j][0]):
                continue

            if not tangent_boundary_is_constrained(state[j]):
                continue

            # ts_j: j's tangent_space
            t_j_ts_j = ti.math.vec3(direction.polar_to_cartesian(phi_t[j]), 0.0)

            n_j = direction.spherical_to_cartesian(normal[j])
            t_from_n_j = basis3.tangent_from_normal(n_j)
            b_from_n_j = ti.math.cross(n_j, t_from_n_j)
            tangent_j_to_world = transform3.compute_frame_to_canonical_matrix(
                t_from_n_j, b_from_n_j, n_j, ti.math.vec3(0.0, 0.0, 0.0)
            )

            t_j_ts_i = transform3.apply_to_vector(
                world_to_tangent_i @ tangent_j_to_world, t_j_ts_j
            )

            t_i_cov_00 += t_j_ts_i[0] * t_j_ts_i[0]
            t_i_cov_01 += t_j_ts_i[0] * t_j_ts_i[1]
            t_i_cov_11 += t_j_ts_i[1] * t_j_ts_i[1]

        t_i_cov = ti.math.mat2(
            [
                [t_i_cov_00, t_i_cov_01],
                [t_i_cov_01, t_i_cov_11],
            ]
        )

        t_i = math.eigenvec2_with_highest_eigenvalue_iterative(t_i_cov)
        phi_t_i = ti.math.atan2(t_i.y, t_i.x)

        phi_t_restricted[i] = phi_t_i


@ti.kernel
def basis3_field_align(
    normal: ti.template(),
    phi_t_in: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    iteration_number: int,
    phi_t_out: ti.template(),
):
    """
    Propagation of the tangents inside the solid

    Notes
    -----
    Chermain et al. (2025), Section 4.2, paragraph Deposition tangent propagation
    """
    origin = ti.math.vec3(0.0)

    for i in ti.grouped(normal):
        # If masked
        if ti.math.isnan(normal[i][0]):
            continue

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        n_i = direction.spherical_to_cartesian(normal[i])

        t_from_n_i = basis3.tangent_from_normal(n_i)
        b_from_n_i = ti.math.cross(n_i, t_from_n_i)
        world_to_tangent_i = transform3.compute_frame_to_canonical_matrix(
            t_from_n_i, b_from_n_i, n_i, ti.math.vec3(0.0, 0.0, 0.0)
        ).transpose()

        t_i_cov_00 = 0.0
        t_i_cov_01 = 0.0
        t_i_cov_11 = 0.0

        for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
            j = i + shifter - ti.math.ivec3(1, 1, 1)

            if not grid3.is_valid_cell_3dindex(j, normal.shape) or (i == j).all():
                continue

            if ti.math.isnan(normal[j][0]):
                continue

            # Ignoring some neighbors may improve convergence
            p_ignore_neighbor = P_IGNORE_NEIGHBOR_START * (
                1.0 - ti.math.min(iteration_number / STOP_IGNORING_NEIGHBOR, 1.0)
            )

            if p_ignore_neighbor > 0.0:
                random_float = random.pcgf_7_to_1(
                    math.uvec7(ti.u32(SEED + iteration_number), i, j)
                )
                if random_float < p_ignore_neighbor:
                    continue

            p_j = grid3.cell_center_point(j, origin, cell_sides_length)
            w_ij = math.eval_triangle_filter_normalized(
                p_j,
                p_i,
                SPATIAL_FILTER_RADIUS * grid3.cell_diagonal_length(cell_sides_length),
            )
            w_ij = w_ij**SPATIAL_WEIGHT_EXPONENT

            t_j_ts_j = ti.math.vec3(direction.polar_to_cartesian(phi_t_in[j]), 0.0)
            n_j = direction.spherical_to_cartesian(normal[j])

            t_from_n_j = basis3.tangent_from_normal(n_j)
            b_from_n_j = ti.math.cross(n_j, t_from_n_j)
            tangent_j_to_world = transform3.compute_frame_to_canonical_matrix(
                t_from_n_j, b_from_n_j, n_j, ti.math.vec3(0.0, 0.0, 0.0)
            )

            t_j_ts_i = transform3.apply_to_vector(
                world_to_tangent_i @ tangent_j_to_world, t_j_ts_j
            )

            t_i_cov_00 += w_ij * t_j_ts_i[0] * t_j_ts_i[0]
            t_i_cov_01 += w_ij * t_j_ts_i[0] * t_j_ts_i[1]
            t_i_cov_11 += w_ij * t_j_ts_i[1] * t_j_ts_i[1]

        t_i_cov = ti.math.mat2(
            [
                [t_i_cov_00, t_i_cov_01],
                [t_i_cov_01, t_i_cov_11],
            ]
        )

        t_i = math.eigenvec2_with_highest_eigenvalue_iterative(t_i_cov)
        phi_t_i = ti.math.atan2(t_i.y, t_i.x)
        if tangent_boundary_is_constrained(state[i]):
            phi_t_i = phi_t_in[i]
        phi_t_out[i] = phi_t_i


@ti.kernel
def basis3_field_prolong(
    normal: ti.template(),
    phi_t: ti.template(),
    normal_prolonged: ti.template(),
    phi_t_prolonged: ti.template(),
    state_prolonged: ti.template(),
):
    for i in ti.grouped(normal):
        is_masked_i = ti.math.isnan(normal[i][0])
        if is_masked_i:
            continue

        n_i = direction.spherical_to_cartesian(normal[i])
        # ts_i: i's tangent_space
        t_i_ts_i = ti.math.vec3(direction.polar_to_cartesian(phi_t[i]), 0.0)

        t_from_n_i = basis3.tangent_from_normal(n_i)
        b_from_n_i = ti.math.cross(n_i, t_from_n_i)
        tangent_i_to_world = transform3.compute_frame_to_canonical_matrix(
            t_from_n_i, b_from_n_i, n_i, ti.math.vec3(0.0, 0.0, 0.0)
        )
        t_i = transform3.apply_to_vector(tangent_i_to_world, t_i_ts_i)

        j_block_origin = i * 2

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal_prolonged.shape)
            if is_invalid_j:
                continue

            is_masked_j = ti.math.isnan(normal_prolonged[j][0])
            if is_masked_j:
                continue

            if not tangent_boundary_is_constrained(state_prolonged[j]):
                n_j = direction.spherical_to_cartesian(normal_prolonged[j])
                t_from_n_j = basis3.tangent_from_normal(n_j)
                b_from_n_j = ti.math.cross(n_j, t_from_n_j)
                world_to_tangent_j = transform3.compute_frame_to_canonical_matrix(
                    t_from_n_j, b_from_n_j, n_j, ti.math.vec3(0.0, 0.0, 0.0)
                ).transpose()

                t_i_ts_j = transform3.apply_to_vector(world_to_tangent_j, t_i)
                phi_t_prolonged[j] = ti.atan2(t_i_ts_j.y, t_i_ts_j.x)


@ti.kernel
def frame_field_filter_point_too_close_to_boundary(
    sdf: ti.template(), cell_sides_length: float, point: ti.template()
):
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    for i in ti.grouped(point):
        if ti.math.isnan(point[i]).any():
            continue

        boundary_normal_i = solid3.sdf_compute_closest_normal_central(
            sdf, i, cell_sides_length
        )

        bn_is_pointing_down = boundary_normal_i.z < 0.0
        if bn_is_pointing_down:
            boundary_normal_i = -boundary_normal_i

        boundary_normal_i_sph = direction.cartesian_to_spherical(boundary_normal_i)
        normal_polar_angle = boundary_normal_i_sph[0]
        is_equator = ti.abs((normal_polar_angle - ti.math.pi * 0.5)) < WALL_MAX_ANGLE
        is_wall_region_i = (
            is_boundary_region(sdf[i], deposition_width * 0.25) and is_equator
        )
        is_top_or_bottom_region_i = (
            is_boundary_region(sdf[i], layer_height * 0.25) and not is_equator
        )

        if is_wall_region_i or is_top_or_bottom_region_i:
            point[i].x = ti.math.nan


@ti.kernel
def sdf_generate_infill(
    input_sdf: ti.template(),
    output_sdf: ti.template(),
    cell_sides_length: float,
    offset: ti.math.vec3,
    period: float,
    shell: float,
    gyroid: bool,
):
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    deposition_width = deposition_width_from_layer_height(layer_height)

    wall_width = shell * deposition_width
    grid_scale = period * deposition_width

    origin = ti.math.vec3(0.0)
    for sdf_i3 in ti.grouped(input_sdf):
        cell_center = (
            grid3.cell_center_point(sdf_i3, origin, cell_sides_length) + offset
        )
        # Wall width: 2 deposition width
        fromCenter = (
            -ti.abs(ti.math.mod(cell_center, grid_scale) - 0.5 * grid_scale)
            + 0.5 * grid_scale
            - 0.5
            * ti.math.vec3(
                deposition_width * 2.0, deposition_width * 2.0, layer_height * 2
            )
        )
        # Wall width: 1 deposition width
        # fromCenter = (
        #     -ti.abs(ti.math.mod(cell_center, grid_scale) - 0.5 * grid_scale)
        #     + 0.5 * grid_scale
        #     - 0.5 * ti.math.vec3(deposition_width, deposition_width, layer_height * 2)
        # )
        # infill_sdf is the SDF of the infill, the intersection and shell will be added after

        # infill_sdf = ti.min(ti.min(fromCenter.x, fromCenter.y), fromCenter.z)
        # infill_sdf = ti.min(fromCenter.x, fromCenter.y)
        infill_sdf = fromCenter.x

        if gyroid:
            p = cell_center / (grid_scale * 0.1)
            p[2] /= 8
            infill_sdf = (
                ti.math.dot(ti.math.sin(p.xyz), ti.math.cos(p.yzx))
                * 0.7
                * (grid_scale * 0.1)
            )
            infill_sdf = abs(infill_sdf) - deposition_width * 0.5

        infill = ti.max(input_sdf[sdf_i3], infill_sdf)
        hollow_sdf = (
            -input_sdf[sdf_i3] - wall_width
            if input_sdf[sdf_i3] < -0.5 * wall_width
            else input_sdf[sdf_i3]
        )
        output_sdf[sdf_i3] = ti.min(infill, hollow_sdf)


@ti.kernel
def flag_supports(
    sdf: ti.template(), cell_3dcount: ti.math.ivec3, flag_field: ti.template()
):
    for sdf_i3 in ti.grouped(sdf):
        flag_field[sdf_i3] = ti.int8(0)
        for z in range(sdf_i3.z, cell_3dcount.z):
            if sdf[ti.math.ivec3(sdf_i3.x, sdf_i3.y, z)] < 0:
                flag_field[sdf_i3] = ti.int8(1)
                break


@ti.kernel
def process_flagged_support(
    input_sdf: ti.template(),
    output_sdf: ti.template(),
    flag_field: ti.template(),
    cell_sides_length: float,
    offset: ti.math.vec3,
    period: float,
):
    origin = ti.math.vec3(0.0)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    nozzle_width = deposition_width_from_layer_height(layer_height)
    grid_scale = period * nozzle_width

    for sdf_i3 in ti.grouped(input_sdf):
        if flag_field[sdf_i3] == 1 and input_sdf[sdf_i3] > 0:
            cell_center = (
                grid3.cell_center_point(sdf_i3, origin, cell_sides_length)
                + ti.math.vec3(0, 0, 4 * layer_height)
                + offset
            )
            fromCenter = (
                -ti.abs(ti.math.mod(cell_center, grid_scale) - 0.5 * grid_scale)
                + 0.5 * grid_scale
                - 0.5 * ti.math.vec3(nozzle_width, nozzle_width, layer_height)
            )
            support_sdf = ti.min(
                ti.min(
                    ti.max(fromCenter.x, fromCenter.y),
                    ti.max(fromCenter.y, fromCenter.z),
                ),
                ti.max(fromCenter.x, fromCenter.z),
            )

            # Write the support SDF in this case
            output_sdf[sdf_i3] = support_sdf
        else:
            output_sdf[sdf_i3] = input_sdf[sdf_i3]


@ti.func
def layer_height_from_deposition_width(nozzle_width: float) -> float:
    return nozzle_width * LAYER_HEIGHT_WRT_NOZZLE


@ti.func
def deposition_width_from_layer_height(nozzle_width: float) -> float:
    return nozzle_width / LAYER_HEIGHT_WRT_NOZZLE


@ti.func
def layer_height_from_space_sampling_period(period: float) -> float:
    return period / SPACE_SAMPLING_WRT_LAYER_HEIGHT


@ti.func
def space_sampling_period_from_layer_height(layer_height: float) -> float:
    return layer_height * SPACE_SAMPLING_WRT_LAYER_HEIGHT


@ti.func
def space_sampling_period_from_cell_sides_length(cell_sides_length: float):
    return cell_sides_length / CELL_SIDES_LENGTH_WRT_SPACE_SAMPLING


@ti.func
def cell_sides_length_from_space_sampling_period(period: float) -> float:
    return period * CELL_SIDES_LENGTH_WRT_SPACE_SAMPLING


@ti.func
def deposition_width_from_cell_sides_length(cell_sides_length: float) -> float:
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    return deposition_width_from_layer_height(layer_height)


@ti.func
def layer_height_from_cell_sides_length(grid_cell_sides_length: float) -> float:
    period = space_sampling_period_from_cell_sides_length(grid_cell_sides_length)
    layer_height = layer_height_from_space_sampling_period(period)
    return layer_height


@ti.kernel
def layer_height_from_cell_sides_length_kernel(grid_cell_sides_length: float) -> float:
    return layer_height_from_cell_sides_length(grid_cell_sides_length)


@ti.func
def cell_sides_length_from_deposition_width(deposition_width: float) -> float:
    # Period
    layer_height = layer_height_from_deposition_width(deposition_width)
    # Nyquist: half the period
    space_sampling_period = space_sampling_period_from_layer_height(layer_height)
    return cell_sides_length_from_space_sampling_period(space_sampling_period)


@ti.kernel
def cell_sides_length_from_deposition_width_kernel(deposition_width: float) -> float:
    return cell_sides_length_from_deposition_width(deposition_width)


@ti.func
def get_top_or_bottom_level_1(cell_width: float, top_and_bottom_count: int) -> float:
    layer_height = layer_height_from_cell_sides_length(cell_width)
    return -layer_height * (top_and_bottom_count + 0.5)


@ti.func
def is_boundary_region(sd_to_closest_point, boundary_width):
    boundary_level_0 = 0.0
    boundary_level_1 = -boundary_width
    return (
        sd_to_closest_point < boundary_level_0
        and sd_to_closest_point > boundary_level_1
    )


@ti.func
def is_after_boundary(sd_to_closest_point):
    boundary_level = 0.0
    return sd_to_closest_point >= boundary_level


@ti.func
def cos_period_along_normal_1ex_from_layer_height(
    layer_height: float,
) -> float:
    return layer_height


@ti.kernel
def cos_period_along_normal_1extraction_from_layer_height_kernel(
    layer_height: float,
) -> float:
    return cos_period_along_normal_1ex_from_layer_height(layer_height)


@ti.func
def cos_triperiod_1ex_from_cell_sides_length(cell_sides_length: float) -> ti.math.vec3:
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    cos_period_along_normal = cos_period_along_normal_1ex_from_layer_height(
        layer_height
    )
    # Tangent: 1 point extraction1 per cos period. Cos period: layer height
    # Bitangent: 1 trajectory extraction per cos period. Cos period: nozzle width
    # Normal: 1 layer extraction per cos period. Cos period: layer_height
    return ti.math.vec3(layer_height, deposition_width, cos_period_along_normal)


@ti.func
def triperiod_from_cell_sides_length(cell_sides_length: float) -> ti.math.vec3:
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    return ti.math.vec3(layer_height, deposition_width, layer_height)


@ti.kernel
def triperiod_from_cell_sides_length_kernel(cell_sides_length: float) -> ti.math.vec3:
    return triperiod_from_cell_sides_length(cell_sides_length)


@ti.kernel
def cos_triperiod_1ex_from_cell_sides_length_kernel(
    cell_sides_length: float,
) -> ti.math.vec3:
    return cos_triperiod_1ex_from_cell_sides_length(cell_sides_length)


@ti.func
def distance_to_phase_1extraction_per_cos_along_normal(d: float, layer_height) -> float:
    # Shift origin with half a layer height
    shift = layer_height * 0.5
    # Then divide by the cosine period to have an increase of one for each
    # cosine period
    cos_period = cos_period_along_normal_1ex_from_layer_height(layer_height)
    # Finally, multiply by two pi to match the period of cos
    return (d + shift) / cos_period * 2.0 * ti.math.pi


@ti.func
def distance_to_phase_bitangent_1ex(d: float, deposition_width) -> float:
    cell_sides_length = cell_sides_length_from_deposition_width(deposition_width)
    cos_period_b = cos_triperiod_1ex_from_cell_sides_length(cell_sides_length)[1]
    # Shift origin with half a deposition width
    shift = deposition_width * 0.5
    # Then divide by the cosine period to have an increase of one for each
    # cosine period.
    # Finally, multiply by two pi to match the period of cos
    return (d + shift) / cos_period_b * 2.0 * ti.math.pi


@ti.func
def neighborhood_radius(cell_sides_length: float) -> float:
    deposition_width = deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = layer_height_from_cell_sides_length(cell_sides_length)
    t_side_length = layer_height * toolpath3.THRESHOLD_T_0
    b_side_length = deposition_width * toolpath3.THRESHOLD_B_1
    n_side_length = layer_height * toolpath3.THRESHOLD_N_2
    # * 1.05 to have some margin
    return ti.math.length(
        ti.math.vec3(t_side_length, b_side_length, n_side_length) * 1.05
    )


@ti.kernel
def neighborhood_radius_kernel(cell_sides_length: float) -> float:
    return neighborhood_radius(cell_sides_length)


@ti.func
def tangent_boundary_constrain(state: ti.u32) -> ti.u32:
    state |= 0b0001
    return state


@ti.func
def tangent_boundary_unconstrain(state: ti.u32) -> ti.u32:
    state &= 0b1110
    return state


@ti.func
def tangent_boundary_is_constrained(state: ti.u32) -> int:
    mask = 0b0001
    masked = mask & state
    return masked == mask


# DEPRECATED
@ti.func
def tangent_inside_constrain(state: ti.u32) -> ti.u32:
    state |= 0b0010
    return state


# DEPRECATED
@ti.func
def tangent_inside_unconstrain(state: ti.u32) -> ti.u32:
    state &= 0b1101
    return state


# DEPRECATED
@ti.func
def tangent_inside_is_constrained(state: ti.u32) -> int:
    mask = 0b0010
    masked = mask & state
    return masked == mask


@ti.func
def tangent_set_is_wall(state: ti.u32) -> ti.u32:
    state |= 0b0100
    return state


@ti.func
def tangent_unset_is_wall(state: ti.u32) -> ti.u32:
    state &= 0b1011
    return state


@ti.func
def tangent_is_wall(state: ti.u32) -> int:
    mask = 0b0100
    masked = mask & state
    return masked == mask
