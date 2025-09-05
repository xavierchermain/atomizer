import sys
import math
import bpy
from mathutils import Vector

# ---------- Parse CLI args ----------
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
if not argv:
    raise ValueError("Please provide a PLY path after '--'")

ply_path = argv[0]
print(f"📂 PLY: {ply_path}")

attr_override = argv[1] if len(argv) > 1 else None

# ---------- Clear scene ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ---------- Import PLY (built-in in Blender 4.5+) ----------
res = bpy.ops.wm.ply_import(filepath=ply_path)
if 'FINISHED' not in res:
    raise RuntimeError("Failed to import PLY. Check the path and permissions.")

# Get imported object
obj = bpy.context.selected_objects[0]
mesh = obj.data

# ---------- Find a vertex color attribute ----------
attr_name = None
if attr_override:
    attr_name = attr_override
elif mesh.color_attributes:
    attr_name = (mesh.color_attributes.active.name 
                 if mesh.color_attributes.active 
                 else mesh.color_attributes[0].name)

if not attr_name:
    print("⚠️ No color attribute found. Material will still be created, but will look grey.")
else:
    print(f"✅ Using color attribute: {attr_name}")

# ---------- Create material ----------
mat = bpy.data.materials.new(name="VertexColorMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
for n in list(nodes): nodes.remove(n)

out = nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0, 0)
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

if attr_name:
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.location = (-400, 0)
    attr_node.attribute_name = attr_name
    links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])

obj.data.materials.clear()
obj.data.materials.append(mat)

# ---------- Utility: compute world-space bounds center & radius ----------
def object_bounds_center_radius(o):
    corners = [o.matrix_world @ Vector(c) for c in o.bound_box]
    center = sum(corners, Vector()) / 8.0
    radius = max((c - center).length for c in corners)
    # Fallback for degenerate bounds
    if radius == 0.0:
        radius = max(o.dimensions) * 0.5
    return center, radius

center, radius = object_bounds_center_radius(obj)

# ---------- Ensure a camera that sees the object ----------
def ensure_camera_for_object(o, center, radius):
    cam = bpy.context.scene.camera
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.data.objects.new("AutoCamera", bpy.data.cameras.new("AutoCamera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam

    cam.data.lens = 50  # mm, reasonable default
    cam.data.clip_start = 0.01
    cam.data.clip_end = max(1000.0, radius * 100.0)

    # Compute distance to fit object in view (simple sphere fit)
    # fov is horizontal by default for camera.angle_x, use the tighter of x/y to be safe.
    angle_x = cam.data.angle_x
    angle_y = cam.data.angle_y
    fov = min(angle_x, angle_y)
    # distance so that radius fits in half-FOV, with margin
    margin = 1.2
    dist = (radius * margin) / math.tan(fov * 0.5)
    if not math.isfinite(dist) or dist <= 0:
        dist = max(1.0, radius * 3.0)

    # Place camera on a nice 3/4 view above and to the side
    cam.location = center + Vector((dist * 0.6, -dist, dist * 0.5))

    # Aim camera at center (-Z forward, Y up for cameras)
    direction = (center - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Optional: set an empty as focus target for depth of field
    if "AutoFocus" not in bpy.data.objects:
        empty = bpy.data.objects.new("AutoFocus", None)
        bpy.context.collection.objects.link(empty)
    else:
        empty = bpy.data.objects["AutoFocus"]
    empty.location = center
    cam.data.dof.use_dof = False  # turn on if you want DoF: True
    cam.data.dof.focus_object = empty

    return cam

cam = ensure_camera_for_object(obj, center, radius)

# ---------- Frame in 3D View & set Material Preview if UI exists ----------
scr = getattr(bpy.context, "screen", None)
if scr is not None:
    # Ensure selection/active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # We’ll try all windows/areas to maximize the chance of hitting a 3D region
    moved_any = False
    for win in bpy.context.window_manager.windows:
        # Make sure the scene is set (some ops need this)
        win.scene = bpy.context.scene
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            # Set Material Preview shading
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    break

            # Try operator first with a correct override
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is not None:
                try:
                    with bpy.context.temp_override(window=win, area=area, region=region, scene=bpy.context.scene, view_layer=bpy.context.view_layer, active_object=obj):
                        r = bpy.ops.view3d.view_selected(use_all_regions=False)
                        if r == {'FINISHED'}:
                            moved_any = True
                            continue
                except Exception as e:
                    print(f"view_selected failed in a region: {e}")

            # Fallback: set RegionView3D directly
            try:
                space = next(s for s in area.spaces if s.type == 'VIEW_3D')
                rv3d = space.region_3d
                # Look at object center and set a reasonable distance
                rv3d.view_location = center
                rv3d.view_distance = max(radius * 2.2, 0.5)
                # Align the view roughly to the camera view orientation
                quat = (center - (center + Vector((1, 0, 0)))).to_track_quat('-Z', 'Y')
                # If you want a nicer angle, reuse the camera direction:
                direction = (center - bpy.context.scene.camera.location).normalized() if bpy.context.scene.camera else Vector((0, -1, 0))
                quat = direction.to_track_quat('-Z', 'Y')
                rv3d.view_rotation = quat
                moved_any = True
            except Exception as e:
                print(f"Direct RegionView3D set failed: {e}")

    if not moved_any:
        print("⚠️ Could not move any 3D view (are you running in background or without a 3D View visible?).")
else:
    print("ℹ️ No UI screen detected; viewport cannot be moved in background mode.")


