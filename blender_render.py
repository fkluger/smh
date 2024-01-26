import bpy
import os
import numpy as np
import mathutils
import pickle

from mathutils.bvhtree import BVHTree
import bmesh

mesh_path = "./city.glb"
target_folder = "./render"

do_render = True

step_sizes = [1, 2, 4]
focal_lengths = [24, 30, 45]

ry = 0
rx_max = 115
rx_min = 90

coarse_paths_file = os.path.join("./", "coarse_paths.pkl")

os.makedirs(target_folder, exist_ok=True)

def delete_all_objects():
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.delete()
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

def get_calibration_matrix_from_blender(cam):
    camd = cam.data
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0

    K = np.array([
        [alpha_u, skew, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1]])
    return K


def get_object_center(o):
    vcos = []
    obs = get_children(o) + [o]
    for ob in obs:
        if ob.data is not None:
            M = np.array(ob.matrix_world)
            v_orig = [np.array([v.co[0], v.co[1], v.co[2], 1]) for v in ob.data.vertices]
            vcos += [M @ v for v in v_orig]
    findCenter = lambda l: (max(l) + min(l)) / 2
    x, y, z = [[v[i] for v in vcos] for i in range(3)]
    object_center = np.array([findCenter(axis) for axis in [x, y, z]])
    return object_center


def get_children(myObject):
    children = []
    for ob in bpy.data.objects:
        if ob.parent == myObject:
            children.append(ob)
    for child in children:
        subchildren = get_children(child)
        children += subchildren

    return children


def polygon_line_intersection(polygon, line):
    p1, p2, p3, plane_normal, _ = polygon
    l1, l2 = line
    line_direction = l2 - l1
    dot_product = np.dot(line_direction, plane_normal)
    if abs(dot_product) < 1e-6:
        return False

    t = np.dot(p1 - l1, plane_normal) / dot_product
    intersection_point = l1 + t * line_direction

    if t < 0 or t > 1:
        return False

    edge1 = p2 - p1
    edge2 = p3 - p2
    edge3 = p1 - p3
    edge_normals = np.cross([edge1, edge2, edge3], [plane_normal, plane_normal, plane_normal])
    side1 = np.dot(intersection_point - p1, edge_normals[0])
    side2 = np.dot(intersection_point - p2, edge_normals[1])
    side3 = np.dot(intersection_point - p3, edge_normals[2])

    if (side1 > 0 and side2 > 0 and side3 > 0) or (side1 < 0 and side2 < 0 and side3 < 0):
        return True
    else:
        return False


def get_all_polygons():
    poly_vertices = []
    for o in bpy.context.scene.objects:
        if o.data is not None:

            M = np.array(o.matrix_world)
            for poly in o.data.polygons:
                c = np.array(poly.center)
                n = np.array(poly.normal)
                c = M[:3, :3] @ c + M[:3, 3]
                n = M[:3, :3] @ n

                verts = []
                for vi in poly.vertices:
                    vert = o.data.vertices[vi].co
                    vert = np.array([vert[0], vert[1], vert[2], 1])
                    vert = M @ vert
                    verts += [vert[:3]]
                poly_vertices += [verts + [n, c]]
    return poly_vertices

def all_polygons_line_intersection(polygons, line):
    for polygon in polygons:
        if polygon_line_intersection(polygon, line):
            return True
    return False

delete_all_objects()

bpy.ops.import_scene.gltf(filepath=mesh_path)

for ob in bpy.context.selected_objects:
    ob.select_set(False)

spectator_positions = []
for o in bpy.context.scene.objects:
    for keyword in ["sedan", "van", "suv", "police", "truck"]:
        if keyword in o.name:
            o.select_set(True)
            spectator_positions += [get_object_center(o)]

            for child in o.children:
                child.select_set(True)

            break

bpy.ops.object.delete()

spectator_positions = [np.array([x, y, 1.80]) for x, y, z in spectator_positions]
coarse_paths = [[spec] for spec in spectator_positions]

def min_path_distance(path1, path2, polygons):
    start1, end1 = path1[0], path1[-1]
    start2, end2 = path2[0], path2[-1]
    d1 = np.linalg.norm(start1 - start2)
    d2 = np.linalg.norm(start1 - end2)
    d3 = np.linalg.norm(end1 - start2)
    d4 = np.linalg.norm(end1 - end2)
    dists = [d1, d2, d3, d4]

    ints = [all_polygons_line_intersection(polygons, [start1, start2]),
            all_polygons_line_intersection(polygons, [start1, end2]),
            all_polygons_line_intersection(polygons, [end1, start2]),
            all_polygons_line_intersection(polygons, [end1, end2])]

    if np.all(ints):
        return None, None

    for i in range(4):
        if not ints[i]:
            is_best = True
            for j in range(3):
                if not ints[(i + j) % 4] and dists[(i + j) % 4] < dists[i]:
                    is_best = False
                    break
            if is_best:
                return dists[i], i

    return None, None


def merge_paths(path1, path2, adj):
    if adj == 0:
        return path1[::-1] + path2
    elif adj == 1:
        return path2 + path1
    elif adj == 2:
        return path1 + path2
    elif adj == 3:
        return path1 + path2[::-1]
    else:
        assert False

all_polygons = get_all_polygons()
polygons = [poly for poly in all_polygons if np.any([p[2] < 2.0 for p in poly[:3]])]

print("polygons: ", len(polygons))
print("positions: ", len(coarse_paths))

if os.path.exists(coarse_paths_file):
    with open(coarse_paths_file, 'rb') as f:
        coarse_paths = pickle.load(f)
else:
    converged = False
    iteration = 0
    while not converged:
        converged = True
        for qi, query_path in enumerate(coarse_paths):
            print(qi)

            min_dist = 1e9
            min_adj = None
            min_idx = None

            for oi in range(qi + 1, len(coarse_paths)):
                other_path = coarse_paths[oi]
                dist, adj = min_path_distance(query_path, other_path, polygons)
                if dist is not None:
                    if dist > 0 and dist < min_dist:
                        min_dist = dist
                        min_adj = adj
                        min_idx = oi

            if min_idx is not None:
                converged = False
                coarse_paths[qi] = merge_paths(query_path, coarse_paths.pop(min_idx), min_adj)

        print(iteration, "paths:")
        for path in coarse_paths:
            print(path)

        iteration += 1

    with open(coarse_paths_file, 'wb') as f:
        pickle.dump(coarse_paths, f, pickle.HIGHEST_PROTOCOL)

bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(90 * np.pi / 180, 0, 0))
cam = bpy.context.object
cam.data.lens = 35
cam.data.sensor_width = 30
cam.data.sensor_height = 30
scene = bpy.data.scenes[0]
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.camera = cam

light_height = 500

def rot_from_vecs(v1, v2):
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)
    return R


def euler_from_mat(R):
    rx = np.arctan2(R[2, 1], R[2, 2])
    tmp = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    ry = np.arctan2(-R[2, 0], tmp)
    rz = np.arctan2(R[1, 0], R[0, 0])
    return rx, ry, rz


# Create a BVH tree and return bvh and vertices in world coordinates
def BVHTreeAndVerticesInWorldFromObj(obj):
    M = np.array(obj.matrix_world)

    verts = [np.array([v.co[0], v.co[1], v.co[2], 1]) for v in obj.data.vertices]
    vertsInWorld = [M @ v for v in verts]
    vertsInWorld = [mathutils.Vector(v[:3]) for v in vertsInWorld]

    bvh = BVHTree.FromPolygons(vertsInWorld, [[vi for vi in p.vertices] for p in obj.data.polygons])

    poly_vertices = []
    M = np.array(obj.matrix_world)
    for poly in obj.data.polygons:
        c = np.array(poly.center)
        n = np.array(poly.normal)
        c = M[:3, :3] @ c + M[:3, 3]
        n = M[:3, :3] @ n

        verts = []
        for vi in poly.vertices:
            vert = obj.data.vertices[vi].co
            vert = np.array([vert[0], vert[1], vert[2], 1])
            vert = M @ vert
            verts += [vert[:3]]
        poly_vertices += [verts + [n, c]]

    return bvh, vertsInWorld, poly_vertices


# Deselect mesh polygons and vertices
def DeselectEdgesAndPolygons(obj):
    for p in obj.data.polygons:
        p.select = False
    for e in obj.data.edges:
        e.select = False


def pixels_to_rays(Kinv, R, resolution):

    xs = np.arange(0, resolution[1], 1)
    ys = np.arange(0, resolution[0], 1)

    xx, yy = np.meshgrid(xs, ys)
    ones = np.ones_like(xx.flatten())
    x = np.stack([xx.flatten(), yy.flatten(), ones], axis=-1)

    X = Kinv @ x.T
    X = R @ X
    X /= np.linalg.norm(X, axis=0, keepdims=True)

    return X.T, x

bpy.ops.object.select_all(action='DESELECT')
mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
for obj in mesh:
    obj.select_set(state=True)
    bpy.context.view_layer.objects.active = obj
bpy.ops.object.join()

only_mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH'][0]

bpy.ops.object.select_all(action='DESELECT')

limit = 0.0001
DeselectEdgesAndPolygons(only_mesh)
scene = bpy.context.scene
bvh, vertices, all_polygons = BVHTreeAndVerticesInWorldFromObj(only_mesh)

me = only_mesh.data

only_mesh.select_set(True)

bpy.ops.object.mode_set(mode='EDIT')

bm = bmesh.from_edit_mesh(me)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()
for face in bm.faces:
    face.select = False
bmesh.update_edit_mesh(me)

poly_vertices = []
poly_normals = []
poly_centers = []

for polygon in all_polygons:
    poly_vertices += [polygon[:3]]
    n = polygon[3]
    poly_normals += [n]
    face_center = polygon[4]
    poly_centers += [face_center]

np.savez(os.path.join(target_folder, "polygons.npz"), centers=poly_centers, normals=poly_normals, vertices=poly_vertices)

light_locations = [
    [x, y, light_height] for x in range(0, -401, -100) for y in range(0, 401, 100)]

for loc in light_locations:
    light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
    light_data.energy = 20e5
    light_object = bpy.data.objects.new(name="my-light", object_data=light_data)
    light_object.location = loc
    bpy.context.collection.objects.link(light_object)


if do_render:
    for orientation_id in [1, 2]:
        for focal_length in focal_lengths:
            cam.data.lens = focal_length
            bpy.context.view_layer.update()

            for target_step_size in step_sizes:

                for path_idx, path in enumerate(coarse_paths):
                    rx = 100
                    old_rz = 0

                    for coarse_idx in range(len(path) - 1):

                        p1 = path[coarse_idx]
                        p2 = path[coarse_idx + 1]
                        vec = p2 - p1
                        path_len = np.linalg.norm(vec)
                        vec = vec / path_len

                        if orientation_id == 0:
                            rtvec = vec
                        elif orientation_id == 1:
                            rtvec = -vec
                        elif orientation_id == 2:
                            rtvec = np.array([vec[1], -vec[0], vec[2]])
                        elif orientation_id == 3:
                            rtvec = np.array([-vec[1], vec[0], vec[2]])

                        R = rot_from_vecs([0, 0, -1], rtvec)
                        _, _, new_rz = euler_from_mat(R)

                        num_steps = max(int(path_len / target_step_size), 1)
                        t_step_size = path_len / num_steps

                        r_diff = new_rz - old_rz
                        r_step_size = r_diff / num_steps

                        for step_idx in range(num_steps):
                            c = p1 + step_idx * t_step_size * vec
                            cam.location = c

                            rz = old_rz + r_step_size * step_idx

                            rx += np.random.normal(0, 1)
                            rx = np.clip(rx, rx_min, rx_max)

                            cam.rotation_euler = [rx * np.pi / 180.0, ry, rz]

                            bpy.context.view_layer.update()

                            scene.render.image_settings.file_format = 'PNG'
                            scene.render.filepath = os.path.join(target_folder, "render_%d_%d_%d_%04d_%04d_%04d.png" % (
                            orientation_id, focal_length, target_step_size, path_idx, coarse_idx, step_idx))

                            poly_path = os.path.join(target_folder, "camera_%d_%d_%d_%04d_%04d_%04d.npz" % (
                                orientation_id, focal_length, target_step_size, path_idx, coarse_idx, step_idx))

                            if os.path.exists(scene.render.filepath) and os.path.exists(poly_path):
                                continue


                            M = mathutils.Matrix().to_4x4()
                            M[1][1] = -1
                            M[2][2] = -1
                            camera_pose = M @ cam.matrix_world.inverted()

                            K = get_calibration_matrix_from_blender(cam)
                            K[1,1] *= -1

                            R_cam = np.array(camera_pose)[:3,:3]
                            t_cam = np.array(camera_pose)[:3,3]

                            M = mathutils.Matrix().to_4x4()
                            M[2][2] = -1
                            Minv = M @ cam.matrix_world.inverted()
                            Kinv = np.linalg.inv(K)
                            R = np.array(Minv)[:3, :3].T
                            t = np.array(Minv)[:3, 3]
                            visible_polys = []

                            rays, pixels = pixels_to_rays(Kinv, R, (scene.render.resolution_y, scene.render.resolution_x))

                            C = cam.location

                            for ri in range(rays.shape[0]):
                                x = pixels[ri,0]
                                y = pixels[ri,1]
                                location, normal, index, distance = bvh.ray_cast(C, rays[ri])

                                if location:
                                    visible_polys += [index]

                            visible_polys = np.array(list(set(visible_polys)))


                            np.savez(poly_path, K=K, R=R, t=t, polygons=visible_polys)

                        old_rz = new_rz