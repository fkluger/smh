import numpy as np

def homography_distance(X, H):
    N = X.shape[0]

    x1 = X[..., 0:2]
    x2 = X[..., 2:4]
    X1 = np.ones([N, 3, 1])
    X2 = np.ones([N, 3, 1])
    X1[..., 0:2, 0] = x1
    X2[..., 0:2, 0] = x2

    HX1 = H[:, None, ...] @ X1[None]
    HX1[..., 0, 0] /= HX1[..., 2, 0]
    HX1[..., 1, 0] /= HX1[..., 2, 0]
    HX1[..., 2, 0] /= HX1[..., 2, 0]
    HX2 = np.linalg.inv(H)[:, None, ...] @ X2[None]
    HX2[..., 0, 0] /= HX2[..., 2, 0]
    HX2[..., 1, 0] /= HX2[..., 2, 0]
    HX2[..., 2, 0] /= HX2[..., 2, 0]

    signed_distances_1 = HX1 - X2
    distances_1 = np.sum(signed_distances_1 * signed_distances_1, axis=-2)[..., 0]
    signed_distances_2 = HX2 - X1
    distances_2 = np.sum(signed_distances_2 * signed_distances_2, axis=-2)[..., 0]

    distances = distances_1 + distances_2

    return distances


def tri_signs(p1, p2, p3):
    return (p1[..., 0]-p3[..., 0])*(p2[..., 1]-p3[..., 1])*(p2[..., 0]-p3[..., 0])*(p1[..., 1]-p3[..., 1])

def signed_volume(a, b, c, d):
    cross_prod = np.cross(b-a, c-a)
    dot_prod = np.sum(cross_prod * (d-a), axis=-1)
    return 1.0/6.0*dot_prod


def check_rays(polygons, origin, rays):
    # https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    N = polygons.shape[0]
    M = rays.shape[0]

    q1 = origin

    does_intersect = np.zeros((N, M), dtype=bool)
    intersection = np.zeros((N, M, 3), dtype=float)

    for i in range(polygons.shape[0]):

        p1 = polygons[i, 0, :3][None, ...]
        p2 = polygons[i, 1, :3][None, ...]
        p3 = polygons[i, 2, :3][None, ...]

        q2 = q1 + rays * 1000

        s1 = signed_volume(q1, p1, p2, p3).squeeze()
        s2 = signed_volume(q2, p1, p2, p3).squeeze()
        check1 = np.sign(s1) != np.sign(s2) # HERE
        s3 = signed_volume(q1, q2, p1, p2).squeeze()
        s4 = signed_volume(q1, q2, p2, p3).squeeze()
        s5 = signed_volume(q1, q2, p3, p1).squeeze()
        check2 = (np.sign(s3) == np.sign(s4)) & (np.sign(s5) == np.sign(s4)) & (np.sign(s3) == np.sign(s5))

        does_intersect[i] = check1 & check2

        n = np.cross(p2 - p1, p3 - p1)
        v1 = p1-q1
        v2 = rays
        t = (np.dot(v1, n.T) / np.dot(v2, n.T))

        s = q1 + t * v2

        intersection[i] = s

    return does_intersect, intersection


def pixel_to_ray(K, R, x):
    Kinv = np.linalg.inv(K)
    X = Kinv @ x.T
    X = R.T @ X
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    return X.T


def ray_cast(K1, K2, R1, R2, t1, t2, poly_verts, points1, points2):

    M1 = np.zeros((4, 4))
    M1[:3, :3] = R1
    M1[:3, -1] = t1
    M1[-1, -1] = 1

    M2 = np.zeros((4, 4))
    M2[:3, :3] = R2
    M2[:3, -1] = t2
    M2[-1, -1] = 1

    C1 = -R1.T @ t1[:, None]
    C2 = -R2.T @ t2[:, None]

    P1 = K1 @ np.eye(3, 4) @ M1
    P2 = K2 @ np.eye(3, 4) @ M2

    verts1 = (P1 @ poly_verts[..., None])[..., 0]
    verts2 = (P2 @ poly_verts[..., None])[..., 0]
    verts1 /= verts1[..., -1][..., None]
    verts2 /= verts2[..., -1][..., None]

    raydir1 = pixel_to_ray(K1, R1, points1)
    raydir2 = pixel_to_ray(K2, R2, points2)

    ray_check1, _ = check_rays(poly_verts, C1.T, raydir1)
    ray_check2, _ = check_rays(poly_verts, C2.T, raydir2)

    return ray_check1 | ray_check2


def get_depth(K, R, t, img_size, poly_verts):

    x, y = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))

    pixels = np.stack([x, y, np.ones_like(x)], axis=-1).reshape((-1, 3))

    rays = pixel_to_ray(K, R, pixels)

    C1 = -R.T @ t[:, None]

    q1 = C1.T

    depth = np.ones((pixels.shape[0]), dtype=float) * np.inf

    for i in range(poly_verts.shape[0]):

        p1 = poly_verts[i, 0, :3][None, ...]
        p2 = poly_verts[i, 1, :3][None, ...]
        p3 = poly_verts[i, 2, :3][None, ...]

        q2 = q1 + rays * 1000

        s1 = signed_volume(q1, p1, p2, p3).squeeze()
        s2 = signed_volume(q2, p1, p2, p3).squeeze()
        check1 = np.sign(s1) != np.sign(s2)
        s3 = signed_volume(q1, q2, p1, p2).squeeze()
        s4 = signed_volume(q1, q2, p2, p3).squeeze()
        s5 = signed_volume(q1, q2, p3, p1).squeeze()
        check2 = (np.sign(s3) == np.sign(s4)) & (np.sign(s5) == np.sign(s4)) & (np.sign(s3) == np.sign(s5))

        does_intersect = check1 & check2

        n = np.cross(p2 - p1, p3 - p1)
        v1 = p1-q1
        v2 = rays
        t = (np.dot(v1, n.T) / np.dot(v2, n.T))

        s = q1 + t * v2

        intersection = R[None, ...] @ (s - C1.T)[..., None]
        d = intersection[:, -1, 0]

        d[np.where(does_intersect == False)] = np.inf

        depth = np.minimum(d, depth)

    depth = depth.reshape(img_size)

    depth[np.where(depth > 1000)] = 1000

    return depth