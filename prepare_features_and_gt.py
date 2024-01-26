import os.path
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.sparse
import shutil
import utils
import pickle
import argparse

parser = argparse.ArgumentParser(
        description='Synthetic Metropolis Homographies: prepare SIFT features and ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', default="./render", help='path to folder with rendered images')
parser.add_argument('--output', default="./dataset", help='path to folder for storing the processed dataset')
parser.add_argument('--path_id', default=-1, type=int, help="only process a specific coarse trajectory")
parser.add_argument('--fine_id', default=-1, type=int, help="only process a specific fine trajectory")
parser.add_argument('--save_depth', action='store_true', help='generate depth maps for each image pair')
parser.add_argument('--save_plots', action='store_true', help='save plots with feature visualisations for each image pair')

opt = parser.parse_args()

threshold = 1.0
min_points = 10

coarse_paths_file = os.path.join(opt.input, "coarse_paths_processed.pkl")
if os.path.exists(coarse_paths_file):
    with open(coarse_paths_file, 'rb') as f:
        coarse_paths = pickle.load(f)
else:
    all_image_files = sorted(glob.glob(os.path.join(opt.input, "render_*.png")))
    split_files = [f.split("_") for f in all_image_files]

    orientation_ids = list(set([int(f[1]) for f in split_files]))
    focal_lengths = list(set([int(f[2]) for f in split_files]))
    step_sizes = list(set([int(f[3]) for f in split_files]))
    coarse_path_ids = list(set([int(f[4]) for f in split_files]))

    coarse_paths = []
    for coarse_path_id in coarse_path_ids:
        coarse_path_frames = [f for f in split_files if int(f[4]) == coarse_path_id]
        fine_paths = []
        fine_path_ids = list(set([int(f[5]) for f in split_files if int(f[4]) == coarse_path_id]))
        for fine_path_id in fine_path_ids:
            all_pairs = []
            fine_path_frames = [f for f in coarse_path_frames if int(f[5]) == fine_path_id]
            for orientation_id in [1, 2]:
                for focal_length in focal_lengths:
                    for step_size in step_sizes:
                        frames = [f for f in fine_path_frames if
                                  int(f[1]) == orientation_id and
                                  int(f[2]) == focal_length and
                                  int(f[3]) == step_size]
                        frames.sort()
                        pairs = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]
                        all_pairs += pairs
            fine_paths += [all_pairs]
        coarse_paths += [fine_paths]

    with open(coarse_paths_file, 'wb') as f:
        pickle.dump(coarse_paths, f, pickle.HIGHEST_PROTOCOL)

obj_data = np.load(os.path.join(opt.input, "polygons.npz"), allow_pickle=True)

poly_centers = obj_data["centers"]
poly_normals = obj_data["normals"]
poly_verts_orig = obj_data["vertices"]
poly_planes_orig = []
vertices = []
for idx in range(poly_centers.shape[0]):
    c = poly_centers[idx]
    n = poly_normals[idx]
    n /= np.linalg.norm(n)
    d = -np.sum(c * n)
    sign = np.sign(d)
    if sign == 0:
        sign = 1
    n, d = n * sign, d * sign
    poly_planes_orig += [np.array([n[0], n[1], n[2], d])]

poly_planes_orig = np.stack(poly_planes_orig, axis=0)

if poly_verts_orig.shape[-1] == 3:
    poly_verts_orig = np.concatenate([poly_verts_orig, np.ones((poly_verts_orig.shape[0], poly_verts_orig.shape[1], 1))], axis=-1)

max_num_planes = 0

for coarse_id, coarse_path in enumerate(coarse_paths):

    if opt.path_id >= 0:
        if not (coarse_id == opt.path_id):
            continue

    for fine_id, image_pairs in enumerate(coarse_path):

        if opt.fine_id >= 0:
            if not (fine_id == opt.fine_id):
                continue

        processed_index = 0

        for pair_idx, image_pair in enumerate(image_pairs):

            target_folder = os.path.join(opt.output, "%d/%02d/%04d" % (coarse_id, fine_id, pair_idx))

            if os.path.exists(os.path.join(target_folder, "features_and_ground_truth.npz")):
                continue

            print(target_folder)

            img1_path = "_".join(image_pair[0])
            img2_path = "_".join(image_pair[1])
            cam1_path = img1_path.replace("render_", "camera_").replace(".png", ".npz")
            cam2_path = img2_path.replace("render_", "camera_").replace(".png", ".npz")

            cam_datas = [np.load(f, allow_pickle=True) for f in [cam1_path, cam2_path]]

            poly_ids1 = cam_datas[0]["polygons"]
            poly_ids2 = cam_datas[1]["polygons"]
            poly_ids = np.array(list(set(poly_ids1.tolist()+poly_ids2.tolist())))

            if poly_ids1.size == 0:
                continue
            if poly_ids2.size == 0:
                continue

            poly_verts1 = poly_verts_orig[poly_ids1]
            poly_verts2 = poly_verts_orig[poly_ids2]
            poly_verts = poly_verts_orig[poly_ids]
            poly_planes = poly_planes_orig[poly_ids]

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()

            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            bf = cv2.BFMatcher()

            try:
                matches = bf.knnMatch(des1, des2, k=2)
            except Exception as e:
                print(e)
                continue

            good = []
            good_points_1 = []
            good_points_2 = []
            ratios = []
            for match in matches:
                if len(match) < 2:
                    continue
                m, n = match
                if m.distance < 0.75*n.distance:
                    good.append([m])
                    idx1 = m.queryIdx
                    idx2 = m.trainIdx
                    p1 = kp1[idx1].pt
                    p2 = kp2[idx2].pt
                    p1 = np.array(list(p1) + [1])
                    p2 = np.array(list(p2) + [1])
                    good_points_1 += [p1]
                    good_points_2 += [p2]
                    ratios += [m.distance / n.distance]

            if len(good_points_1) == 0 or len(good_points_2) == 0:
                continue

            good_points_1 = np.stack(good_points_1, axis=1)
            good_points_2 = np.stack(good_points_2, axis=1)
            ratios = np.stack(ratios, axis=0)

            F_mats = []
            residuals = []
            points1 = good_points_1.T
            points2 = good_points_2.T

            K1 = cam_datas[0]["K"]
            K2 = cam_datas[1]["K"]
            R1 = cam_datas[0]["R"].T
            t1 = cam_datas[0]["t"]
            R2 = cam_datas[1]["R"].T
            t2 = cam_datas[1]["t"]

            R = R2 @ R1.T
            t = t2 - R2 @ R1.T @ t1

            M1 = np.zeros((4,4))
            M1[:3,:3] = R1
            M1[:3, -1] = t1
            M1[-1, -1] = 1

            M2 = np.zeros((4,4))
            M2[:3,:3] = R2
            M2[:3, -1] = t2
            M2[-1, -1] = 1

            M1i = np.linalg.inv(M1)

            P1 = K1 @ np.eye(3, 4) @ M1
            P2 = K2 @ np.eye(3, 4) @ M2

            verts1 = (P1 @ poly_verts1[..., None])[..., 0]
            verts2 = (P2 @ poly_verts2[..., None])[..., 0]
            verts1 /= verts1[..., -1][..., None]
            verts2 /= verts2[..., -1][..., None]

            planes = poly_planes @ M1i
            planes = planes / np.linalg.norm(planes[:, :3], axis=-1, keepdims=True)

            n = planes[:, :3]
            d = planes[:, 3]
            Hs = R[None, ...] - (t[None, :, None] @ n[:, None]) / d[:, None, None]
            Hs = K2[None] @ Hs @ np.linalg.inv(K1)[None]

            correspondences = np.concatenate([points1[:, :2], points2[:, :2]], axis=-1)
            hom_res = utils.homography_distance(correspondences, Hs)

            hom_check = hom_res < threshold

            ray_check = utils.ray_cast(K1, K2, R1, R2, t1, t2, poly_verts, points1, points2)

            poly_check = hom_check & ray_check

            inliers = np.sum(poly_check, axis=0) > 0
            inlier_counts = np.sum(poly_check, axis=-1)

            valid_polys = np.nonzero(inlier_counts)[0]

            planes = planes[valid_polys]
            Hs = Hs[valid_polys]
            hom_res = hom_res[valid_polys]
            poly_check = poly_check[valid_polys]
            inlier_counts = inlier_counts[valid_polys]

            normal_sim = np.arccos(np.clip(np.abs(planes[:, :3] @ planes[:, :3].T), a_min=1e-16, a_max=1-(1e-16))) * 180 / np.pi
            distance_sim = np.abs(planes[:, -1][:, None] - planes[:, -1][None])
            plane_sim = (normal_sim < 2) & (distance_sim < 0.1)

            num_clusters, cluster_assignments = scipy.sparse.csgraph.connected_components(plane_sim, directed=False)
            clusters = []
            clusters_inliers = []
            clusters_residuals = []
            clusters_planes = []
            for ci in range(num_clusters):
                cluster = np.nonzero(cluster_assignments == ci)[0]
                cluster_inliers = np.max(poly_check[cluster], axis=0)
                cluster_inlier_count = sum(cluster_inliers)

                if cluster_inlier_count > min_points:
                    clusters += [cluster]
                    clusters_inliers += [cluster_inliers]
                    clusters_residuals += [np.min(hom_res[cluster], axis=0)]
                    clusters_planes += [planes[cluster]]
                pass

            if len(clusters_inliers) == 0:
                continue

            clusters_inliers = np.stack(clusters_inliers, axis=0)
            clusters_residuals = np.stack(clusters_residuals, axis=0) + (1-clusters_inliers) * 1e9
            total_inliers = np.max(clusters_inliers, axis=0)

            converged = False
            old_indices = None
            while not converged:

                assignments = np.argmin(clusters_residuals, axis=0)
                assignments = ((assignments+1)*total_inliers).astype(int) - 1

                indices, counts = np.unique(assignments, return_counts=True)
                indices, counts = indices[1:], counts[1:]

                tokeep = np.nonzero(counts >= min_points)

                indices = indices[tokeep]

                planes = planes[indices]
                clusters_inliers = clusters_inliers[indices]
                clusters_residuals = clusters_residuals[indices]
                if clusters_inliers.size == 0:
                    break

                total_inliers = np.max(clusters_inliers, axis=0)

                if old_indices is not None:
                    if np.all(indices == old_indices):
                        converged = True
                old_indices = indices

            if clusters_inliers.size == 0:
                continue
            if clusters_residuals.size == 0:
                continue

            converged = False
            while not converged:
                converged = True

                assignments = np.argmin(clusters_residuals, axis=0)
                assignments = ((assignments + 1) * total_inliers).astype(int) - 1

                for ci in range(np.max(assignments)+1):

                    sel1 = points1[np.nonzero(assignments==ci)]
                    sel2 = points2[np.nonzero(assignments==ci)]

                    u, s, vh = np.linalg.svd(sel1)
                    l1 = vh[-1]
                    l1 /= np.linalg.norm(l1[:2])
                    u, s, vh = np.linalg.svd(sel2)
                    l2 = vh[-1]
                    l2 /= np.linalg.norm(l2[:2])

                    res1 = np.abs(sel1 @ l1[:, None])
                    res2 = np.abs(sel2 @ l2[:, None])

                    if np.mean(res1) < 5 or np.mean(res2) < 5:
                        converged = False

                        planes = np.delete(planes, ci, axis=0)
                        clusters_inliers = np.delete(clusters_inliers, ci, axis=0)
                        clusters_residuals = np.delete(clusters_residuals, ci, axis=0)
                        if clusters_inliers.size == 0:
                            break
                        total_inliers = np.max(clusters_inliers, axis=0)

                        break

                if clusters_residuals.size == 0:
                    break

            if clusters_inliers.size == 0:
                continue
            if clusters_residuals.size == 0:
                continue

            indices, counts = np.unique(assignments, return_counts=True)
            indices, counts = indices[1:], counts[1:]
            sorting = np.argsort(counts)[::-1]
            counts = counts[sorting]
            planes = planes[sorting]
            clusters_inliers = clusters_inliers[sorting]
            clusters_residuals = clusters_residuals[sorting]
            assignments = np.argmin(clusters_residuals, axis=0)
            assignments = ((assignments + 1) * total_inliers).astype(int) - 1

            num_planes = clusters_inliers.shape[0]
            max_num_planes = max(num_planes, max_num_planes)

            os.makedirs(target_folder, exist_ok=True)

            if opt.save_depth:
                depth1 = utils.get_depth(K1, R1, t1, (1024, 1024), poly_verts)
                depth2 = utils.get_depth(K2, R2, t2, (1024, 1024), poly_verts)

                depth1_path = os.path.join(target_folder, "depth1.png")
                depth2_path = os.path.join(target_folder, "depth2.png")
                depth1_uint16 = np.clip(depth1 * (65535.0 / 1000.0), a_max=65535, a_min=0).astype(np.uint16)
                depth2_uint16 = np.clip(depth2 * (65535.0 / 1000.0), a_max=65535, a_min=0).astype(np.uint16)
                cv2.imwrite(depth1_path, depth1_uint16)
                cv2.imwrite(depth2_path, depth2_uint16)

            np.savez(os.path.join(target_folder, "features_and_ground_truth.npz"),
                     labels=assignments+1, points1=points1, points2=points2, ratios=ratios, planes=planes,
                     K1=K1, K2=K2, R=R, t=t)

            shutil.copyfile(img1_path, os.path.join(target_folder, 'render0.png'))
            shutil.copyfile(img2_path, os.path.join(target_folder, 'render1.png'))
            shutil.copyfile(cam1_path, os.path.join(target_folder, 'camera0.npz'))
            shutil.copyfile(cam2_path, os.path.join(target_folder, 'camera1.npz'))

            processed_index += 1

            if opt.save_plots:
                num_plots = num_planes + 1
                fig, axs = plt.subplots(5, 5)
                plt.tight_layout()
                axs = [ax for x in axs for ax in x]
                for i in range(5*5):
                    axs[i].imshow(gray1, cmap="Greys_r")
                    idxs = np.nonzero(assignments == i)
                    axs[i].scatter(points1[idxs, 0], points1[idxs, 1], c="#1E88E5", s=0.1)
                    axs[i].set_axis_off()
                axs[-1].imshow(gray1, cmap="Greys_r")
                idxs = np.nonzero(assignments == -1)
                axs[-1].scatter(points1[idxs, 0], points1[idxs, 1], c="#D81B60", s=0.1)
                for ax in axs:
                    ax.set_axis_off()
                plt.tight_layout()

                plt.savefig(os.path.join(target_folder, "vis.jpg"), dpi=600)
                plt.close(fig)

                fig, axs = plt.subplots(1, 2)
                plt.tight_layout()
                axs[0].imshow(depth1, vmin=0, vmax=100, cmap="cividis")
                axs[1].imshow(depth2, vmin=0, vmax=100, cmap="cividis")
                for ax in axs:
                    ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(os.path.join(target_folder, "vis_depth.jpg"), dpi=300)
                plt.close(fig)

print("%d pairs, %d max. planes" % (processed_index, max_num_planes))