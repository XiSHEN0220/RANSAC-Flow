from getResults import *
from PIL import Image
import os
from matplotlib import pyplot as plt
from pathlib import Path
from utils_ply import write_ply
from tqdm import tqdm_notebook
from matplotlib import cm
cmap = cm.get_cmap("jet")
from matplotlib.colors import Normalize
import torch
import trimesh
import numpy as np
import pickle

def mesh_triangles(match, idx_mat):
    h, w = match.shape

    i,j = np.arange(h-1)[:, None], np.arange(1, w)[None, :]
    triangles_up = match[i, j-1] & match[i+1, j-1] & match[i, j]
    triangles_down = match[i+1, j] & match[i+1, j-1] & match[i, j]

    triangle_up_idx = np.stack([
        idx_mat[i, j - 1],
        idx_mat[i + 1, j - 1],
        idx_mat[i, j]
    ], axis=2)

    triangle_down_idx = np.stack([
        idx_mat[i + 1, j],
        idx_mat[i + 1, j - 1],
        idx_mat[i, j]
    ], axis=2)

    return np.concatenate(
       [triangle_up_idx[triangles_up], triangle_down_idx[triangles_down]],
        axis=0
    )


def compute_focal_lengths(fund, Issize, Itsize):
    F_tensor = torch.tensor(fund).float()
    
    f_t = torch.nn.Parameter(max(Issize) * torch.ones(1))
    f_s = torch.nn.Parameter(max(Itsize) * torch.ones(1))

    opt = torch.optim.SGD([f_s, f_t], lr=100, momentum=0.9)

    for lr in [1, .1]:
        opt.lr = lr
        for i in range(2000):
            K1 = torch.eye(3)
            K1[0, 0] = f_s
            K1[1, 1] = f_s

            K2 = torch.eye(3)
            K2[0, 0] = f_t
            K2[1, 1] = f_t

            E = K2.t() @ F_tensor @ K1
            u, s, v = torch.svd(E / torch.norm(E))
            l = s[0] - s[1]
            opt.zero_grad()
            l.backward()
            opt.step()

    return f_s.item(), f_t.item()

def get_rgba_im(depthmap):
    vmin, vmax = np.percentile(depthmap[depthmap > 0], [5, 95], )

    alphas = np.zeros(depthmap.shape)
    alphas[depthmap > 0] = 1.

    colors = Normalize(vmin, vmax, clip=True)(depthmap)
    colors = cmap(colors)

    colors[..., -1] = alphas
    return Image.fromarray((colors * 255).astype(np.uint8))


def compute_3D_pos(n_pts1, n_pts2, R, t):

    P1 = np.concatenate((
        np.eye(3),
        np.zeros((3, 1))
    ), axis=1)  # 3 * 4

    P2 = np.concatenate((R, t), axis=1) # 3 * 4

    # build A matrix of shape N * 4 * 4
    A = np.stack((
        n_pts1[:, 0:1] * P1[2, :] - P1[0, :],
        n_pts1[:, 1:2] * P1[2, :] - P1[1, :],
        n_pts2[:, 0:1] * P2[2, :] - P2[0, :],
        n_pts2[:, 1:2] * P2[2, :] - P2[1, :]
    ), axis=1)

    _, _, V = np.linalg.svd(A)

    X = V[:, -1, :]  # N * 4

    coord = X[:, :3] / X[:, 3:4]
    
    return coord

def get_point_cloud_color(source, target, pts1, pts2, by_path=True):
    if by_path:
        imA = Image.open(os.path.join(scene_path, images_name[source]))
        imB = Image.open(os.path.join(scene_path, images_name[target]))
        imA = ResizeMinResolution(minSize, imA, strideNet)
        imB = ResizeMinResolution(minSize, imB, strideNet)
    else:
        imA = source
        imB = target

    npA = np.array(imA).astype(int)
    npB = np.array(imB).astype(int)
    cA = npA[np.clip(pts1[:, 1].astype(int), 0, npA.shape[0] - 1),
             np.clip(pts1[:, 0].astype(int), 0, npA.shape[1] - 1)]
    cB = npB[np.clip(pts2[:, 1].astype(int), 0, npB.shape[0] - 1),
             np.clip(pts2[:, 0].astype(int), 0, npB.shape[1] - 1)]

    return (cA + cB) / 2
    
def save_ply(point_cloud, name, use_mask=True):
    coordinates, colors, med_tri_angle, mask = point_cloud
    
    if not use_mask:
        mask = np.ones((coordinates.shape[0], 1), np.bool)

    pos_mask = mask.squeeze(1).astype(np.bool) & (coordinates[:, 2] > 0)
    print(pos_mask.shape)
    write_ply(name, (coordinates[pos_mask, :], colors[pos_mask]), field_names)
    
def compute_triangulation_angle(point_cloud, R, t):
    ray1 = point_cloud
    ray2 = point_cloud + (R.T @ t).T
        
    cos = np.sum(ray1 * ray2, axis=1) / np.linalg.norm(ray1, axis=1) / np.linalg.norm(ray2, axis=1)
    return np.arccos(cos) / np.pi * 180
    
def reproject_point_cloud(point_cloud, K, R=None, t=None):
    if R is None:
        R = np.eye(3)
        t = np.zeros((3, 1))
        
    on_im_plane = (point_cloud @ R.T + t.T)    
    in_pixel = on_im_plane @ K.T
    return in_pixel[:, :2] / in_pixel[:, 2:], on_im_plane[:, 2]

def compute_Kmatrix(idx, K_list, org_imsizes, resized_shapes):
    ka = K_list[idx]

    ka[0, 2] = org_imsizes[idx][0] / 2
    ka[1, 2] = org_imsizes[idx][1] / 2

    rescale1 = np.diag([
        resized_shapes[idx][0] / org_imsizes[idx][0],
        resized_shapes[idx][1] / org_imsizes[idx][1],
        1.
    ])
    
    return rescale1 @ ka

def compute_and_save(flow, match, Is, It, sizeA, sizeB, savepath, thresholds, suffix=""):
    fundRansacThresh = thresholds["fundamental_ransac"]
    essRansacThresh = thresholds["essential_ransac"]
    Isw, Ish = sizeA
    Itw, Ith = sizeB

    pts1, pts2, idx_mat = matches_from_flow(flow,
                                            match,
                                            (Isw, Ish),
                                            (Itw, Ith),
                                            angle=0)

    resF = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=fundRansacThresh, method=cv2.FM_RANSAC)

    f_s, f_t = compute_focal_lengths(resF[0], (Isw, Ish), (Itw, Ith))
    norm_pts1 = pts1 / f_s
    norm_pts2 = pts2 / f_t

    (R, t), mask = opencv_decompose(norm_pts1, norm_pts2, True, essRansacThresh)
    coords = compute_3D_pos(norm_pts1, norm_pts2, R, t)

    est_pts1, depth1 = reproject_point_cloud(coords, np.eye(3))
    est_pts2, depth2 = reproject_point_cloud(coords, np.eye(3), R, t)

    int_pts1 = np.rint(pts1).astype(int)
    int_pts2 = np.rint(pts2).astype(int)

    depthmap1 = -1 * np.ones((Ish, Isw))
    depthmap2 = -1 * np.ones((Ith, Itw))

    depthmap1[np.clip(int_pts1[:, 1], 0, Ish - 1), np.clip(int_pts1[:, 0], 0, Isw - 1)] = depth1
    depthmap2[np.clip(int_pts2[:, 1], 0, Ith - 1), np.clip(int_pts2[:, 0], 0, Itw - 1)] = depth2

    imA = get_rgba_im(depthmap1)
    imB = get_rgba_im(depthmap2)

    imA.save(savepath / "depthmapSource_{}.png".format(suffix))
    imB.save(savepath / "depthmapTarget_{}.png".format(suffix))

    triangles = mesh_triangles(match, idx_mat)

    colors = get_point_cloud_color(Is.resize((Isw, Ish)), It.resize((Itw, Ith)), pts1, pts2, by_path=False)
    tri_colors = colors[triangles[:, 1]]

    mesh = trimesh.Trimesh(
        vertices= coords,
        faces = triangles,
        face_colors=tri_colors,
        vertex_colors=colors
    )

    max_depth= np.percentile(coords[:, 2], 99)
    min_depth = np.percentile(coords[:, 2], 1)
    mesh.update_vertices(mesh.vertices[:, 2] > min_depth)
    mesh.update_vertices(mesh.vertices[:, 2] < max_depth)
    mesh.update_faces(mesh.area_faces < thresholds["max_triangle_size"])
    mesh.export(savepath / "mesh_{}.ply".format(suffix))


if __name__ == "__main__":
    minSize = 480
    strideNet = 16
    stop = 5
    matchability = 0.99
    scene = "sacre_coeur"
    multiH = True
    ransac=True
    threshold = 0.0005
    match_threshold = 0.95
    gtPath = '../../data/YFCC/images/'
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']

    res = dict()

    for scene in ["sacre_coeur", "notre_dame_front_facade", "reichstag", "buckingham_palace"]:
        coarsePath = Path("/media/hdd/Down8_CoarsePlus") / scene
        finePath = Path("/media/hdd/Down8_FinePlus/") / scene
        maskPath = finePath


        flowList = os.listdir(finePath)
        testPair = '../../data/YFCC/pairs'
        res[scene] = []
        with open(os.path.join(testPair, scene + '-te-1000-pairs.pkl'), 'rb') as f :
            pairs_ids = pickle.load(f)


        scene_path = os.path.join(gtPath, scene, 'test') 

        with open(os.path.join(scene_path, "images.txt")) as f:
            images_name = [tmp.strip() for tmp in f.readlines()]

        with open(os.path.join(scene_path, "calibration.txt")) as f:
            calib_name = [tmp.strip() for tmp in f.readlines()]

        r_list = list()
        t_list = list()
        geoms = list()
        resized_shapes = list()
        org_imsizes = list()
        K_list = list()


        # Read image infos
        for im, calib in zip(images_name, calib_name):

            calib_h5 = h5py.File(os.path.join(scene_path, calib))      
            r_list.append(np.array(calib_h5["R"]))
            t_list.append(np.array(calib_h5["T"]).T)
            geoms.append(calib_h5)
            org_imsizes.append(np.array(calib_h5['imsize'][0]).tolist())

            K_list.append(np.array(calib_h5['K']))

            resized_shapes.append(getResizedSize(minSize, org_imsizes[-1], strideNet))

        for i in tqdm(range(len(pairs_ids))):
            idA, idB = pairs_ids[i]
            ## read flow and matchability

            flow, match, maskBG = getFlow(i, finePath, flowList, coarsePath, maskPath, multiH, match_threshold)
            if len(flow) == 0 : 
                res[scene].append(None)
                continue

            # compute relative pose
            r = r_list[idB] @ r_list[idA].T
            t = t_list[idB] - r @ t_list[idA]

            pts1, pts2 = matches_from_flow(flow, match, resized_shapes[idA], resized_shapes[idB], matchability)
            if len(pts1) == 0 :
                res[scene].append(None)
                continue 

            norm_pts1 = norm_kp(org_imsizes[idA], resized_shapes[idA], K_list[idA], pts1)
            norm_pts2 = norm_kp(org_imsizes[idB], resized_shapes[idB], K_list[idB], pts2)

            decomposed, mask = opencv_decompose(norm_pts1, norm_pts2, ransac, threshold)

            if decomposed is None:
                res[scene].append(None)

            else:
                err_r, err_t = evaluate_R_t(r, t, decomposed[0], decomposed[1])

                coords = compute_3D_pos(norm_pts1, norm_pts2, decomposed[0], decomposed[1])
                res[scene].append({
                    "point_cloud": coords,
                    "color": get_point_cloud_color(idA, idB, pts1, pts2),
                    "triangulation_angle": compute_triangulation_angle(coords, decomposed[0], decomposed[1]),
                    "im1_coordinate": pts1,
                    "im2_coordinate": pts2,
                    "inlier_mask": mask,
                    "background_mask": maskBG,
                    "pred_r": decomposed[0],
                    "pred_t": decomposed[1],
                    "error_r": err_r,
                    "error_t": err_t
                })
                

    with open("results.pkl", "wb") as f:    
        pickle.dump(res,f)
