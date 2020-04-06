import pandas as pd 
from torch.utils.data import Dataset
import os.path as osp
import numpy as np 
import torch 
import cv2

class HPatchesDataset(Dataset):
    """
    HPatches dataset (for evaluation)
    Args:
        csv_file: csv file with ground-truth data
        image_path_orig: filepath to the dataset (full resolution)
        transforms: image transformations (data preprocessing)
        image_size: size (tuple) of the output images
    Output:
        source_image: source image
        target_image: target image
        correspondence_map: pixel correspondence map
            between source and target views
        mask: valid/invalid correspondences
    """

    def __init__(self,
                 csv_file,
                 image_path_orig,
                 transforms,
                 image_size=(240, 240)):
        self.df = pd.read_csv(csv_file)
        self.image_path_orig = image_path_orig
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        obj = str(data.obj)
        im1_id, im2_id = str(data.im1), str(data.im2)
        h_scale, w_scale = self.image_size[0], self.image_size[1]

        h_ref_orig, w_ref_orig = data.Him.astype('int'), data.Wim.astype('int')
        h_trg_orig, w_trg_orig, _ = \
            cv2.imread(osp.join(self.image_path_orig,
                                obj,
                                im2_id + '.ppm'), -1).shape

        H = data[5:].astype('double').values.reshape((3, 3))

        '''
        As gt homography is calculated for (h_orig, w_orig) images,
        we need to
        map it to (h_scale, w_scale)
        H_scale = S * H * inv(S)
        '''
        S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                       [0, h_scale / h_ref_orig, 0],
                       [0, 0, 1]])
        S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                       [0, h_scale / h_trg_orig, 0],
                       [0, 0, 1]])

        H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))

        # inverse homography matrix
        Hinv = np.linalg.inv(H_scale)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        Xwarp = \
            (2 * XwarpHom / (ZwarpHom + 1e-8) / (w_scale - 1) - 1)
        Ywarp = \
            (2 * YwarpHom / (ZwarpHom + 1e-8) / (h_scale - 1) - 1)
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                               Ywarp.view(h_scale, w_scale)], dim=-1)

        # mask
        mask = grid_gt.ge(-1) & grid_gt.le(1)
        mask = mask[:, :, 0] & mask[:, :, 1]

        img1 = \
            cv2.resize(cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im1_id + '.ppm'), -1),
                       self.image_size)
        img2 = \
            cv2.resize(cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im2_id + '.ppm'), -1),
                       self.image_size)
        _, _, ch = img1.shape
        if ch == 3:
            img1_tmp = cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im1_id + '.ppm'), -1)
            img2_tmp = cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im2_id + '.ppm'), -1)
            img1 = cv2.cvtColor(cv2.resize(img1_tmp,
                                           self.image_size),
                                cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.resize(img2_tmp,
                                           self.image_size),
                                cv2.COLOR_BGR2RGB)

        # global transforms
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)

        return {'source_image': img1,
                'target_image': img2,
                'correspondence_map': grid_gt,
                'mask': mask.long()}
