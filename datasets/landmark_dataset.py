import os
import cv2
import torch
import random
import numpy as np
import albumentations as augs
from types import SimpleNamespace
from torch.utils.data import Dataset

from utils import load_pts, flip_landmarks, encode_landmarks


class LandmarkDataset(Dataset):
    def __init__(self, tsv_path, partitions, config=None, geometric_transform=None,
                 content_transform=None, random_flip=False):
        self._samples = []
        tsv_folder = os.path.dirname(tsv_path)
        total_weights = 0.0
        with open(tsv_path, 'r') as f:
            for line in f.read().splitlines()[1:]:
                fields = line.split('\t')
                subset, split = fields[:2]
                if (subset, split) in partitions:
                    weight = np.ceil(5 * 0.8 ** int(fields[2]))
                    total_weights += weight
                    im_path, pts_path = fields[3:5]
                    if not os.path.isabs(im_path):
                        im_path = os.path.abspath(os.path.join(tsv_folder, im_path))
                    if not os.path.isabs(pts_path):
                        pts_path = os.path.abspath(os.path.join(tsv_folder, pts_path))
                    face_box = np.array([float(x) for x in fields[5:9]]).reshape((-1, 2))
                    self._samples.append({'subset': subset, 'split': split, 'weight': weight,
                                          'im_path': im_path, 'pts_path': pts_path,
                                          'face_box': face_box})
        for sample in self._samples:
            sample['weight'] /= total_weights / len(self._samples)
        if config is None:
            self.config = LandmarkDataset.create_config()
        else:
            self.config = config
        self.geometric_transform = geometric_transform
        self.content_transform = content_transform
        self.random_flip = random_flip

    @staticmethod
    def create_config(image_size=256, heatmap_size=64, heatmap_gaussian_size=5, heatmap_gaussian_sigma=1,
                      crop_ratio=0.55, temp_padding_factor=1):
        return SimpleNamespace(image_size=image_size, heatmap_size=heatmap_size,
                               heatmap_gaussian_size=heatmap_gaussian_size,
                               heatmap_gaussian_sigma=heatmap_gaussian_sigma,
                               crop_ratio=crop_ratio, temp_padding_factor=temp_padding_factor)

    @staticmethod
    def get_partitions(dataset, split):
        if dataset == '300w':
            if split == 'train':
                return [('afw', ''), ('frgc', ''), ('helen', 'trainset'), ('lfpw', 'trainset'), ('xm2vts', '')]
            elif split == 'val':
                return [('helen', 'testset'), ('ibug', ''), ('lfpw', 'testset')]
            elif split == 'test':
                return [('300W', '01_Indoor'), ('300W', '01_Outdoor')]
            else:
                raise ValueError(f"{split} must be set to either train, val, or test.")
        elif dataset == '300w_lp':
            if split == 'train':
                return [('afw', ''), ('frgc', ''), ('helen', 'trainset'), ('lfpw', 'trainset'), ('xm2vts', '')]
            elif split == 'val':
                return [('helen', 'testset'), ('ibug', ''), ('lfpw', 'testset')]
            elif split == 'test':
                return [('300W', '01_Indoor'), ('300W', '01_Outdoor')]
            else:
                raise ValueError(f"{split} must be set to either train, val, or test.")
        else:
            raise ValueError(f"{dataset} must be set to either 300w or 300w_lp.")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        # Load the image and the landmarks
        image = cv2.cvtColor(cv2.imread(self._samples[item]['im_path']), cv2.COLOR_BGR2RGB)
        landmarks = load_pts(self._samples[item]['pts_path'])

        # Create the transform to properly crop out the face region
        im_hight, im_width = image.shape[:2]
        face_box = self._samples[item]['face_box']
        face_centre = face_box.mean(axis=0)
        face_size = (face_box[1] - face_box[0]).mean()
        crop_box_size = face_size / self.config.crop_ratio
        crop_box_tl = np.round(face_centre - crop_box_size / 2.0 * (1 + self.config.temp_padding_factor))
        crop_box_br = crop_box_tl + np.round(crop_box_size * (1 + self.config.temp_padding_factor)) + 1
        crop_margin = (int(-crop_box_tl[1]), int(crop_box_br[0] - im_width),
                       int(crop_box_br[1] - im_hight), int(-crop_box_tl[0]))
        crop_trans = augs.Compose(
            [augs.CropAndPad(crop_margin),
             augs.Resize(int(np.round(self.config.image_size * (1 + self.config.temp_padding_factor))),
                         int(np.round(self.config.image_size * (1 + self.config.temp_padding_factor))))])

        # Add other data augmentation transforms
        all_transforms = [crop_trans]
        if self.geometric_transform is not None:
            all_transforms.append(self.geometric_transform)
        all_transforms.append(augs.CenterCrop(self.config.image_size, self.config.image_size))
        if self.content_transform is not None:
            all_transforms.append(self.content_transform)
        composite_trans = augs.Compose(
            all_transforms, keypoint_params=augs.KeypointParams(format='xy', remove_invisible=False))

        # Transform the image and the landmarks
        trans_res = composite_trans(image=image, keypoints=landmarks)
        image, landmarks = trans_res['image'], np.array(trans_res['keypoints'])

        # Random flip needs to be handled manually
        if self.random_flip and random.random() < 0.5:
            landmarks = flip_landmarks(landmarks, image.shape[1])
            image = cv2.flip(image, 1)

        # Image to tensor
        image = torch.from_numpy(image.astype(np.float32).transpose((2, 0, 1))).div_(255.0)

        # Create the label heatmaps
        heatmaps = encode_landmarks(landmarks / self.config.image_size * self.config.heatmap_size,
                                    self.config.heatmap_size, self.config.heatmap_size,
                                    self.config.heatmap_gaussian_size, self.config.heatmap_gaussian_sigma)

        return image, heatmaps, landmarks, self._samples[item]
