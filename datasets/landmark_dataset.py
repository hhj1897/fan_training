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
    def __init__(self, tsv_path, partitions, config=None, random_flip=False,
                 geometric_transform=None, content_transform=None):
        self._samples = []
        self.weight_norm = 0.0
        tsv_folder = os.path.dirname(tsv_path)
        with open(tsv_path, 'r') as f:
            for line in f.read().splitlines()[1:]:
                fields = line.split('\t')
                subset, split = fields[:2]
                if (subset, split) in partitions:
                    weight = np.ceil(5 * 0.8 ** int(fields[2])) if int(fields[2]) >= 0 else 1.0
                    self.weight_norm += weight
                    im_path, pts_path = fields[3:5]
                    if not os.path.isabs(im_path):
                        im_path = os.path.abspath(os.path.join(tsv_folder, im_path))
                    if not os.path.isabs(pts_path):
                        pts_path = os.path.abspath(os.path.join(tsv_folder, pts_path))
                    face_box = np.array([float(x) for x in fields[5:9]]).reshape((-1, 2))
                    self._samples.append({'subset': subset, 'split': split, 'weight': weight,
                                          'im_path': im_path, 'pts_path': pts_path,
                                          'face_box': face_box})
        self.weight_norm /= len(self._samples)
        if config is None:
            self.config = LandmarkDataset.create_config()
        else:
            self.config = config
        self.random_flip = random_flip
        self.geometric_transform = geometric_transform
        self.content_transform = content_transform

    @staticmethod
    def create_config(image_size=256, heatmap_size=64, heatmap_gaussian_size=5, heatmap_gaussian_sigma=1,
                      heatmap_with_subpixel_sampling=True, crop_ratio=0.55, temp_padding_factor=1,
                      use_improved_preprocessing=True):
        return SimpleNamespace(image_size=image_size, heatmap_size=heatmap_size,
                               heatmap_gaussian_size=heatmap_gaussian_size,
                               heatmap_gaussian_sigma=heatmap_gaussian_sigma,
                               heatmap_with_subpixel_sampling=heatmap_with_subpixel_sampling,
                               crop_ratio=crop_ratio, temp_padding_factor=temp_padding_factor,
                               use_improved_preprocessing=use_improved_preprocessing)

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

        # Random flip needs to be handled manually
        im_hight, im_width = image.shape[:2]
        face_box = self._samples[item]['face_box']
        landmark_bbox = np.vstack((landmarks.min(axis=0), landmarks.max(axis=0)))
        if self.random_flip and random.random() < 0.5:
            landmarks = flip_landmarks(landmarks, im_width)
            face_box = np.array([[im_width - face_box[1, 0], face_box[0, 1]],
                                 [im_width - face_box[0, 0], face_box[1, 1]]])
            landmark_bbox = np.array([[im_width - landmark_bbox[1, 0], landmark_bbox[0, 1]],
                                      [im_width - landmark_bbox[0, 0], landmark_bbox[1, 1]]])
            image = cv2.flip(image, 1)
        face_corners = np.array([face_box[0], [face_box[1, 0], face_box[0, 1]],
                                 face_box[1], [face_box[0, 0], face_box[1, 1]]])
        landmark_bbox_corners = np.array([landmark_bbox[0], [landmark_bbox[1, 0], landmark_bbox[0, 1]],
                                          landmark_bbox[1], [landmark_bbox[0, 0], landmark_bbox[1, 1]]])

        face_centre = face_box.mean(axis=0)
        face_size = (face_box[1] - face_box[0]).mean()
        crop_box_size = face_size / self.config.crop_ratio
        keypoint_params = augs.KeypointParams(format='xy', remove_invisible=False)
        if self.config.use_improved_preprocessing:
            # Create the transform to crop out the face region with some margin added
            crop_box_tl = np.round(face_centre - crop_box_size / 2.0 * (1 + self.config.temp_padding_factor))
            crop_box_br = crop_box_tl + np.round(crop_box_size * (1 + self.config.temp_padding_factor)) + 1
            crop_margin = (int(-crop_box_tl[1]), int(crop_box_br[0] - im_width),
                           int(crop_box_br[1] - im_hight), int(-crop_box_tl[0]))
            crop_trans = augs.Compose(
                [augs.CropAndPad(crop_margin, pad_mode=cv2.BORDER_CONSTANT),
                 augs.Resize(int(np.round(self.config.image_size * (1 + self.config.temp_padding_factor))),
                             int(np.round(self.config.image_size * (1 + self.config.temp_padding_factor))))])

            # Add other geometric transforms
            all_transforms = [crop_trans]
            if self.geometric_transform is not None:
                all_transforms.append(self.geometric_transform)
            all_transforms.append(augs.CenterCrop(self.config.image_size, self.config.image_size))
            composite_trans = augs.Compose(all_transforms, keypoint_params=keypoint_params)

            # Transform the image and the landmarks
            trans_res = composite_trans(image=image, keypoints=np.vstack((landmarks, face_corners,
                                                                          landmark_bbox_corners)))
            image, warped_keypoints = trans_res['image'], trans_res['keypoints']
            landmarks = np.array(warped_keypoints[:landmarks.shape[0]])
            face_corners = np.array(warped_keypoints[-face_corners.shape[0] - landmark_bbox_corners.shape[0]:
                                                     -landmark_bbox_corners.shape[0]])
            landmark_bbox_corners = np.array(warped_keypoints[-landmark_bbox_corners.shape[0]:])

            # Create the label heatmaps
            heatmaps = encode_landmarks(landmarks / self.config.image_size * self.config.heatmap_size,
                                        self.config.heatmap_size, self.config.heatmap_size,
                                        self.config.heatmap_gaussian_size, self.config.heatmap_gaussian_sigma,
                                        self.config.heatmap_with_subpixel_sampling)
        else:
            # Crop the image
            crop_box_tl = np.round(face_centre - crop_box_size / 2.0)
            crop_box_br = crop_box_tl + np.round(crop_box_size) + 1
            crop_margin = (int(-crop_box_tl[1]), int(crop_box_br[0] - im_width),
                           int(crop_box_br[1] - im_hight), int(-crop_box_tl[0]))
            crop_trans = augs.Compose([augs.CropAndPad(crop_margin, pad_mode=cv2.BORDER_CONSTANT),
                                       augs.Resize(self.config.image_size, self.config.image_size)],
                                      keypoint_params=keypoint_params)
            crop_res = crop_trans(image=image, keypoints=np.vstack((landmarks, face_corners,
                                                                    landmark_bbox_corners)))
            image, warped_keypoints = crop_res['image'], crop_res['keypoints']
            landmarks = np.array(warped_keypoints[:landmarks.shape[0]])
            face_corners = np.array(warped_keypoints[-face_corners.shape[0] - landmark_bbox_corners.shape[0]:
                                                     -landmark_bbox_corners.shape[0]])
            landmark_bbox_corners = np.array(warped_keypoints[-landmark_bbox_corners.shape[0]:])

            # Create the label heatmaps
            heatmaps = encode_landmarks(landmarks / self.config.image_size * self.config.heatmap_size,
                                        self.config.heatmap_size, self.config.heatmap_size,
                                        self.config.heatmap_gaussian_size, self.config.heatmap_gaussian_sigma,
                                        self.config.heatmap_with_subpixel_sampling)

            if self.geometric_transform is not None:
                # Compose the geometric transform, also with some margin added
                padded_image_size = int(round(self.config.image_size * (1 + self.config.temp_padding_factor)))
                composite_trans = augs.ReplayCompose(
                    [augs.PadIfNeeded(padded_image_size, padded_image_size, border_mode=cv2.BORDER_CONSTANT),
                     self.geometric_transform, augs.CenterCrop(self.config.image_size, self.config.image_size)],
                    keypoint_params=keypoint_params)

                # Apply the composite geometric transform to the image and the landmarks
                trans_res = composite_trans(image=image, keypoints=np.vstack((landmarks, face_corners,
                                                                              landmark_bbox_corners)))
                image, warped_keypoints = trans_res['image'], trans_res['keypoints']
                landmarks = np.array(warped_keypoints[:landmarks.shape[0]])
                face_corners = np.array(warped_keypoints[-face_corners.shape[0] - landmark_bbox_corners.shape[0]:
                                                         -landmark_bbox_corners.shape[0]])
                landmark_bbox_corners = np.array(warped_keypoints[-landmark_bbox_corners.shape[0]:])

                # Apply the same transform to the heatmaps
                heatmaps = augs.resize(heatmaps.numpy().transpose(1, 2, 0),
                                       self.config.image_size, self.config.image_size)
                heatmaps = augs.resize(augs.ReplayCompose.replay(trans_res['replay'], image=heatmaps,
                                                                 keypoints=[])['image'],
                                       self.config.heatmap_size, self.config.heatmap_size)
                heatmaps = torch.from_numpy(heatmaps.transpose(2, 0, 1))

        # Apply the content transform to the image
        if self.content_transform is not None:
            image = self.content_transform(image=image)['image']

        # Image to tensor
        image = torch.from_numpy(image.astype(np.float32).transpose((2, 0, 1))).div_(255)

        return (image, heatmaps, torch.from_numpy(landmarks), torch.from_numpy(face_corners),
                torch.from_numpy(landmark_bbox_corners), self._samples[item])
