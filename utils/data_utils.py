import torch
import numpy as np
from copy import deepcopy


__all__ = ['load_pts', 'save_pts', 'get_iods', 'get_bbox_sizes', 'get_landmark_symmetry',
           'flip_landmarks', 'flip_heatmaps', 'encode_landmarks', 'decode_landmarks']


def load_pts(pts_path, one_based=True):
    with open(pts_path, 'r') as f:
        num_points = 0
        lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            fields = [x.strip() for x in line.split(':')]
            if len(fields) >= 2 and fields[0].lower() == 'n_points':
                num_points = int(fields[1])
                break
        if num_points > 0:
            return np.array(
                [float(x) for x in
                 ' '.join(lines[idx + 1:]).replace('\t', ' ').split('{')[1].split('}')[0].strip().split()],
                dtype=float).reshape((num_points, -1)) - (1 if one_based else 0)
        else:
            return np.array([])


def save_pts(pts_path, landmarks, one_based=True):
    with open(pts_path, 'w') as f:
        f.write(f'version: 1\n')
        f.write(f'n_points: {landmarks.shape[0]}\n{{\n')
        for landmark in landmarks:
            f.write(' '.join([f'{x + (1 if one_based else 0):.6f}' for x in landmark]) + '\n')
        f.write('}')


def get_iods(landmarks):
    return sum([(landmarks[..., 45, idx] - landmarks[..., 36, idx]) ** 2
                for idx in range(landmarks.shape[-1])]) ** 0.5


def get_bbox_sizes(bbox_corners):
    bbox_widths = sum([(bbox_corners[..., 1, idx] - bbox_corners[..., 0, idx]) ** 2
                       for idx in range(bbox_corners.shape[-1])]) ** 0.5
    bbox_heights = sum([(bbox_corners[..., 2, idx] - bbox_corners[..., 1, idx]) ** 2
                        for idx in range(bbox_corners.shape[-1])]) ** 0.5
    bbox_sizes = (bbox_widths * bbox_heights) ** 0.5
    return bbox_sizes, bbox_widths, bbox_heights


def get_landmark_symmetry():
    return ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (17, 26), (18, 25), (19, 24),
            (20, 23), (21, 22), (36, 45), (37, 44), (38, 43), (39, 42), (41, 46), (40, 47), (31, 35), (32, 34),
            (50, 52), (49, 53), (48, 54), (60, 64), (61, 63), (67, 65), (59, 55), (58, 56), (68, 75), (69, 74),
            (70, 73), (71, 72), (76, 82), (77, 81), (78, 80), (79, 83), (84, 86), (85, 87), (88, 89), (92, 95),
            (93, 94), (99, 96), (98, 97), (91, 90))


def flip_landmarks(landmarks, im_width, landmark_symmetry=get_landmark_symmetry()):
    if landmarks is torch.Tensor:
        result = landmarks.clone()
    else:
        result = deepcopy(landmarks)
    for cur_pair in landmark_symmetry:
        if cur_pair[0] < landmarks.shape[-2] and cur_pair[1] < landmarks.shape[-2]:
            result[..., cur_pair[0], :] = landmarks[..., cur_pair[1], :]
            result[..., cur_pair[1], :] = landmarks[..., cur_pair[0], :]
    result[..., 0] = im_width - result[..., 0]

    return result


def flip_heatmaps(heatmaps, landmark_symmetry=get_landmark_symmetry()):
    result = heatmaps.clone()
    for cur_pair in landmark_symmetry:
        if cur_pair[0] < heatmaps.shape[-3] and cur_pair[1] < heatmaps.shape[-3]:
            result[..., cur_pair[0], :, :] = heatmaps[..., cur_pair[1], :, :]
            result[..., cur_pair[1], :, :] = heatmaps[..., cur_pair[0], :, :]

    return result.flip(-1)


def encode_landmarks(landmarks, heatmap_width, heatmap_height, gaussian_size, sigma, use_subpixel_sampling=True):
    heatmaps = torch.zeros(landmarks.shape[0], heatmap_height, heatmap_width, dtype=torch.float32)
    for idx, pt in enumerate(landmarks):
        # Check that any part of the gaussian is in-bounds
        if use_subpixel_sampling:
            tl = np.round(pt - gaussian_size * sigma).astype(int)
            br = np.round(pt + gaussian_size * sigma).astype(int) + 1
        else:
            tl = (pt - gaussian_size * sigma).astype(int)
            br = (pt + gaussian_size * sigma).astype(int) + 1
        if tl[0] < heatmaps.shape[2] and tl[1] < heatmaps.shape[1] and br[0] > 0 and br[1] > 0:
            # Generate gaussian
            kernel_dim = br - tl
            if use_subpixel_sampling:
                x = torch.arange(0.5, kernel_dim[0], dtype=torch.float32)
                y = torch.arange(0.5, kernel_dim[1], dtype=torch.float32).unsqueeze(-1)
                x0, y0 = pt - tl
            else:
                x = torch.arange(0, kernel_dim[0], dtype=torch.float32)
                y = torch.arange(0, kernel_dim[1], dtype=torch.float32).unsqueeze(-1)
                x0, y0 = kernel_dim // 2
            # The gaussian is not normalized, we want the center value to be 1
            g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -tl[0]), min(br[0], heatmaps.shape[2]) - tl[0]
            g_y = max(0, -tl[1]), min(br[1], heatmaps.shape[1]) - tl[1]
            # Image range
            img_x = max(0, tl[0]), min(br[0], heatmaps.shape[2])
            img_y = max(0, tl[1]), min(br[1], heatmaps.shape[1])

            heatmaps[idx, img_y[0]: img_y[1], img_x[0]: img_x[1]] = g[g_y[0]: g_y[1], g_x[0]: g_x[1]]
    return heatmaps


def decode_landmarks(heatmaps, gamma=1.0, radius=0.1):
    if heatmaps.dim() == 3:
        landmarks, scores = decode_landmarks(heatmaps.unsqueeze(0))
        return landmarks[0], scores[0]

    heatmaps = heatmaps.contiguous()
    scores = heatmaps.max(dim=3)[0].max(dim=2)[0]

    if radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] < heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2:
        # Find peaks in all heatmaps
        m = heatmaps.view(heatmaps.shape[0] * heatmaps.shape[1], -1).argmax(1)
        all_peaks = torch.cat(
            [(m / heatmaps.shape[3]).trunc().view(-1, 1), (m % heatmaps.shape[3]).view(-1, 1)], dim=1
        ).reshape((heatmaps.shape[0], heatmaps.shape[1], 1, 1, 2)).repeat(
            1, 1, heatmaps.shape[2], heatmaps.shape[3], 1).float()

        # Apply masks created from the peaks
        all_indices = torch.zeros_like(all_peaks) + torch.stack(
            [torch.arange(0.0, all_peaks.shape[2],
                          device=all_peaks.device).unsqueeze(-1).repeat(1, all_peaks.shape[3]),
             torch.arange(0.0, all_peaks.shape[3],
                          device=all_peaks.device).unsqueeze(0).repeat(all_peaks.shape[2], 1)], dim=-1)
        heatmaps = heatmaps * ((all_indices - all_peaks).norm(dim=-1) <= radius *
                               (heatmaps.shape[2] * heatmaps.shape[3]) ** 0.5).float()

    # Prepare the indices for calculating centroids
    x_indices = (torch.zeros((*heatmaps.shape[:2], heatmaps.shape[3]), device=heatmaps.device) +
                 torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device))
    y_indices = (torch.zeros(heatmaps.shape[:3], device=heatmaps.device) +
                 torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device))

    # Finally, find centroids as landmark locations
    heatmaps = heatmaps.clamp_min(0.0)
    if gamma != 1.0:
        heatmaps = heatmaps.pow(gamma)
    m00s = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
    xs = heatmaps.sum(dim=2).mul(x_indices).sum(dim=2).div(m00s)
    ys = heatmaps.sum(dim=3).mul(y_indices).sum(dim=2).div(m00s)

    return torch.stack((xs, ys), dim=-1), scores
