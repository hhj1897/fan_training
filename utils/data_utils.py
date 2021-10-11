import numpy as np
from copy import deepcopy


__all__ = ['load_pts', 'save_pts', 'flip_landmarks', 'plot_landmarks']


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


def flip_landmarks(landmarks, im_width):
    pts_pairs = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                 [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45], [37, 44],
                 [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                 [49, 53], [48, 54], [60, 64], [61, 63], [67, 65], [59, 55], [58, 56])

    result = deepcopy(landmarks)
    for cur_pair in pts_pairs:
        result[cur_pair[0], :] = landmarks[cur_pair[1]]
        result[cur_pair[1], :] = landmarks[cur_pair[0]]
    result[:, 0] = im_width - result[:, 0]

    return result
