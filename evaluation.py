import torch
import sklearn
import numpy as np
from utils import decode_landmarks


__all__ = ['evaluate_model']


def evaluate_model(model, data_loader, threshold, norm_by_face_height=False, use_sample_weight=False, pbar=None):
    all_errors = []
    device = next(model.parameters()).device
    if pbar is not None:
        pbar.reset(total=len(data_loader))
    for images, _, landmarks, face_corners, metadata in data_loader:
        images = images.to(device)
        landmarks = landmarks.to(device)
        face_corners = face_corners.to(device)
        heatmaps = model(images)[-1]
        predicted_landmarks = decode_landmarks(heatmaps)[0]
        predicted_landmarks[..., 0] *= images.shape[-1] / heatmaps.shape[-1]
        predicted_landmarks[..., 1] *= images.shape[-2] / heatmaps.shape[-2]
        errors = ((predicted_landmarks - landmarks).norm(dim=-1)).mean(dim=-1)
        if norm_by_face_height:
            # Normalise by Face height
            norm_factors = (face_corners[:, 2] - face_corners[:, 1]).norm(dim=-1)
        else:
            # Normalise by IOD
            norm_factors = (landmarks[:, 45] - landmarks[:, 36]).norm(dim=-1)
        norm_factors.clamp_min(torch.finfo(norm_factors.dtype).eps)
        normalised_errors = errors / norm_factors
        errors = normalised_errors.detach().cpu().numpy().tolist()
        if use_sample_weight:
            for error, weight in zip(errors, metadata['weight']):
                all_errors += [error] * max(1, int(weight))
        else:
            all_errors += errors
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    # Compute AUC
    auc_metric = 0.0
    sorted_errors = np.array([0] + sorted(all_errors))
    ticks = np.arange(0, len(all_errors) + 1) / max(1, len(all_errors))
    if threshold > 0:
        truncated_errors = sorted_errors[sorted_errors <= threshold]
        truncated_ticks = ticks[:len(truncated_errors)]
        if truncated_errors[-1] < threshold:
            if len(truncated_errors) < len(sorted_errors):
                next_err = sorted_errors[len(truncated_errors)]
                truncated_ticks = np.append(truncated_ticks,
                                            truncated_ticks[-1] + (threshold - truncated_errors[-1]) /
                                            (next_err - truncated_errors[-1]) / len(all_errors))
            else:
                truncated_ticks = np.append(truncated_ticks, truncated_ticks[-1])
            truncated_errors = np.append(truncated_errors, threshold)
        auc_metric = sklearn.metrics.auc(truncated_errors, truncated_ticks) / truncated_errors[-1]

    return auc_metric, sorted_errors, ticks
