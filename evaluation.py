import torch
import sklearn
import numpy as np
from utils import flip_heatmaps, decode_landmarks
from losses import heatmap_mse_loss, landmark_distance_loss


__all__ = ['run_model', 'compute_landmark_errors', 'compute_auc', 'compute_accuracy']


@torch.no_grad()
def run_model(model, data_loader, pbar=None, gamma=1.0, radius=0.1):
    results = {'heatmap_mse_losses': np.array([]), 'landmark_distance_losses': np.array([]),
               'heatmap_errors': np.array([]), 'predicted_landmarks': np.array([]), 'landmark_scores': np.array([]),
               'landmarks': np.array([]), 'face_corners': np.array([]), 'landmark_bbox_corners': np.array([]),
               'sample_weights': np.array([])}
    if pbar is not None:
        pbar.reset(total=len(data_loader))
        pbar.refresh()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for images, heatmaps, landmarks, face_corners, landmark_bbox_corners, metadata in data_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        landmarks = landmarks.to(device)

        # Predict heatmaps
        all_predicted_heatmaps = model(torch.cat((images, images.flip(-1))))
        predicted_heatmaps = [htm[:images.shape[0]] for htm in all_predicted_heatmaps]
        predicted_heatmaps2 = [flip_heatmaps(htm[images.shape[0]:]) for htm in all_predicted_heatmaps]

        # Compute heatmap mse loss (without reduction)
        htm_mse_losses = torch.stack(
            (heatmap_mse_loss(predicted_heatmaps, heatmaps, reduce=False),
             heatmap_mse_loss(predicted_heatmaps2, heatmaps, reduce=False)),
            dim=1).detach().cpu().numpy()
        if results['heatmap_mse_losses'].shape[0] > 0:
            results['heatmap_mse_losses'] = np.concatenate((results['heatmap_mse_losses'], htm_mse_losses))
        else:
            results['heatmap_mse_losses'] = htm_mse_losses

        # Compute landmark distance loss (without reduction)
        lmk_dist_losses = torch.stack(
            (landmark_distance_loss(predicted_heatmaps, landmarks, images.shape, reduce=None),
             landmark_distance_loss(predicted_heatmaps2, landmarks, images.shape, reduce=None)),
            dim=1).detach().cpu().numpy()
        if results['landmark_distance_losses'].shape[0] > 0:
            results['landmark_distance_losses'] = np.concatenate(
                (results['landmark_distance_losses'], lmk_dist_losses))
        else:
            results['landmark_distance_losses'] = lmk_dist_losses

        # Compute heatmap errors
        num_landmarks = min(predicted_heatmaps[-1].shape[-3], heatmaps.shape[-3])
        num_landmarks2 = min(predicted_heatmaps2[-1].shape[-3], heatmaps.shape[-3])
        htm_errors = torch.stack(
            (((predicted_heatmaps[-1][..., :num_landmarks, :, :] -
               heatmaps[..., :num_landmarks, :, :]) ** 2).mean(dim=(-1, -2)),
             ((predicted_heatmaps2[-1][..., :num_landmarks2, :, :] -
               heatmaps[..., :num_landmarks2, :, :]) ** 2).mean(dim=(-1, -2))),
            dim=1).detach().cpu().numpy()
        if results['heatmap_errors'].shape[0] > 0:
            results['heatmap_errors'] = np.concatenate((results['heatmap_errors'], htm_errors))
        else:
            results['heatmap_errors'] = htm_errors

        # Predict landmarks
        lmk, scores = decode_landmarks(predicted_heatmaps[-1], gamma=gamma, radius=radius)
        lmk2, scores2 = decode_landmarks(predicted_heatmaps2[-1], gamma=gamma, radius=radius)
        predicted_landmarks = torch.stack((lmk, lmk2), dim=1)
        predicted_landmarks[..., 0] *= images.shape[-1] / predicted_heatmaps[-1].shape[-1]
        predicted_landmarks[..., 1] *= images.shape[-2] / predicted_heatmaps[-1].shape[-2]
        predicted_landmarks = predicted_landmarks.detach().cpu().numpy()
        if results['predicted_landmarks'].shape[0] > 0:
            results['predicted_landmarks'] = np.concatenate((results['predicted_landmarks'], predicted_landmarks))
        else:
            results['predicted_landmarks'] = predicted_landmarks
        landmark_scores = torch.stack((scores, scores2), dim=1).detach().cpu().numpy()
        if results['landmark_scores'].shape[0] > 0:
            results['landmark_scores'] = np.concatenate((results['landmark_scores'], landmark_scores))
        else:
            results['landmark_scores'] = landmark_scores

        # Record landmarks
        if results['landmarks'].shape[0] > 0:
            results['landmarks'] = np.concatenate((results['landmarks'], landmarks.detach().cpu().numpy()))
        else:
            results['landmarks'] = landmarks.detach().cpu().numpy()

        # Record face corners
        if results['face_corners'].shape[0] > 0:
            results['face_corners'] = np.concatenate((results['face_corners'], face_corners.detach().cpu().numpy()))
        else:
            results['face_corners'] = face_corners.detach().cpu().numpy()

        # Record landmark bounding box corners
        if results['landmark_bbox_corners'].shape[0] > 0:
            results['landmark_bbox_corners'] = np.concatenate(
                (results['landmark_bbox_corners'], landmark_bbox_corners.detach().cpu().numpy()))
        else:
            results['landmark_bbox_corners'] = landmark_bbox_corners.detach().cpu().numpy()

        # Record sample weights
        results['sample_weights'] = np.concatenate(
            (results['sample_weights'], metadata['weight'].detach().cpu().numpy()))

        if pbar is not None:
            pbar.update(1)
            pbar.refresh()

    if was_training:
        model.train()
    return results


def compute_landmark_errors(predicted_landmarks, landmarks):
    if landmarks.ndim < predicted_landmarks.ndim:
        landmarks = np.expand_dims(landmarks, tuple(range(1, 1 + predicted_landmarks.ndim - landmarks.ndim)))
    num_landmarks = min(predicted_landmarks.shape[-2], landmarks.shape[-2])
    return np.linalg.norm(predicted_landmarks[..., :num_landmarks, :] - landmarks[..., :num_landmarks, :], axis=-1)


def compute_auc(landmark_errors, threshold, normalisation_factors=None, sample_weights=None, reduction_axis=-1):
    if landmark_errors.ndim > 1 and reduction_axis is not None:
        landmark_errors = landmark_errors.mean(axis=reduction_axis)
    if normalisation_factors is not None:
        normalisation_factors = np.clip(normalisation_factors, np.finfo(landmark_errors.dtype).eps, None)
        if normalisation_factors.ndim < landmark_errors.ndim:
            normalisation_factors = np.expand_dims(normalisation_factors,
                                                   tuple(range(normalisation_factors.ndim, landmark_errors.ndim)))
        landmark_errors = landmark_errors / normalisation_factors
    if sample_weights is not None:
        resampled_errors = []
        for error, weight in zip(landmark_errors, sample_weights):
            if error.ndim > 0:
                resampled_errors += error.flatten().tolist() * max(1, int(weight))
            else:
                resampled_errors += [error] * max(1, int(weight))
        landmark_errors = resampled_errors
    else:
        landmark_errors = landmark_errors.flatten()

    # Compute AUC
    auc_metric = 0.0
    sorted_errors = np.array([0] + sorted(landmark_errors))
    ticks = np.arange(0, len(landmark_errors) + 1) / max(1, len(landmark_errors))
    if threshold > 0:
        truncated_errors = sorted_errors[sorted_errors <= threshold]
        truncated_ticks = ticks[:len(truncated_errors)]
        if truncated_errors[-1] < threshold:
            if len(truncated_errors) < len(sorted_errors):
                next_err = sorted_errors[len(truncated_errors)]
                truncated_ticks = np.append(truncated_ticks,
                                            truncated_ticks[-1] + (threshold - truncated_errors[-1]) /
                                            (next_err - truncated_errors[-1]) / len(landmark_errors))
            else:
                truncated_ticks = np.append(truncated_ticks, truncated_ticks[-1])
            truncated_errors = np.append(truncated_errors, threshold)
        auc_metric = sklearn.metrics.auc(truncated_errors, truncated_ticks) / truncated_errors[-1]

    return auc_metric, sorted_errors, ticks


def compute_accuracy(landmark_errors, threshold, normalisation_factors=None, sample_weights=None):
    if normalisation_factors is not None:
        normalisation_factors = np.clip(normalisation_factors, np.finfo(landmark_errors.dtype).eps, None)
        if normalisation_factors.ndim < landmark_errors.ndim:
            normalisation_factors = np.expand_dims(normalisation_factors,
                                                   tuple(range(normalisation_factors.ndim, landmark_errors.ndim)))
        landmark_errors = landmark_errors / normalisation_factors
    if sample_weights is not None:
        sample_weights = np.array(sample_weights, dtype=landmark_errors.dtype)
        if sample_weights.ndim < landmark_errors.ndim:
            sample_weights = np.expand_dims(sample_weights, tuple(range(sample_weights.ndim, landmark_errors.ndim)))
        return np.mean((landmark_errors <= threshold) * sample_weights) / np.clip(
            sample_weights.mean(), np.finfo(sample_weights.dtype).eps, None)
    else:
        return np.mean(landmark_errors <= threshold)
