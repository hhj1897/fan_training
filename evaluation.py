import sklearn
import numpy as np
from utils import decode_landmarks
from losses import heatmap_mse_loss, landmark_distance_loss


__all__ = ['run_model_to_compute_errors', 'compute_auc', 'compute_accuracy']


def run_model_to_compute_errors(model, data_loader, pbar=None, gamma=1.0, radius=0.1):
    all_errors = {'heatmap_mse_losses': np.array([]), 'landmark_distance_losses': np.array([]),
                  'heatmap_errors': np.array([]), 'landmark_errors': np.array([]),
                  'iods': np.array([]), 'face_heights': np.array([]), 'sample_weights': np.array([])}
    if pbar is not None:
        pbar.reset(total=len(data_loader))
        pbar.refresh()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for images, heatmaps, landmarks, face_corners, metadata in data_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        landmarks = landmarks.to(device)

        # Predict heatmaps
        predicated_heatmaps = model(images)

        # Compute heatmap mse loss (without reduction)
        all_errors['heatmap_mse_losses'] = np.concatenate(
            (all_errors['heatmap_mse_losses'],
             heatmap_mse_loss(predicated_heatmaps, heatmaps, sample_weights=None).detach().cpu().numpy()))

        # Compute landmark distance loss (normalised by face heights, without reduction)
        all_errors['landmark_distance_losses'] = np.concatenate(
            (all_errors['landmark_distance_losses'],
             landmark_distance_loss(predicated_heatmaps, landmarks, images.shape,
                                    sample_weights=None).detach().cpu().numpy()))

        # Compute heatmap errors
        htm_errors = ((predicated_heatmaps[-1] - heatmaps) ** 2).mean(dim=(-1, -2)).detach().cpu().numpy()
        if all_errors['heatmap_errors'].shape[0] > 0:
            all_errors['heatmap_errors'] = np.vstack((all_errors['heatmap_errors'], htm_errors))
        else:
            all_errors['heatmap_errors'] = htm_errors

        # Compute landmark errors
        predicted_landmarks = decode_landmarks(predicated_heatmaps[-1], gamma=gamma, radius=radius)[0]
        predicted_landmarks[..., 0] *= images.shape[-1] / predicated_heatmaps[-1].shape[-1]
        predicted_landmarks[..., 1] *= images.shape[-2] / predicated_heatmaps[-1].shape[-2]
        lmk_errors = (predicted_landmarks - landmarks).norm(dim=-1).detach().cpu().numpy()
        if all_errors['landmark_errors'].shape[0] > 0:
            all_errors['landmark_errors'] = np.vstack((all_errors['landmark_errors'], lmk_errors))
        else:
            all_errors['landmark_errors'] = lmk_errors

        # Compute IODs
        all_errors['iods'] = np.concatenate(
            (all_errors['iods'],
             (landmarks[:, 45] - landmarks[:, 36]).norm(dim=-1).detach().cpu().numpy()))

        # Compute face heights
        all_errors['face_heights'] = np.concatenate(
            (all_errors['face_heights'],
             (face_corners[:, 2] - face_corners[:, 1]).norm(dim=-1).detach().cpu().numpy()))

        # Record sample weights
        all_errors['sample_weights'] = np.concatenate(
            (all_errors['sample_weights'], metadata['weight'].detach().cpu().numpy()))

        if pbar is not None:
            pbar.update(1)
            pbar.refresh()

    if was_training:
        model.train()
    return all_errors


def compute_auc(all_errors, threshold, sample_weights=None):
    if sample_weights is not None:
        resampled_errors = []
        for error, weight in zip(all_errors, sample_weights):
            resampled_errors += [error] * max(1, int(weight))
        all_errors = resampled_errors

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


def compute_accuracy(landmark_errors, threshold, sample_weights=None):
    if sample_weights is not None:
        sample_weights = np.array(sample_weights, dtype=landmark_errors.dtype)
        if sample_weights.ndim < landmark_errors.ndim:
            sample_weights = np.expand_dims(sample_weights, np.arange(sample_weights.ndim, landmark_errors.ndim))
        return np.mean((landmark_errors <= threshold) * sample_weights) / np.clip(
            sample_weights.mean(), np.finfo(sample_weights.dtype).eps, None)
    else:
        return np.mean(landmark_errors <= threshold)
