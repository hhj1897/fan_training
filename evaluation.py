import sklearn
import numpy as np
from utils import decode_landmarks


__all__ = ['run_model_to_compute_errors', 'compute_auc']


def run_model_to_compute_errors(model, data_loader, pbar=None):
    all_errors = {'heatmap_errors': np.array([]), 'losses': np.array([]), 'point_errors': np.array([]),
                  'iods': np.array([]), 'face_heights': np.array([]), 'sample_weights': np.array([])}
    if pbar is not None:
        pbar.reset(total=len(data_loader))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for images, heatmaps, landmarks, face_corners, metadata in data_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        landmarks = landmarks.to(device)

        # Predict heatmaps
        preds = model(images)

        # Compute heatmap errors
        loss = ((preds[-1] - heatmaps) ** 2).mean(dim=(1, 2, 3))
        all_errors['heatmap_errors'] = np.concatenate(
            (all_errors['heatmap_errors'], loss.detach().cpu().numpy()))

        # Compute MSE losses (with intermediate supervision)
        for pred in preds[:-1]:
            loss += ((pred - heatmaps) ** 2).mean(dim=(1, 2, 3))
        loss /= len(preds)
        all_errors['losses'] = np.concatenate((all_errors['losses'], loss.detach().cpu().numpy()))

        # Compute points errors
        predicted_landmarks = decode_landmarks(preds[-1])[0]
        predicted_landmarks[..., 0] *= images.shape[-1] / preds[-1].shape[-1]
        predicted_landmarks[..., 1] *= images.shape[-2] / preds[-1].shape[-2]
        all_errors['point_errors'] = np.concatenate(
            (all_errors['point_errors'],
             ((predicted_landmarks - landmarks).norm(dim=-1)).mean(dim=-1).detach().cpu().numpy()))

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

    if was_training:
        model.train()
    if pbar is not None:
        pbar.close()
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
