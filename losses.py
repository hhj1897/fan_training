import torch
from utils import decode_landmarks


__all__ = ['heatmap_mse_loss', 'landmark_distance_loss']


def heatmap_mse_loss(predicted_heatmaps, heatmaps, sample_weights=None, reduce=True):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([heatmap_mse_loss(x, heatmaps, sample_weights, reduce)
                    for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        num_landmarks = min(predicted_heatmaps.shape[-3], heatmaps.shape[-3])
        loss = ((predicted_heatmaps[..., :num_landmarks, :, :] -
                 heatmaps[..., :num_landmarks, :, :]) ** 2).mean(dim=(-1, -2, -3))
        if sample_weights is not None:
            loss *= sample_weights
        return loss.mean() if reduce else loss


def landmark_distance_loss(predicted_heatmaps, landmarks, image_shape, normalisation_factors=None,
                           sample_weights=None, gamma=1.0, radius=0.1, reduce=True):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([landmark_distance_loss(x, landmarks, image_shape, normalisation_factors, sample_weights, gamma,
                                           radius, reduce) for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        num_landmarks = min(predicted_heatmaps.shape[-3], landmarks.shape[-2])
        predicted_landmarks = decode_landmarks(predicted_heatmaps[..., :num_landmarks, :, :], gamma, radius)[0]
        predicted_landmarks[..., 0] *= image_shape[-1] / predicted_heatmaps.shape[-1]
        predicted_landmarks[..., 1] *= image_shape[-2] / predicted_heatmaps.shape[-2]
        loss = (predicted_landmarks - landmarks[..., :num_landmarks, :]).norm(dim=-1).mean(dim=-1)
        if normalisation_factors is not None:
            if isinstance(normalisation_factors, torch.Tensor):
                loss /= normalisation_factors.to(loss.dtype).clamp_min(torch.finfo(loss.dtype).eps)
            else:
                loss /= max(torch.finfo(loss.dtype).eps, normalisation_factors)
        if sample_weights is not None:
            loss *= sample_weights
        return loss.mean() if reduce else loss
