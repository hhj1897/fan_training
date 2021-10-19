import torch
from utils import decode_landmarks


__all__ = ['heatmap_mse_loss', 'landmark_distance_loss']


def heatmap_mse_loss(predicted_heatmaps, heatmaps, sample_weights=1):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([heatmap_mse_loss(x, heatmaps, sample_weights)
                    for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        loss = ((predicted_heatmaps - heatmaps) ** 2).mean(dim=(-1, -2, -3))
        if sample_weights is not None:
            loss = (loss * sample_weights).mean()
        return loss


def landmark_distance_loss(predicted_heatmaps, landmarks, image_shape, normalisation_factors=1, sample_weights=1,
                           gamma=1.0, radius=0.1):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([landmark_distance_loss(x, landmarks, image_shape, normalisation_factors, sample_weights,
                                           gamma, radius) for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        predicted_landmarks = decode_landmarks(predicted_heatmaps, gamma, radius)[0]
        predicted_landmarks[..., 0] *= image_shape[-1] / predicted_heatmaps.shape[-1]
        predicted_landmarks[..., 1] *= image_shape[-2] / predicted_heatmaps.shape[-2]
        loss = (predicted_landmarks - landmarks).norm(dim=-1).mean(dim=-1)
        if normalisation_factors is not None:
            if isinstance(normalisation_factors, torch.Tensor):
                loss /= normalisation_factors.to(loss.dtype).clamp_min(torch.finfo(loss.dtype).eps)
            else:
                loss /= max(torch.finfo(loss.dtype).eps, normalisation_factors)
        if sample_weights is not None:
            loss = (loss * sample_weights).mean()
        return loss
