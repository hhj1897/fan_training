from utils import decode_landmarks


__all__ = ['heatmap_mse_loss', 'landmark_distance_loss']


def heatmap_mse_loss(predicted_heatmaps, heatmaps, sample_weights=None):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([heatmap_mse_loss(x, heatmaps, sample_weights)
                    for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        if sample_weights is not None:
            return (((predicted_heatmaps - heatmaps.to(predicted_heatmaps.device)) ** 2).mean(dim=(-1, -2, -3)) *
                    sample_weights.to(predicted_heatmaps.device)).mean()
        else:
            return ((predicted_heatmaps - heatmaps.to(predicted_heatmaps.device)) ** 2).mean()


def landmark_distance_loss(predicted_heatmaps, landmarks, image_shape, sample_weights=None, gamma=1.0, radius=0.1):
    if isinstance(predicted_heatmaps, (list, tuple)):
        return sum([landmark_distance_loss(x, landmarks, image_shape, sample_weights, gamma, radius)
                    for x in predicted_heatmaps]) / len(predicted_heatmaps)
    else:
        predicted_landmarks = decode_landmarks(predicted_heatmaps, gamma, radius)[0]
        predicted_landmarks[..., 0] *= image_shape[-1] / predicted_heatmaps.shape[-1]
        predicted_landmarks[..., 1] *= image_shape[-2] / predicted_heatmaps.shape[-2]
        if sample_weights is not None:
            return ((predicted_landmarks - landmarks.to(predicted_heatmaps.device)).norm(dim=-1).mean(dim=-1) *
                    sample_weights.to(predicted_heatmaps.device)).mean()
        else:
            return (predicted_landmarks - landmarks.to(predicted_heatmaps.device)).norm(dim=-1).mean()
