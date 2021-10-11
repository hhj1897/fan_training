import cv2
import numpy as np


__all__ = ['plot_landmarks']


def plot_landmarks(frame, landmarks, scores=None, threshold=0.2,
                   connection_colour=(0, 255, 0), landmark_colour=(0, 0, 255),
                   connection_thickness=1, landmark_radius=1):
    if scores is None:
        scores = np.full((landmarks.shape[0],), threshold + 1.0, dtype=float)
    for idx in range(len(landmarks) - 1):
        if (idx != 16 and idx != 21 and idx != 26 and idx != 30 and
                idx != 35 and idx != 41 and idx != 47 and idx != 59):
            if scores[idx] >= threshold and scores[idx + 1] >= threshold:
                cv2.line(frame, tuple(landmarks[idx].astype(int).tolist()),
                         tuple(landmarks[idx + 1].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
        if idx == 30:
            if scores[30] >= threshold and scores[33] >= threshold:
                cv2.line(frame, tuple(landmarks[30].astype(int).tolist()),
                         tuple(landmarks[33].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
        elif idx == 36:
            if scores[36] >= threshold and scores[41] >= threshold:
                cv2.line(frame, tuple(landmarks[36].astype(int).tolist()),
                         tuple(landmarks[41].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
        elif idx == 42:
            if scores[42] >= threshold and scores[47] >= threshold:
                cv2.line(frame, tuple(landmarks[42].astype(int).tolist()),
                         tuple(landmarks[47].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
        elif idx == 48:
            if scores[48] >= threshold and scores[59] >= threshold:
                cv2.line(frame, tuple(landmarks[48].astype(int).tolist()),
                         tuple(landmarks[59].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
        elif idx == 60:
            if scores[60] >= threshold and scores[67] >= threshold:
                cv2.line(frame, tuple(landmarks[60].astype(int).tolist()),
                         tuple(landmarks[67].astype(int).tolist()), color=connection_colour,
                         thickness=connection_thickness, lineType=cv2.LINE_AA)
    for landmark, score in zip(landmarks, scores):
        if score >= threshold:
            cv2.circle(frame, tuple(landmark.astype(int).tolist()), landmark_radius, landmark_colour, -1)
