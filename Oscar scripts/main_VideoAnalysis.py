import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


class FrameGenerator:
    def __init__(self, file_paths, n_frames, training=False):
        self.file_paths = file_paths
        self.n_frames = n_frames
        self.training = training
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # Background subtractor

    def __call__(self):
        for video_path, label in self.file_paths:
            frames = self._extract_frames(video_path)
            yield frames, label

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.n_frames).astype(int)
        frames = []

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(frame)

                # Optionally, use the mask to keep the moving objects
                frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

                # Resize and normalize
                frame = cv2.resize(frame, (128, 128))
                frame = frame / 255.0  # Normalize to [0, 1]
                frames.append(frame)
        cap.release()

        # Pad with black frames if fewer than n_frames
        while len(frames) < self.n_frames:
            frames.append(np.zeros((128, 128, 3)))
        return np.array(frames)