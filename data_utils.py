import os
import glob
import random
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

VIDEO_DIR = "videos"
SAVED_MODEL_DIR = "saved_models"
PREDICTED_VIDEOS_DIR = "predicted videos"
TIMESTEPS = 5
FPS = 15
IMG_WIDTH = IMG_HEIGHT = 256
IMG_SIZE = (IMG_WIDTH,IMG_HEIGHT)


def get_train_test_files(split_ratio=0.8, shuffle=True):
    files = []
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
    files.extend(video_files)
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.MOV"))
    files.extend(video_files)

    # sort files
    files = sorted(files)
    if shuffle:
        random.shuffle(files)

    total_files = len(files)
    len_train_files = int(total_files * split_ratio)
    return files[:len_train_files], files[len_train_files:]


def normalize_image(img):
    return img / 255.0


def denormalize(img):
    return (img * 255).astype('uint8')


def _get_xy_pair(frames, timesteps, frame_mode, im_size):
    assert frame_mode in ["all", "unique"], "frame_mode must be either of unique or all"
    rng = range(timesteps, len(frames) - 1, timesteps) if frame_mode == "unique" else range(timesteps, len(frames) - 1)

    def resize_image(np_image):
        return np.array(Image.fromarray(np_image, mode="RGB").resize(im_size))

    pairs = []
    for i in rng:
        x = frames[i - timesteps: i]
        x = list(map(resize_image, x))
        y = resize_image(frames[i + 1])

        pairs.append((x, y))
    return pairs


def get_data_gen(files, timesteps=5, fps=15, batch_size=32, frame_mode="unique", for_training=True, im_size=IMG_SIZE):
    x_batch = []
    y_batch = []
    while True:
        for file in files:
            clip = VideoFileClip(file, audio=False)
            frames = list(clip.iter_frames(fps=fps))
            clip.close()
            for pair in _get_xy_pair(frames, timesteps=timesteps, frame_mode=frame_mode, im_size=im_size):
                x = pair[0]
                y = pair[1]
                x_batch.append(x)
                y_batch.append(y)
                if len(x_batch) >= batch_size:
                    yield normalize_image(np.array(x_batch)), normalize_image(np.array(y_batch))
                    x_batch = []
                    y_batch = []
        if not for_training:
            break
