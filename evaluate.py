import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow.keras.models import load_model
from moviepy.editor import CompositeVideoClip, ImageSequenceClip
from data_utils import get_data_gen, get_train_test_files, denormalize, PREDICTED_VIDEOS_DIR, SAVED_MODEL_DIR, \
    IMG_WIDTH, IMG_HEIGHT, TIMESTEPS, normalize_image, VIDEO_DIR, _get_xy_pair, IMG_SIZE, FPS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
# TODO: evaluate frame-by-frame



def generate_video(saved_model_path, video_file):
    """Uses the trained model to predict the frames and produce a video out of them"""
    # load model
    model = load_model(saved_model_path)
    video_path = VIDEO_DIR + '/' + video_file
    test_gen = get_data_gen(files=[video_path], timesteps=TIMESTEPS, batch_size=batch_size,
                            im_size=(IMG_WIDTH, IMG_HEIGHT))

    y_true = []
    y_pred = []
    clip = VideoFileClip(video_path, audio=False)
    frames = list(clip.iter_frames(fps=FPS))
    for _ in range(len(frames)):
        x, y = next(test_gen)
        y_true.extend(y)

        predictions = model.predict(x)
        y_pred.extend(predictions)

    build_video(y_true, y_pred, video_file)


def generate_future_prediction_video(saved_model_path, video_file):
    # load model
    model = load_model(saved_model_path)
    print(model.summary())
    video_path = VIDEO_DIR + '/' + video_file
    test_gen = get_data_gen(files=[video_path], timesteps=TIMESTEPS, batch_size=batch_size,
                            im_size=(IMG_WIDTH, IMG_HEIGHT))

    y_true = []
    y_pred = []

    clip = VideoFileClip(video_path, audio=False)
    frames = list(clip.iter_frames(fps=FPS))
    for i in range(len(frames)):
        start_x, y = next(test_gen)
        y_true.extend(y)
        if len(y_pred) < 7:
            print(len(y_pred))
            predictions = model.predict(start_x)
        else:
            def resize_image(np_image):
                return np.array(Image.fromarray(np_image, mode="RGB").resize(IMG_SIZE))

            print(f'i: {i}')
            print(f'len(y_pred): {len(y_pred)}')
            # TODO:ensure these are correct frames
            previous_predicted_frames = y_pred[i - TIMESTEPS - 2:i]
            print(f'len(previous_predicted_frames): {len(previous_predicted_frames)}')
            # previous_predicted_frames = y_pred[i - TIMESTEPS:i]
            pairs = _get_xy_pair(previous_predicted_frames, timesteps=5, frame_mode='unique', im_size=IMG_SIZE)
            print(f'len(pairs): {len(pairs)}')
            pair = pairs[0]
            x = pair[0]
            # x = list(map(resize_image, x))
            x = normalize_image(np.array([x]))
            print(f'len(x): {len(x)}')

            predictions = model.predict(x)
        y_pred.extend(predictions)

    build_video(y_true, y_pred, video_file + '-future')


def build_video(ground_frames, predicted_frames, filename):
    clip1 = ImageSequenceClip([denormalize(i) for i in ground_frames], fps=FPS)
    clip2 = ImageSequenceClip([denormalize(i) for i in predicted_frames], fps=FPS)
    clip2 = clip2.set_position((clip1.w, 0))
    video = CompositeVideoClip((clip1, clip2), size=(clip1.w * 2, clip1.h))
    video.write_videofile(PREDICTED_VIDEOS_DIR + "/{}.mp4".format(filename), fps=FPS)


def plot_different_models(timesteps=[5, 10]):
    """
    Compares ssim/psnr of different models. The models for each of the supplied timestap
    must be present
    param TIMESTEPS A list of numbers indicating the TIMESTEPS that were used for training different models
    """

    psnrs = {}
    ssims = {}
    for ts in timesteps:
        model_name = "r_p2p_gen_t{}.model".format(ts)
        model = load_model(model_name)
        train_files, test_files = get_train_test_files()
        test_gen = get_data_gen(files=train_files, timesteps=ts, batch_size=batch_size, im_size=(IMG_WIDTH, IMG_HEIGHT))

        y_true = []
        y_pred = []

        for _ in range(200):
            x, y = next(test_gen)
            y_true.extend(y)

            predictions = model.predict_on_batch(x)
            y_pred.extend(predictions)
        psnrs[ts] = [peak_signal_noise_ratio(denormalize(yt), denormalize(p)) for yt, p in zip((y_true), (y_pred))]
        ssims[ts] = [structural_similarity(denormalize(yt), denormalize(p), multichannel=True) for yt, p in
                     zip((y_true), (y_pred))]

    plt.boxplot([psnrs[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_psnrs_all.png")

    plt.figure()
    plt.boxplot([ssims[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_ssims_all.png")





if __name__ == "__main__":
    # params

    batch_size = 1

    # end params
    # plot_different_models(timesteps=[5, 10])
    generate_video(SAVED_MODEL_DIR + "/water_flow_v0.model", video_file='IMG_1543.MOV')
    generate_future_prediction_video(SAVED_MODEL_DIR + "/water_flow_v0.model", video_file='IMG_1543.MOV')
    # working_out()
