from model import Pix2Pix
from data_utils import get_train_test_files, get_data_gen, SAVED_MODEL_DIR, TIMESTEPS, IMG_WIDTH, IMG_HEIGHT


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

#


if __name__ == "__main__":
    # params

    batch_size = 2

    # end params

    train_files, test_files = get_train_test_files()
    train_gen = get_data_gen(files=train_files, timesteps=TIMESTEPS, batch_size=batch_size, im_size=(IMG_WIDTH, IMG_HEIGHT))
    gan = Pix2Pix(im_height=IMG_HEIGHT, im_width=IMG_WIDTH, lookback=TIMESTEPS - 1)
    print("Generator Summary")
    gan.generator.summary()
    print()
    print("Discriminator Summary")
    gan.discriminator.summary()
    print()
    print("Combined Summary")
    gan.combined.summary()



    gan.train(train_gen, epochs=1000, batch_size=batch_size, save_interval=200,
              save_file_name=SAVED_MODEL_DIR+"/water_flow_v0.model")

    print(train_files)
    print(test_files)
    with open(SAVED_MODEL_DIR+"/water_flow_v0.model/train videos.txt",'w') as train_file:

        train_file.writelines(str(train_files))
    train_file.close()

    with open(SAVED_MODEL_DIR+"/water_flow_v0.model/test videos.txt",'w') as test_file:

        test_file.writelines(str(test_files))
    test_file.close()


