import h5py
import numpy as np
import cv2
import os


def get_data(filename, dataset_ID):

    data_path_main = "./datasets/us2mr/"
    data_path_train = data_path_main + "train" + dataset_ID + "/"
    data_path_test = data_path_main + "test" + dataset_ID + "/"

    if not os.path.exists(data_path_main):
        os.mkdir(data_path_main)
    if not os.path.exists(data_path_train):
        os.mkdir(data_path_train)
    if not os.path.exists(data_path_test):
        os.mkdir(data_path_test)

    h5_file = h5py.File(filename, mode="r")
    keys = list(h5_file.keys())

    # TRAIN SPLIT
    for series_num in range(int(len(keys) / 2)):
        series = np.array(h5_file[keys[series_num]])
        for frame_num in range(len(series)):
            im = series[frame_num]
            im = 255 * ((im - np.min(im)) / np.ptp(im))
            im = np.rot90(im, 1)
            cv2.imwrite(
                data_path_train
                + str(series_num).zfill(4)
                + "-"
                + str(frame_num).zfill(4)
                + ".jpg",
                im
            )

    # TEST SPLIT
    for series_num in range(int(len(keys) / 2), int(len(keys))):
        series = np.array(h5_file[keys[series_num]])
        for frame_num in range(len(series)):
            im = series[frame_num]
            im = 255 * ((im - np.min(im)) / np.ptp(im))
            im = np.rot90(im, 1)
            cv2.imwrite(
                data_path_test
                + str(series_num).zfill(4)
                + "-"
                + str(frame_num).zfill(4)
                + ".jpg",
                im
            )

get_data("../mrus/us_images_resampled800.h5", "A")
get_data("../mrus/mr_images_resampled800.h5", "B")
