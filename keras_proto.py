#!/usr/bin/python3

import os
import time
import numpy as np
import argparse
import tensorflow as tf
import cv2 as cv
from contexttimer import Timer

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import keras_retinanet.losses


def get_session():
    """
    use this environment flag to change which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    """
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # There's probably some options worth checking out off config.
    return tf.Session(config=config)


"""
Getting rid of this for now; we're just going to hard-code the test
params right into it.

parser = argparse.ArgumentParser()
parser.add_argument("--camera",
                    type=str,
                    help="use video device for input, specify 0, 1, 2, etc.")
parser.add_argument("--imgdir",
                    type=str,
                    help="specify directory to look for test images in.")
args = parser.parse_args()
"""

# Pretrained classes in the model
classNames = {0: 'Hatch Panel',
              1: 'Vision Target',
              2: 'Cargo'}


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
# load retinanet model
model = models.load_model('resnet_20190124_inf.h5', backbone_name='resnet50')
"""
Compile isn't necessary anymore for prediction.
Was hoping this would speed it up but isn't working anyway.
model.compile(
    loss={
        'regression': keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
"""

# image_directory = args.imgdir
image_directory = "testing_data/images/"
image_files = os.listdir(image_directory)

for image_file in image_files:
    print("reading: {}".format(image_file))
    img = cv.imread(image_directory + '/' + image_file)
    draw = img.copy()
    img = preprocess_image(img)
    img, scale = resize_image(img)
    rows = img.shape[0]
    cols = img.shape[1]
    print("{}x{}".format(rows, cols))
    with Timer() as t:
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
        boxes, scores, labels = model.predict(np.expand_dims(img, axis=0),
                                              batch_size=None, verbose=0,
                                              steps=1)
    print("predict time: {}".format(t.elapsed))
    boxes /= scale
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.6:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.2f}".format(classNames[label], score)
        draw_caption(draw, b, caption)

    cv.imshow('Keras RetinaNet', draw)
    cv.waitKey(0)
cv.destroyAllWindows()
