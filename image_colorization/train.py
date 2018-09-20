from itertools import islice
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import keras.backend as K

from image_colorization.generators.img_generators import unet_generator
from image_colorization.utils.generic_utils import get_filenames

from image_colorization.model.unets import build_unet_model

from image_colorization.model.custom_losses import bce_dice_loss, focal_dice_loss
from image_colorization.model.lovasz_hinge_loss import lovasz_loss
from image_colorization.utils.generic_utils import LoadParameters, get_model_and_preprocessor

params = LoadParameters()
ROOT_DIR = params.params['root_fp']
TRAIN_IMAGE_DIR = params.params['train_image_fp']
VALID_IMAGE_DIR = params.params['valid_image_fp']
WEIGHTS_DIR = params.params['weights_fp']
TENSORBOARD_DIR = params.params['tensorboard_fp']
ONLY_PNEUMONIA = params.params['only_pneumonia']
PNEUMONIA_DF = params.params['pneumonia_df']
#############################################################
BATCH_SIZE = 12
LR = 0.0002  # 0.001 is default
EPOCHS = 150
LOSS = bce_dice_loss #binary_crossentropy #focal_dice_loss #lovasz_loss #bce_dice_loss #focal_dice_loss #mixedPenalty(iouWeight=2, penaltyWeight=2)#bce_dice_loss
MODEL_NAME = params.params['model_name']
_, PREPROCESSOR = get_model_and_preprocessor(MODEL_NAME)
NUM_FOLDS = 10
START_FOLD = 0
END_FOLD = 5
#############################################################


def train_single_fold(train_gen, valid_gen, model, model_fp, steps_per_epoch, validation_steps, callbacks=()):
    model.compile(loss=LOSS, optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['mse'])

    checkpoint = ModelCheckpoint(model_fp, monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='min', period=1)
    callbacks = [checkpoint] + list(callbacks)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=EPOCHS,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=valid_gen,
                                  validation_steps=validation_steps,
                                  class_weight=None,
                                  max_queue_size=15,
                                  workers=3,
                                  use_multiprocessing=False,
                                  )


if __name__ == '__main__':
    image_filenames = get_filenames(image_folder=TRAIN_IMAGE_DIR)

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=1),
                 ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.75, patience=3, min_lr=0.000075, verbose=1),
                 TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=0, write_graph=True, write_images=True)
                 ]

    train_gen = unet_generator(TRAIN_IMAGE_DIR, BATCH_SIZE, preprocessing_function=PREPROCESSOR, horizontal_flip=True)

    valid_gen = unet_generator(VALID_IMAGE_DIR, BATCH_SIZE, preprocessing_function=PREPROCESSOR, shuffle=False)


    model_fp = os.path.join(ROOT_DIR, '/colorization_model.hdf5')
    model = build_unet_model(MODEL_NAME)
    train_single_fold(train_gen, valid_gen, model, model_fp, steps_per_epoch, validation_steps, callbacks=callbacks)
