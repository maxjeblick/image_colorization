import numpy as np

from keras.preprocessing import image

from image_colorization.utils.generic_utils import get_filenames

SEED = 42


def _transform_flow_generator(flow_gen, image_fp):
    images_fn = get_filenames(image_fp)
    flow_gen.filenames = images_fn
    flow_gen.samples = len(images_fn)
    flow_gen.n = len(images_fn)


def unet_generator(image_fp, batch_size, preprocessing_function, **kwargs):
    """

    :param images: salt images, in the range 0-255
    :param masks: masks, in the range 0-1
    :param batch_size:
    :param preprocessor:
    :param kwargs:
    :return:
    """
    # Creating the training Image and Mask generator
    bw_image_datagen = image.ImageDataGenerator(**kwargs)
    color_image_datagen = image.ImageDataGenerator(**kwargs)

    # Keep the same seed for image and mask generators so they fit together
    x = bw_image_datagen.flow_from_directory(image_fp, batch_size=batch_size, shuffle=True, seed=SEED,
                                          class_mode=None, color_mode='grayscale')
    y = color_image_datagen.flow_from_directory(image_fp, batch_size=batch_size, shuffle=True, seed=SEED,
                                         class_mode=None, color_mode='rgb')

    _transform_flow_generator(bw_image_datagen, image_fp)
    _transform_flow_generator(color_image_datagen, image_fp)
    while True:
        bw_images = next(x)
        color_images = next(y)

        bw_images = np.concatenate([bw_images, bw_images, bw_images], axis=3)
        bw_images = preprocessing_function(bw_images)
        yield [bw_images, color_images]


if __name__ == '__main__':
    pass
