import csv
import os
import pathlib

import numpy as np
import yaml
from attrdict import AttrDict

from keras.applications import DenseNet169, Xception, ResNet50, VGG16
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input


CONFIG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'config.yaml')


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class LoadParameters(_Borg):
    def __init__(self, fallback_file=CONFIG_PATH):
        _Borg.__init__(self)

        self.fallback_file = fallback_file
        self.params = self._read_yaml()

    def _read_yaml(self):
        with open(self.fallback_file) as f:
            config = yaml.load(f)
        return AttrDict(config)


def get_filenames(image_folder, suffix='.jpg'):
    # load and shuffle filenames
    filenames = os.listdir(image_folder)
    filenames = [filename for filename in filenames if filename.endswith(suffix)]
    return filenames


def get_model_and_preprocessor(model_name):
    return {'xception': [Xception, xception_preprocess_input],
            'densenet': [DenseNet169, densenet_preprocess_input],
            'resnet50': [ResNet50, resnet50_preprocess_input],
            'vgg16': [VGG16, vgg16_preprocess_input]
            }[model_name]


def divisorGenerator(n):
    '''
    Yields all divisors of n
    :param n:
    :return:
    '''
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)


if __name__=='__main__':
    params = LoadParameters()
    print(params.params)
    print(type(params.params['img_size']))
    print(type(params.params['only_pneumonia']))
