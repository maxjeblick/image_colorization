from keras.applications import Xception, DenseNet169, ResNet50, VGG16
from keras.layers import UpSampling2D, Concatenate, Conv2D, Activation, Input, Dense, SpatialDropout2D, AveragePooling2D, Conv2DTranspose, Dropout
from keras.models import Model

from keras.applications.xception import Xception

from image_colorization.model.custom_layers import double_conv_layer, bottleneck, residual_block, squeeze_excite_block, convolution_block, BatchActivate
from image_colorization.model.unet_xception.unet_xception import get_xception_connecting_layers
from image_colorization.model.unet_densenet.unet_densenet import get_densenet_169_connecting_layers
from image_colorization.model.unet_resnet.unet_resnet import get_resnet50_connecting_layers
from image_colorization.model.unet_vgg16.unet_vgg16 import get_vgg16_connecting_layers
from image_colorization.utils.generic_utils import LoadParameters


params = LoadParameters()
img_size = params.params['img_size']
INPUT_SHAPE = (img_size, img_size, 3)
NUM_OUTPUT_CHANNELS = params.params['num_output_cannels']
NUM_FILTERS = params.params['num_filters']


def build_backbone_model(pretrained_model, weights=None, pretrained_weights_fp=None, upsample=True):

    inp = Input(shape=INPUT_SHAPE)

    if upsample:
        inp = UpSampling2D()(inp)

    pretrained_model_ = pretrained_model(include_top=False, weights=weights,
                      input_tensor=inp,
                      pooling='avg')

    x = Dense(1, activation='sigmoid')(pretrained_model_.layers[-1].output)
    model = Model(pretrained_model_.input, x)

    if pretrained_weights_fp is not None:
        model.load_weights(pretrained_weights_fp)

    return model


def unet_v1(backbone_model, connecting_layers, filters=NUM_FILTERS):
    '''
    0.812 on local lb.  31,782,425 Params
    '''
    [act_128, act_64, act_32, act_16, act_8] = connecting_layers

    up_16 = UpSampling2D(size=(2, 2))(act_8)
    up_conv_16 = double_conv_layer(up_16, 16 * filters, dropout=0.15, batch_norm=False)
    up_16_concat = Concatenate()([up_conv_16, act_16])

    up_32 = UpSampling2D(size=(2, 2))(up_16_concat)
    up_conv_32 = double_conv_layer(up_32, 8 * filters, dropout=0.15, batch_norm=False)
    up_32_concat = Concatenate()([up_conv_32, act_32])

    up_64 = UpSampling2D(size=(2, 2))(up_32_concat)
    up_conv_64 = double_conv_layer(up_64, 6 * filters, dropout=0.15, batch_norm=False)
    up_64_concat = Concatenate()([up_conv_64, act_64])

    up_128 = UpSampling2D(size=(2, 2))(up_64_concat)#464 layers with unet_xception
    up_conv_128 = double_conv_layer(up_128, 3 * filters, dropout=0.15, batch_norm=False)
    up_128_concat = Concatenate()([up_conv_128, act_128])

    up_128_concat = Conv2D(2 * filters, (2, 2), activation='elu', padding='same')(up_128_concat)
    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), activation='sigmoid', padding='same')(up_128_concat)

    model = Model(backbone_model.input, conv_final)

    return model


def unet_v2(backbone_model, connecting_layers, filters=NUM_FILTERS):
    '''
    0.812 on local lb.  31,782,425 Params. Overfits with unet_xception
    '''

    [act_128, act_64, act_32, act_16, _] = connecting_layers

    act_8 = AveragePooling2D(pool_size=(2, 2))(act_16)

    # bottleneck_layer =  bottleneck(act_8_2048, filters_bottleneck=2048, depth=2, mode='cascade')
    act_8 = bottleneck(act_8, filters_bottleneck=16 * filters, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu')

    # bottleneck_layer =  bottleneck(act_8_2048, filters_bottleneck=2048, depth=2, mode='cascade')

    up_16 = UpSampling2D(size=(2, 2))(act_8)
    up_conv_16 = double_conv_layer(up_16, 16 * filters)
    up_16_concat = Concatenate()([up_conv_16, act_16])

    up_32 = UpSampling2D(size=(2, 2))(up_16_concat)
    up_conv_32 = double_conv_layer(up_32, 8 * filters)
    up_32_concat = Concatenate()([up_conv_32, act_32])

    up_64 = UpSampling2D(size=(2, 2))(up_32_concat)
    up_conv_64 = double_conv_layer(up_64, 4 * filters)
    up_64_concat = Concatenate()([up_conv_64, act_64])

    up_128 = UpSampling2D(size=(2, 2))(up_64_concat)
    up_conv_128 = double_conv_layer(up_128, 3 * filters)
    up_128_concat = Concatenate()([up_conv_128, act_128])

    up_128_final = Conv2D(1 * filters, (2, 2), activation='relu', padding='same')(up_128_concat)
    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), activation='sigmoid')(up_128_final)

    model = Model(backbone_model.input, conv_final)

    return model


def unet_v3(backbone_model, connecting_layers, filters=NUM_FILTERS):
    '''
    0.812 on local lb.  31,782,425 Params
    '''

    [act_128, act_64, act_32, act_16, _] = connecting_layers

    act_8 = AveragePooling2D(pool_size=(2,2))(act_16)

    # bottleneck_layer =  bottleneck(act_8_2048, filters_bottleneck=2048, depth=2, mode='cascade')
    act_8 = bottleneck(act_8, filters_bottleneck=16 * filters, mode='cascade', depth=4,
                       kernel_size=(3, 3), activation='relu')

    # bottleneck_layer =  bottleneck(act_8_2048, filters_bottleneck=2048, depth=2, mode='cascade')

    up_16 = UpSampling2D(size=(2, 2))(act_8)
    up_16 = Conv2D(16 * filters, kernel_size=(3, 3), activation='relu', padding='same')(up_16)
    up_16_concat = Concatenate()([up_16, act_16])
    up_conv_16 = double_conv_layer(up_16_concat, 16 * filters, dropout=0.3)

    up_32 = UpSampling2D(size=(2, 2))(up_conv_16)
    up_32 = Conv2D(8 * filters, kernel_size=(3, 3), activation='relu', padding='same')(up_32)
    up_32_concat = Concatenate()([up_32, act_32])
    up_conv_32 = double_conv_layer(up_32_concat, 8 * filters, dropout=0.3)

    up_64 = UpSampling2D(size=(2, 2))(up_conv_32)
    up_64 = Conv2D(4 * filters, kernel_size=(3, 3), activation='relu', padding='same')(up_64)
    up_64_concat = Concatenate()([up_64, act_64])
    up_conv_64 = double_conv_layer(up_64_concat, 4 * filters)

    up_128 = UpSampling2D(size=(2, 2))(up_conv_64)
    up_128 = Conv2D(3 * filters, kernel_size=(3, 3), activation='relu', padding='same')(up_128)
    up_128 = Concatenate()([up_128, act_128])
    up_conv_128 = double_conv_layer(up_128, 3 * filters)
    up_128_concat = Concatenate()([up_conv_128, act_128])

    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), activation='sigmoid')(up_128_concat)

    model = Model(backbone_model.input, conv_final)

    return model


def unet_v4(backbone_model, connecting_layers, filters=NUM_FILTERS):

    [act_128, act_64, act_32, act_16, act_8] = connecting_layers

    up_16 = Conv2DTranspose(20 * filters, kernel_size=(2, 2), strides=(2, 2),  activation='relu', padding='valid')(act_8)
    up_conv_16 = double_conv_layer(up_16, 16 * filters)
    up_16_concat = Concatenate()([up_conv_16, act_16])

    up_32 = Conv2DTranspose(10 * filters, kernel_size=(2, 2), strides=(2, 2),  activation='relu', padding='valid')(up_16_concat)
    up_conv_32 = double_conv_layer(up_32, 8 * filters)
    up_32_concat = Concatenate()([up_conv_32, act_32])
    up_32_concat = squeeze_excite_block(up_32_concat, ratio=16)

    up_64 = Conv2DTranspose(5 * filters, kernel_size=(2, 2), strides=(2, 2),  activation='relu', padding='valid')(up_32_concat)
    up_conv_64 = double_conv_layer(up_64, 6 * filters)
    up_64_concat = Concatenate()([up_conv_64, act_64])
    up_64_concat = squeeze_excite_block(up_64_concat, ratio=10)

    up_128 = Conv2DTranspose(3 * filters, kernel_size=(2, 2), strides=(2, 2),  activation='relu', padding='valid')(up_64_concat)
    up_conv_128 = double_conv_layer(up_128, 2 * filters)
    up_128_concat = Concatenate()([up_conv_128, act_128])
    up_128_concat = Conv2D(3 * filters, (2, 2), activation='relu', padding='same')(up_128_concat)

    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), activation='sigmoid')(up_128_concat)

    model = Model(backbone_model.input, conv_final)

    return model


def unet_v5(backbone_model, connecting_layers, filters=NUM_FILTERS, DropoutRatio = 0.2, ACTIVATION='relu'):
    [act_128, act_64, act_32, act_16, act_8] = connecting_layers

    # 8 -> 16
    deconv4 = Conv2DTranspose(filters * 8, (3, 3), strides=(2, 2), padding="same")(act_8)
    uconv4 = Concatenate()([deconv4, act_16])
    uconv4 = SpatialDropout2D(DropoutRatio)(uconv4)

    uconv4 = Conv2D(filters * 8, (3, 3), activation='relu', padding="same")(uconv4)
    uconv4 = residual_block(uconv4, filters * 8)
    uconv4 = residual_block(uconv4, filters * 8)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(filters * 6, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = Concatenate()([deconv3, act_32])
    uconv3 = SpatialDropout2D(DropoutRatio)(uconv3)

    uconv3 = Conv2D(filters * 6, (3, 3), activation='relu', padding="same")(uconv3)
    uconv3 = residual_block(uconv3, filters * 6)
    uconv3 = residual_block(uconv3, filters * 6)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = Concatenate()([deconv2, act_64])

    uconv2 = SpatialDropout2D(DropoutRatio)(uconv2)
    uconv2 = Conv2D(filters * 4, (3, 3), activation='relu', padding="same")(uconv2)
    uconv2 = residual_block(uconv2, filters * 4)
    uconv2 = residual_block(uconv2, filters * 4)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(filters * 3, (3, 3), strides=(2, 2), padding="same")(uconv2)

    uconv1 = SpatialDropout2D(DropoutRatio)(deconv1)
    uconv1 = Conv2D(filters * 3, (3, 3), activation='relu', padding="same")(uconv1)
    uconv1 = residual_block(uconv1, filters * 3)
    uconv1 = residual_block(uconv1, filters * 3)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Concatenate()([uconv1, act_128])
    uconv1 = Conv2D(filters * 3, (2, 2), activation='relu', padding="same")(uconv1)
    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(backbone_model.input, conv_final)
    return model


def unet_v6(backbone_model, connecting_layers, filters=NUM_FILTERS, activation='relu'):
    '''
    0.827 on local lb 2fold cv.
    '''

    #[act_128_196, act_64_364, act_32_984, act_16_1752, act_8_2048]
    [act_128, act_64, act_32, act_16, act_8] = connecting_layers

    up_16 = UpSampling2D(size=(2, 2))(act_8)
    up_conv_16 = double_conv_layer(up_16, 16 * filters, dropout=0.2, batch_norm=False)
    act_16 = Conv2D(16 * filters, (2, 2), activation=activation, padding='same')(act_16)
    up_16_concat = Concatenate()([up_conv_16, act_16])

    up_32 = UpSampling2D(size=(2, 2))(up_16_concat)
    up_conv_32 = double_conv_layer(up_32, 8 * filters, dropout=0.2, batch_norm=False)
    act_32 = Conv2D(8 * filters, (2, 2), activation=activation, padding='same')(act_32)
    up_32_concat = Concatenate()([up_conv_32, act_32])

    up_64 = UpSampling2D(size=(2, 2))(up_32_concat)
    up_conv_64 = double_conv_layer(up_64, 4 * filters, dropout=0.2, batch_norm=False)
    act_64 = Conv2D(4 * filters, (2, 2), activation=activation, padding='same')(act_64)
    up_64_concat = Concatenate()([up_conv_64, act_64])

    up_128 = UpSampling2D(size=(2, 2))(up_64_concat)#464 layers with unet_xception
    up_conv_128 = double_conv_layer(up_128, 2 * filters, dropout=0.2, batch_norm=False)
    act_128 = Conv2D(2 * filters, (2, 2), activation=activation, padding='same')(act_128)
    up_128_concat = Concatenate()([up_conv_128, act_128])

    up_128_final = Conv2D(1 * filters, (2, 2), activation=activation, padding='same')(up_128_concat)
    conv_final = Conv2D(NUM_OUTPUT_CHANNELS, (1, 1), activation='sigmoid', padding='same')(up_128_final)

    model = Model(backbone_model.input, conv_final)

    return model


def build_unet_model(model_name, unet_model=unet_v1):
    pretrained_model, connecting_layers = {'xception': [Xception, get_xception_connecting_layers],
                                           'densenet': [DenseNet169, get_densenet_169_connecting_layers],
                                           'resnet50': [ResNet50, get_resnet50_connecting_layers],
                                           'vgg16': [VGG16, get_vgg16_connecting_layers]
                                           }[model_name]

    backbone_model_ = build_backbone_model(pretrained_model, weights='imagenet')
    model = unet_model(backbone_model_, connecting_layers(backbone_model_))

    return model


if __name__ == '__main__':

    backbone_model_ = build_backbone_model(Xception, pretrained_weights_fp=None)
    connecting_layers = get_xception_connecting_layers(backbone_model_)
    model = unet_v4(backbone_model_, connecting_layers)
    model.compile(optimizer='Adam', loss='mse')
    model.summary()
