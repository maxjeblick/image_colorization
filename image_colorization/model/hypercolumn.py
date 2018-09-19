from keras.layers import UpSampling2D, Concatenate, Conv2D, SpatialDropout2D
from keras.models import Model


def create_hypercolumn(unet):
    '''
    Add additional upsampling layers to an existing unet
    :param unet:
    :return:
    '''
    upsampling_layers = [unet.get_layer('up_sampling2d_{}'.format(i)).output for i in range(2, 5)]
    # shape is (16, 16, x), (32, 32, xx), (64, 64, xxx) and
    upsampling_layers = [UpSampling2D(size=(2 ** (3-i), 2**(3-i)), name='upsample_{}'.format(i))(layer) for i, layer in enumerate(upsampling_layers)]

    last_128_layer = unet.layers[-2].output
    layers_128 = Concatenate(name='final_concat')(upsampling_layers + [last_128_layer])
    layers_128 = SpatialDropout2D(rate=0.3, name='final_spatial_dropout')(layers_128)
    #layers_128 = Conv2D(40, (2, 2), activation='relu', padding='same')(layers_128)
    conv_final = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='final_conv2d')(layers_128)

    hypercolumn_model = Model(unet.input, conv_final)

    return hypercolumn_model


if __name__ == "__main__":
    from rsna_pneumonia.model.unets import unet_v1, build_backbone_model
    from rsna_pneumonia.model.unet_xception.unet_xception import get_xception_connecting_layers
    from keras.applications.xception import Xception

    backbone_model_ = build_backbone_model(Xception, pretrained_weights_fp=None)
    connecting_layers = get_xception_connecting_layers(backbone_model_)
    model = unet_v1(backbone_model_, connecting_layers)
    model.summary()
    hypercolumn_model = create_hypercolumn(model)
    hypercolumn_model.summary()
