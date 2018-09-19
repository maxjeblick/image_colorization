from keras.layers import ZeroPadding2D, Concatenate

from image_colorization.model.custom_layers import double_conv_layer

def get_resnet50_connecting_layers(backbone_model):
    act_128_64 = backbone_model.get_layer('activation_1').output
    act_63_256 = backbone_model.get_layer('activation_10').output
    act_32_728 = backbone_model.get_layer('activation_22').output
    act_16_1024 = backbone_model.get_layer('activation_40').output
    act_8_2048 = backbone_model.get_layer('activation_49').output

    act_64_256 = ZeroPadding2D(padding=((1, 0), (1, 0)), data_format='channels_last')(act_63_256)

    # add extra conv layer to extract features at highest resolution
    act_128_64_conv = double_conv_layer(act_128_64, 64)
    act_128_128 = Concatenate()([act_128_64_conv, act_128_64])

    return [act_128_128, act_64_256, act_32_728, act_16_1024, act_8_2048]


if __name__=='__main__':
    from keras.applications.resnet50 import ResNet50
    from image_colorization.model.unets import unet_v1, build_backbone_model

    backbone_model_ = build_backbone_model(ResNet50, pretrained_weights_fp=None)
    connecting_layers = get_resnet50_connecting_layers(backbone_model_)
    model = unet_v1(backbone_model_, connecting_layers)
    model.summary()
