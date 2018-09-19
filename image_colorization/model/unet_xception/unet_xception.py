from keras.layers import ZeroPadding2D, Concatenate


def get_xception_simple_connecting_layers(backbone_model):
    act_125_128 = backbone_model.get_layer('block2_sepconv2_bn').output
    act_63_256 = backbone_model.get_layer('block3_sepconv2_bn').output
    act_32_728 = backbone_model.get_layer('block4_sepconv2_bn').output
    act_16_1024 = backbone_model.get_layer('block13_sepconv2_bn').output
    act_8_2048 = backbone_model.get_layer('block14_sepconv2_bn').output

    act_64_256 = ZeroPadding2D(padding=((1, 0), (1, 0)), data_format='channels_last')(act_63_256)
    act_128_128 = ZeroPadding2D(padding=((2, 1), (2, 1)), data_format='channels_last')(act_125_128)

    return [act_128_128, act_64_256, act_32_728, act_16_1024, act_8_2048]


def get_xception_connecting_layers(backbone_model):
    act_125_128 = backbone_model.get_layer('block2_sepconv2_bn').output
    act_125_64 = backbone_model.get_layer('block1_conv2_act').output
    act_125_196 = Concatenate()([act_125_128, act_125_64])

    act_63_256 = backbone_model.get_layer('block3_sepconv2_bn').output
    act_63_128 = backbone_model.get_layer('add_1').output
    act_63_364 = Concatenate()([act_63_256, act_63_128])

    act_32_728 = backbone_model.get_layer('block4_sepconv2_bn').output
    act_32_256 = backbone_model.get_layer('add_2').output
    act_32_984 = Concatenate()([act_32_728, act_32_256])

    act_16_1024 = backbone_model.get_layer('block13_sepconv2_bn').output
    act_16_728 = backbone_model.get_layer('add_11').output
    act_16_1752 = Concatenate()([act_16_1024, act_16_728])
    act_8_2048 = backbone_model.get_layer('block14_sepconv2_bn').output

    act_64_364 = ZeroPadding2D(padding=((1, 0), (1, 0)), data_format='channels_last')(act_63_364)
    act_128_196 = ZeroPadding2D(padding=((2, 1), (2, 1)), data_format='channels_last')(act_125_196)

    return [act_128_196, act_64_364, act_32_984, act_16_1752, act_8_2048]


if __name__=='__main__':
    from keras.applications.xception import Xception
    from image_colorization.model.unets import unet_v1, build_backbone_model

    backbone_model_ = build_backbone_model(Xception, pretrained_weights_fp=None)
    connecting_layers = get_xception_connecting_layers(backbone_model_)
    model = unet_v1(backbone_model_, connecting_layers)
    model.summary()
