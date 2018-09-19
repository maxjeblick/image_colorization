def get_densenet_169_connecting_layers(backbone_model):
    act_128_64 = backbone_model.get_layer('conv1/relu').output
    act_64_256 = backbone_model.get_layer('pool2_relu').output
    act_32_512 = backbone_model.get_layer('pool3_relu').output
    act_16_1280 = backbone_model.get_layer('pool4_relu').output
    act_8_1664 = backbone_model.get_layer('conv5_block32_concat').output

    return [act_128_64, act_64_256, act_32_512, act_16_1280, act_8_1664]


def get_densenet_201_connecting_layers(backbone_model):
    act_128_64 = backbone_model.get_layer('conv1/relu').output
    act_64_128 = backbone_model.get_layer('pool2_conv').output
    act_32_256 = backbone_model.get_layer('pool3_conv').output
    act_16_896 = backbone_model.get_layer('pool4_conv').output
    act_8_1920 = backbone_model.get_layer('conv5_block32_concat').output

    return [act_128_64, act_64_128, act_32_256, act_16_896, act_8_1920]


if __name__=='__main__':
    from keras.applications import DenseNet169
    from image_colorization.model.unets import unet_v1, build_backbone_model

    backbone_model_ = build_backbone_model(DenseNet169, pretrained_weights_fp=None)
    connecting_layers = get_densenet_169_connecting_layers(backbone_model_)
    model = unet_v1(backbone_model_, connecting_layers)
    model.summary()
