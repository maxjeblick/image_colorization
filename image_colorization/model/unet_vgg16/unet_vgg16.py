def get_vgg16_connecting_layers(backbone_model):
    act_128_64 = backbone_model.get_layer('block2_conv2').output

    act_64_128 = backbone_model.get_layer('block3_conv3').output

    act_32_256 = backbone_model.get_layer('block4_conv3').output

    act_16_512 = backbone_model.get_layer('block5_conv3').output

    act_8_512 = backbone_model.get_layer('block5_pool').output

    return [act_128_64, act_64_128, act_32_256, act_16_512, act_8_512]

if __name__=='__main__':
    from keras.applications.vgg16 import VGG16
    from image_colorization.model.unets import unet_v1, build_backbone_model

    backbone_model_ = build_backbone_model(VGG16, pretrained_weights_fp=None)
    backbone_model_.summary()
    connecting_layers = get_vgg16_connecting_layers(backbone_model_)
    model = unet_v1(backbone_model_, connecting_layers)
    model.summary()
