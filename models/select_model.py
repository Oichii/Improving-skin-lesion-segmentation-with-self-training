from models.SegmentationUnet import UNet
import segmentation_models_pytorch as smp


def select_model(net_name, in_channels=3, width=1, backbone=''):
    if net_name == 'unet':
        model = UNet(in_channels=in_channels, n_classes=1, width=width)
        return model

    elif net_name == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
        )
        return model
    elif net_name == "deeplab":
        model = smp.DeepLabV3(
            encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
        )
        return model
    elif net_name == "deeplabp":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
        )
        return model

