import torch
import torchvision
from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet34_Weights

from torchvision.models.detection.roi_heads import maskrcnn_loss, fastrcnn_loss, maskrcnn_inference, keypointrcnn_loss, keypointrcnn_inference
from torchvision.models.detection.anchor_utils import AnchorGenerator

from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation, Mask2FormerConfig, MaskFormerConfig


import timm  # timm is a PyTorch library that provides access to pre-trained models
# from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from collections import OrderedDict
import types

    
def convert_relu_to_leakyrelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU())
        else:
            convert_relu_to_leakyrelu(child)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    

def maskrcnn(backbone, n_channels=2, leaky_relu=False, printparams=False, pretrained_backbone=False, path=None):

    backbone  = resnet_fpn_backbone(backbone, pretrained=False, trainable_layers=5)

    model = MaskRCNN(backbone=backbone, num_classes=2, trainable_backbone_layers=5, weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # replace the pre-trained head with a new one

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2) # and replace the mask predictor with a new one

    model.backbone.body.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=1392, max_size=1392, image_mean=[0] * n_channels, image_std=[1] * n_channels)

    if leaky_relu:
        model = convert_relu_to_leakyrelu(model)
    
    if printparams:
        print('Parameters: {}'.format(get_n_params(model)))

    if path:
        model.load_state_dict(torch.load(path))

    return model


def yolo(version='n'):
    model = YOLO('yolov8{}-seg.pt'.format(version)).model
    old_layer = model.model[0].conv
    model.model[0].conv = nn.Conv2d(2, old_layer.out_channels, kernel_size=old_layer.kernel_size, stride=(1, 1), padding=old_layer.padding, bias=old_layer.bias)
    return model

def maskformer():
    custom_config = MaskFormerConfig()
    backbone_config = custom_config.backbone_config
    backbone_config.in_channels = 2 # or num_channels?
    custom_config.backbone_config = backbone_config
    model = MaskFormerForInstanceSegmentation(custom_config)
    return model


if __name__ == '__main__':
    maskformer()