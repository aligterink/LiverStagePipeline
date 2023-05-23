import torch
import torchvision
from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet34_Weights

from torchvision.models.detection.roi_heads import maskrcnn_loss, fastrcnn_loss, maskrcnn_inference, keypointrcnn_loss, keypointrcnn_inference
from torchvision.models.detection.anchor_utils import AnchorGenerator

import timm  # timm is a PyTorch library that provides access to pre-trained models

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from collections import OrderedDict
import types

# class Identity(torch.nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
        
#     def forward(self, x, targets):
#         return torch.Tensor(x, targets
    
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

def make_trainable(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def maskrcnn_rn50(path=None):
    num_classes = 2
    # model = maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5, weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)

    # replace bark with new_bark for this object only
    # model.forward = types.MethodType(mrcnn_forward, model)
    # roi_heads = model.roi_heads
    # roi_heads.forward = types.MethodType(roiheads_forward, roi_heads)
    # model.roi_heads = roi_heads
    # model.eager_outputs = types.MethodType(eager_outputs2, model)


    in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes) # replace the pre-trained head with a new one

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes) # and replace the mask predictor with a new one

    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=1392, max_size=1392, image_mean=[0, 0], image_std=[1, 1])
    # model.transform = Identity()
    model.backbone.body.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # convert_relu_to_leakyrelu(model)

    for param in model.parameters():
        if param.requires_grad == False:
            print('Non-trainable parameter found')

    if path:
        model.load_state_dict(torch.load(path))
    return model

def maskrcnn_rn34():
    backbone  = resnet_fpn_backbone("resnet34", pretrained=False, trainable_layers=5)

    model = MaskRCNN(backbone=backbone, num_classes=2, trainable_backbone_layers=5, weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # replace the pre-trained head with a new one

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2) # and replace the mask predictor with a new one
    model
    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0, 0], image_std=[1, 1])
    model.backbone.body.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    convert_relu_to_leakyrelu(model)

    return model

# def maskrcnn_rn18():
#     backbone  = resnet_fpn_backbone("resnet18", pretrained=True, trainable_layers=5)

#     model = MaskRCNN(backbone=backbone, num_classes=2, trainable_backbone_layers=5, weights=ResNet34_Weights.DEFAULT)
    
#     in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # replace the pre-trained head with a new one

#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
#     hidden_layer = 256
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2) # and replace the mask predictor with a new one

#     model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0, 0], image_std=[1, 1])
#     model.backbone.body.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#     # convert_relu_to_leakyrelu(model)

#     return model


    

def maskrcnn(backbone, n_channels=2, leaky_relu=False, printparams=False, pretrained_backbone=False, path=None):

    backbone  = resnet_fpn_backbone(backbone, pretrained=False, trainable_layers=5)

    # anchor_sizes = ((16,), (24,), (32,), (64,), (128,))
    # # anchor_sizes = ((8,), (16,), (32,))

    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # rpn_anchor_generator = AnchorGenerator(
    #     anchor_sizes, aspect_ratios
    # )
    # model = MaskRCNN(backbone=backbone, rpn_anchor_generator=rpn_anchor_generator, num_classes=2, trainable_backbone_layers=5, weights=None)

    model = MaskRCNN(backbone=backbone, num_classes=2, trainable_backbone_layers=5, weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # replace the pre-trained head with a new one

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2) # and replace the mask predictor with a new one

    # model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0, 0], image_std=[1, 1])
    model.backbone.body.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=1392, max_size=1392, image_mean=[0] * n_channels, image_std=[1] * n_channels)
    # model.backbone.body.conv1.in_channels = n_channels

    if leaky_relu:
        model = convert_relu_to_leakyrelu(model)
    
    if printparams:
        print('Parameters: {}'.format(get_n_params(model)))

    if path:
        model.load_state_dict(torch.load(path))

    return model

# def maskrcnn_vit():


if __name__ == '__main__':
    # maskrcnn('vit')
    # print(timm.list_models())
    maskrcnn('resnet50', printparams=True)
    # print(dir(tim?m.models))
    