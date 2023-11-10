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

from utils import mask_utils
from multiprocessing import Pool
import timm  # timm is a PyTorch library that provides access to pre-trained models
# from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from collections import OrderedDict
import types
import numpy as np

import torch
from multiprocessing import Pool, set_start_method

torch.set_default_dtype(torch.float64)

def mask_3d_to_2d(mask_3d, max_overlap=0.5):
    mask_3d = (mask_3d.squeeze(1) > 0.5).bool()
    
    # Initialize 2D tensor with zeros
    mask_2d = torch.zeros(mask_3d.size()[1:], dtype=torch.int32, device=mask_3d.device)
    
    for i in range(mask_3d.size(0)):
        # Create a mask for the current layer
        current_layer = mask_3d[i, :, :]
        
        # Compare overlap of the current layer with the 2D tensor
        intersection = (current_layer & mask_2d).sum()
        mask_sum = current_layer.sum()
        overlap = intersection.float() / mask_sum.float() if mask_sum > 0 else torch.tensor(0.0)
        
        # If the IoU is less than the threshold, add the layer to the output tensor
        if overlap <= max_overlap:
            mask_2d[current_layer] = i + 1
    return mask_2d

### Mask R-CNN and helper funcs
def parse_maskrcnn(model, batch):

    X = batch['images']

    if all(k in batch for k in ['boxes', 'labels', 'masks_3d']): # when in train mode
        y = [{'boxes': batch['boxes'][i], 'labels': batch['labels'][i], 'masks': batch['masks_3d'][i]} for i in range(len(batch['images']))]
        output = model(X, y) # Forward pass
        loss = sum(loss for loss in output.values())
        return loss

    else: # when in evaluation mode
        output = model(X)
        
        # GPU code
        masks_3d = [sample['masks'] for sample in output]
        batch_masks_2d = [mask_3d_to_2d(mask_3d).cpu().numpy() for mask_3d in masks_3d]
        
        return batch_masks_2d

def maskrcnn(backbone='resnet101', n_channels=2, pretrained_backbone=False, path=None):

    backbone  = resnet_fpn_backbone(backbone, pretrained=pretrained_backbone, trainable_layers=5)
    model = MaskRCNN(backbone=backbone, num_classes=2, trainable_backbone_layers=5, weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # replace the pre-trained head with a new one

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2) # and replace the mask predictor with a new one

    model.backbone.body.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=1392, max_size=1392, image_mean=[0] * n_channels, image_std=[1] * n_channels)

    if path:
        model.load_state_dict(torch.load(path))

    return model, parse_maskrcnn


def yolo(version='n'): # takes polygon input
    model = YOLO('yolov8{}-seg.pt'.format(version)).model
    old_layer = model.model[0].conv
    model.model[0].conv = nn.Conv2d(2, old_layer.out_channels, kernel_size=old_layer.kernel_size, stride=(1, 1), padding=old_layer.padding, bias=old_layer.bias)
    return model

### Maskformer and helper funcs
def parse_maskformer(model, batch):
    # Forward pass
    output = model(pixel_values=batch['images'], mask_labels=[m.type(torch.float32) for m in batch['masks_3d']], class_labels=batch['labels'])

    processor = MaskFormerImageProcessor()
    # results = processor.post_process_instance_segmentation(output)# target_sizes=[batch['images'].size[::-1]])
    results = processor.post_process_instance_segmentation(output, target_sizes=[1392, 1040])
    print(results[0]['segmentation'].shape)

    return output.loss

def maskformer():
    custom_config = MaskFormerConfig()
    backbone_config = custom_config.backbone_config
    backbone_config.num_channels = 2
    # backbone_config.id2label = {0: 'parasite'}
    backbone_config.image_size = (1392, 1040)
    backbone_config.in_channels = 2
    custom_config.backbone_config = backbone_config
    custom_config.mask_feature_size = 1392
    
    model = MaskFormerForInstanceSegmentation(custom_config)
    print(custom_config)
    return model


if __name__ == '__main__':
    model = maskformer()
    # pixel_values = torch.rand((12, 3, 1040, 1392))
    # mask_labels = [torch.rand((12, 3, 1040, 1392)) for i in range(10)]
    # class_labels = [torch.ones((12)) for i in range(10)]
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # model = model.to(device)
    # pixel_values = pixel_values.to(device)
    # mask_labels = [labels.to(device) for labels in mask_labels]
    # class_labels = [labels.to(device) for labels in class_labels]

    # output = model(
    #     pixel_values = pixel_values,
    #     mask_labels = mask_labels,
    #     class_labels = class_labels
    # )
