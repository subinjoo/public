# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 2019
@author: Subin Joo
"""
import torch
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

# load trained model
def loadTrainedModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model checkpoint
    checkpoint = 'RCNN_model_ssd.pth.tar'
    map_location = ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint,map_location=map_location)
    
    return checkpoint['model'].to(device)
    
# object detection
def teethObjectDetection(original_image,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    image = normalize(to_tensor(resize(original_image)))    
    image = image.to(device) # Move to default device

    predicted_locs, predicted_scores = model(image.unsqueeze(0)) # detection

    # Detect objects using SSD
    det_boxes, _, _ = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
                                                             max_overlap=0.5, top_k=40)

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes[0].to(device) * original_dims # ratio -> pixel
    
    return det_boxes.tolist()
