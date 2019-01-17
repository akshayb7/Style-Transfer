from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torchvision import models, transforms

from image_helpers import image_tensor_to_numpy,load_image_as_tensor
from layer_helpers import get_features, gram_matrix

GPU = 'cuda'
def style_transfer(content_path, style_path, model, style_weights=None, alpha=1, beta=1e3,steps=2000):
    '''
    Function to use with VGG-19 only. Requires modification otherwise.
    
    content_path: Path to content image
    style_path: Path to style image
    model: Model to be used
    style_weights: dict, contains layer names as key and weights for layers as values
    alpha: weightage for content image (default=1)
    beta: weightage for style image (default=1e3)
    steps: no. of iterations to run (default=2000)
    '''
    content = load_image_as_tensor(content_path).to(GPU)
    style = load_image_as_tensor(style_path,shape=content.shape[2:]).to(GPU)
    content_features = get_features(content, model)
    style_features = get_features(style, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content.clone().requires_grad_(True).to(GPU)
    if style_weights is None:
        style_weights = {'conv1_1': 1,
                         'conv2_1': 0.8,
                         'conv3_1': 0.5,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}
    optimizer = optim.Adam([target], lr=0.003)
    for i in range(1, steps+1):
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, w, h = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d*w*h)
        total_loss = (alpha * content_loss) + (beta * style_loss)
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
    return content, target 

def plot_images(content, target):
    '''
    Plot the content and target images
    content: image_tensor
    target: image_tensor
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image_tensor_to_numpy(content))
    ax2.imshow(image_tensor_to_numpy(target))
    ax1.set_title('Content image', fontsize=14)
    ax2.set_title('Target image', fontsize=14)