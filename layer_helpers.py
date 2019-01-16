import torch

def get_features(image, model, layers=None):
    '''
    Function to return the activation maps of the relevant layers of an image.
    Returns the activations of VGG19 matching Gatys et al (2016) if explicit layers are not provided
    
    image: tensor, the image tensor whose activations are to be calculated
    model: pass the trained CNN model 
    layers: dict, contaning names of the desired layers in the architecture as the keys and the names we want as their values.
    '''
    # Naming the layers of VGGNet as per the Gatys et al paper
    if layers is None:
        layers = {'0': 'conv1_1', # Style layers
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1',
                  '21':'conv4_2'} # Content layer
    
    features = {} # Dict to store layer activations of the image
    x = image # Starting tensor of image
    
    for name,layer in model._modules.items(): # ._modules gives an Ordered dict of each layer with its name
        x = layer(x) # Pass the previous output of a layer through the next layer
        if name in layers: # If output of the layer we want store its activation
            features[layers[name]] = x 
    
    return features

def gram_matrix(tensor):
    '''
    Calculates the Gram matrix of a given tensor
    '''
    # Get the depth, width, height of the tensor
    _, d, w, h = tensor.shape
    
    # Reshape the tensor
    tensor = tensor.view(d, w*h) # d rows of w*h flattened tensors 
    
    # Multiply the tensor with its transpose to get the gram matrix (correlations)
    return torch.mm(tensor, tensor.t())