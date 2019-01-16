def load_image_as_tensor(img_path, max_size=400, shape=None):
    '''
    Function to load in images and convert them into provided size in x-y dimensions.
    The function will also convert the images to their respective tensors.
    
    img_path: String, path to the input image
    max_size: Int, Max. size allowed for the image while loading in the image (Default:400)
    shape: Int/Tuple, If provided will overide the image size parameter
    '''
    # Load in the image in RGB format
    img = Image.open(img_path).convert('RGB')
    
    # Set the size of the image to be processed
    if max(img.size) > max_size: # Large images slow down processing
        size = max_size
    else:
        size = max(img.size)
    
    # Override the max-size if the user provides a shape for the image 
    if shape is not None:
        size = shape
        
    # Define the transforms for converting image to a tensor
    transform = transforms.Compose([transforms.Resize(size),  # Resize
                                    transforms.ToTensor(), # Convert to tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])]) # Normalize with imagenet stats
    image = transform(img)
    image = image[:3,:,:] # Select the RBG channels only if more channels available (alpha, etc.)
    image = image.unsqueeze(0) # Convert into a single batch of image with an added batch dimension
    
    return image


def image_tensor_to_numpy(tensor):
    '''
    Function to de-normalize an image tensor and convert into numpy array for plotting as image.
    '''
    image = tensor.to('cpu').clone().detach() # Move tensor to cpu and create another copy
    image = image.numpy().squeeze() # Convert to numpy array and remove the batch dimension
    image = image.transpose(1,2,0) # Convert array to (height x width x channel) format
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485,0.456,0.406)) # De-normalize images
    image = image.clip(0,1) # Keep values for each pixel between 0 and 1
    
    return image