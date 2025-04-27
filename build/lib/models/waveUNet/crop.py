def centre_crop(x, target):
    """
    Center-crop x to match target's size
    :param x: Input tensor
    :param target: Target tensor with smaller size
    :return: Cropped input tensor
    """
    if x.shape[-1] == target.shape[-1]:
        return x
        
    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    
    # Handle odd-sized differences by putting the extra pixel at the end
    crop_left = diff // 2
    crop_right = diff - crop_left
    
    if crop_left > 0 or crop_right > 0:
        return x[:, :, crop_left:-crop_right] if crop_right > 0 else x[:, :, crop_left:]
    else:
        return x