from PIL import Image
import numpy as np
import cv2

def letterbox_image_padded(image, size=None):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih, c = image_copy.shape
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_copy = cv2.resize(image_copy, (nh, nw), cv2.INTER_CUBIC)
    pad_t, pad_l = (w - nw) // 2, (h - nh) // 2
    pad_r, pad_d = h - pad_l - nh, w - pad_t - nw
    new_image = cv2.copyMakeBorder(image_copy, pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))

    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)
    new_image = np.asarray(new_image)[np.newaxis, :, :, :]
    
    return new_image, meta

def letterbox_image_padded_PIL(image, size=None):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    if size is None:
        new_image = Image.new('RGB', image_copy.size, (0, 0, 0))
        new_image.paste(image_copy)
        meta = None
    else:
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (0, 0, 0))
        new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
        meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    
    return new_image, meta
