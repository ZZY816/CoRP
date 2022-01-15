import os
import torch
import time
from torchvision import transforms
import numpy as np
import cv2
"""
mkdir:
    Create a folder if "path" does not exist.
"""
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

"""
write_doc:
    Write "content" into the file(".txt") in "path".
"""
def write_doc(path, content):
    with open(path, 'a') as file:
        file.write(content)

"""
get_time:
    Obtain the current time.
"""
def get_time():
    torch.cuda.synchronize()
    return time.time()

def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def save_tensor_merge(tenor_im, tensor_mask, path, colormap='RED'):
    im = tenor_im.cpu().detach().clone()
    im = im.squeeze(0).numpy()
    im = ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-20)) * 255
    im = np.array(im,np.uint8)
    mask = tensor_mask.cpu().detach().clone()
    mask = mask.squeeze(0).numpy()
    mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-20)) * 255
    mask = np.clip(mask, 0, 255)
    mask = np.array(mask, np.uint8)
    if colormap == 'RED':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_JET)
    elif colormap == 'PINK':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_PINK)
    elif colormap == 'BONE':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_BONE)
    # exec('cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_' + colormap+')')
    im = im.transpose((1, 2, 0))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    mix = cv2.addWeighted(im, 0.5, mask, 0.5, 0)
    cv2.imwrite(path, mask)
