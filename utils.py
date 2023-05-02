import cv2
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from post_processing import post_processing


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def inference(img, model):
    # Receive numpy array image
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[2] == 2:
        new_img = np.ones((img.shape[0], img.shape[1], 3))
        new_img[:, :, 0:2] = img
        new_img[:, :, 2] = img[:, :, 1]
        img = new_img
        
    img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    img = to_tensor(img).unsqueeze(0).to('cpu').numpy()
    start = time.perf_counter()

    with torch.no_grad():
        output_class, output_seg_lungs, output_seg_infected = model(img).values()
    output_class = output_class.argmax(1)
    output_seg_lungs = (np.transpose(output_seg_lungs.argmax(1), (1, 2, 0))*255).astype('uint8')
    output_seg_infected = (np.transpose(output_seg_infected.argmax(1), (1, 2, 0))*255).astype('uint8')
    _, output_seg_lungs, output_seg_infected, _, illustrate_im = post_processing(output_class, output_seg_lungs, output_seg_infected)
    
    end = time.perf_counter()
    inference_time = end - start
    return output_seg_lungs, output_seg_infected, illustrate_im, inference_time
