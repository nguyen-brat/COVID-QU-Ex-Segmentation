import time
import torch
import numpy as np
from dataset import Covid
from post_processing import post_processing
import matplotlib.pyplot as plt
from openvino.runtime import Core
import torchvision.transforms as transforms

# Load the network to OpenVINO Runtime.
onnx_path = 'weights/model_final.onnx'
ie = Core()
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
test_data = Covid("dataset/Infection Segmentation Data/", mode='test')

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

for i in range(len(test_data)):
    image, label_class, label_seg_lungs, label_seg_infected = test_data[i]
    
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    fig.add_subplot(3, 3, 2)
    plt.imshow(label_seg_lungs.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    fig.add_subplot(3, 3, 3)
    plt.imshow(label_seg_infected.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    
    image = to_tensor(image).unsqueeze(0).to('cpu').numpy()
    with torch.no_grad():
        output_class, output_seg_lungs, output_seg_infected = compiled_model_onnx(image).values()

    output_class = output_class.argmax(1)
    output_seg_lungs = (np.transpose(output_seg_lungs.argmax(1), (1, 2, 0))*255).astype('uint8')
    output_seg_infected = (np.transpose(output_seg_infected.argmax(1), (1, 2, 0))*255).astype('uint8')
      
    _, output_seg_lungs, output_seg_infected, infected_ratio, illustrate_im = post_processing(output_class, output_seg_lungs, output_seg_infected)
    
    fig.add_subplot(3, 3, 4)
    plt.imshow(output_seg_lungs,cmap='gray')
    fig.add_subplot(3, 3, 5)
    plt.imshow(output_seg_infected,cmap='gray')    
    fig.add_subplot(3, 3, 6)
    plt.imshow(illustrate_im,cmap='gray')

plt.show()