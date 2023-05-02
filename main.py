import streamlit as st
import cv2
import numpy as np
from PIL import Image
import glob
import torch
from openvino.runtime import Core
import torchvision.transforms as transforms
from post_processing import post_processing

img_show = st.container()
image = st.container()
ie = Core()

origin_covid19_img_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\COVID-19\\images\\*')
infection_covid19_mask_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\COVID-19\\infection masks\\*')
lung_covid19_mask_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\COVID-19\\lung masks\\*')

non_covid_img_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Non-COVID\\images\*')
non_covid_infection_mask_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Non-COVID\\infection masks\*')
non_covid_lung_mask_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Non-COVID\\lung masks\*')

normal_img_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Normal\\images\*')
normal_infection_mask_img_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Normal\\infection masks\*')
normal_lung_mask_img_paths = glob.glob(r'datasets\\Infection Segmentation Data\\Test\\Normal\\lung masks\*')

with img_show:
    origin_covid19_img = Image.open(origin_covid19_img_paths[0])
    origin_covid19_img = np.array(origin_covid19_img)
    infection_covid19_mask = Image.open(infection_covid19_mask_paths[0])
    infection_covid19_mask = np.array(infection_covid19_mask)
    lung_covid19_mask = Image.open(lung_covid19_mask_paths[0])
    lung_covid19_mask = np.array(lung_covid19_mask)
    
    non_covid_img = Image.open(non_covid_img_paths[0])
    non_covid_img = np.array(non_covid_img)
    non_covid_infection_mask = Image.open(non_covid_infection_mask_paths[0])
    non_covid_infection_mask = np.array(non_covid_infection_mask)
    non_covid_lung_mask = Image.open(non_covid_lung_mask_paths[0])
    non_covid_lung_mask = np.array(non_covid_lung_mask)
    
    normal_img = Image.open(normal_img_paths[0])
    normal_img = np.array(normal_img)
    normal_infection_mask_img = Image.open(normal_infection_mask_img_paths[0])
    normal_infection_mask_img = np.array(normal_infection_mask_img)
    normal_lung_mask_img = Image.open(normal_lung_mask_img_paths[0])
    normal_lung_mask_img = np.array(normal_lung_mask_img)
    
    st.title("Some sample Image and Ground truth")
    covid, not_covid, normal = st.tabs(["Covid image sample", "None covid image sample", "Normal image sample"])
    with covid:
        img1, img2, img3 = st.columns(3)
        with img1:
            st.text('Original covid-19 image')
            st.image(origin_covid19_img)
        with img2:
            st.text('Infection covid image mask')
            st.image(infection_covid19_mask)
        with img3:
            st.text('Lung image mask')
            st.image(lung_covid19_mask)
            
    with not_covid:
        img1, img2, img3 = st.columns(3)
        with img1:
            st.text('Original not covid image')
            st.image(non_covid_img)
        with img2:
            st.text('Infection covid image mask')
            st.image(non_covid_infection_mask)
        with img3:
            st.text('Lung image mask')
            st.image(non_covid_lung_mask)
            
    with normal:
        img1, img2, img3 = st.columns(3)
        with img1:
            st.text('Original covid-19 image')
            st.image(normal_img)
        with img2:
            st.text('Infection covid image mask')
            st.image(normal_infection_mask_img)
        with img3:
            st.text('Lung image mask')
            st.image(normal_lung_mask_img)

# Load model
onnx_path = 'weights/model_final.onnx'
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

# To tensor
to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

with image:
    st.title('Input image')
    im = st.file_uploader('chosse a image', type = ['pnj', 'jpg','png'])
    if im is not None:
        im = Image.open(im)
        img = np.array(im)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        st.image(im)        
        img = to_tensor(img).unsqueeze(0).to('cpu').numpy()
        with torch.no_grad():
            output_class, output_seg_lungs, output_seg_infected = compiled_model_onnx(img).values()
        output_class = output_class.argmax(1)
        output_seg_lungs = (np.transpose(output_seg_lungs.argmax(1), (1, 2, 0))*255).astype('uint8')
        output_seg_infected = (np.transpose(output_seg_infected.argmax(1), (1, 2, 0))*255).astype('uint8')
        _, output_seg_lungs, output_seg_infected, infected_ratio, illustrate_im = post_processing(output_class, output_seg_lungs, output_seg_infected)

        img1, img2, img3 = st.columns(3)
        with img1:
            st.image(output_seg_lungs)
        with img2:
            st.image(output_seg_infected)
        with img3:
            st.image(illustrate_im)