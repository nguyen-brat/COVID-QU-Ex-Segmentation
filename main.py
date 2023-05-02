import cv2
import glob
import torch
import numpy as np
from PIL import Image
import streamlit as st
from openvino.runtime import Core
from utils import inference
from post_processing import post_processing


img_show = st.container()
input_image = st.container()

origin_covid19_img_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\COVID-19\images\*')
infection_covid19_mask_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\COVID-19\infection masks\*')
lung_covid19_mask_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\COVID-19\lung masks\*')

non_covid_img_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Non-COVID\images\*')
non_covid_infection_mask_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Non-COVID\infection masks\*')
non_covid_lung_mask_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Non-COVID\lung masks\*')

normal_img_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Normal\images\*')
normal_infection_mask_img_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Normal\infection masks\*')
normal_lung_mask_img_paths = glob.glob(r'datasets\Infection Segmentation Data\Test\Normal\lung masks\*')

# Load model
ie = Core()
onnx_path = 'weights/model_final.onnx'
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

# Warm up the model
dummy = torch.rand(1, 3, 256, 256)
compiled_model_onnx(dummy.numpy())
###################

with img_show:
    origin_covid19_imgs = []
    infection_covid19_masks = []
    lung_covid19_masks = []
    
    non_covid_imgs = []
    non_covid_infection_masks = []
    non_covid_lung_masks = []
    
    normal_imgs = []
    normal_infection_mask_imgs = []
    normal_lung_mask_imgs = []
    
    for origin_covid19_img_path in origin_covid19_img_paths:
        origin_covid19_imgs.append(cv2.imread(origin_covid19_img_path))
    for infection_covid19_mask_path in infection_covid19_mask_paths:
        infection_covid19_masks.append(cv2.imread(infection_covid19_mask_path))
    for lung_covid19_mask_path in lung_covid19_mask_paths:
        lung_covid19_masks.append(cv2.imread(lung_covid19_mask_path))
    
    for non_covid_img_path in non_covid_img_paths:
        non_covid_imgs.append(cv2.imread(non_covid_img_path))
    for non_covid_infection_mask_path in non_covid_infection_mask_paths:
        non_covid_infection_masks.append(cv2.imread(non_covid_infection_mask_path))
    for non_covid_lung_mask_path in non_covid_lung_mask_paths:
        non_covid_lung_masks.append(cv2.imread(non_covid_lung_mask_path))
        
    for normal_img_path in normal_img_paths:
        normal_imgs.append(cv2.imread(normal_img_path))
    for normal_infection_mask_img_path in normal_infection_mask_img_paths:
        normal_infection_mask_imgs.append(cv2.imread(normal_infection_mask_img_path))
    for normal_lung_mask_img_path in normal_lung_mask_img_paths:
        normal_lung_mask_imgs.append(cv2.imread(normal_lung_mask_img_path))
    
    st.title("Some sample images")
    images = st.tabs(["Sample images 1", "Sample images 2", "Sample images 3", "Sample images 4"])
    for i, image in enumerate(images):
        with image:
            covid, non_covid, normal = st.tabs(["Covid image", "Non-Covid image", "Normal image"])
            with covid:
                output_seg_lungs, output_seg_infected, illustrate_im, inference_time = inference(origin_covid19_imgs[i], compiled_model_onnx)
                st.text(f"Inference time: {inference_time} s")
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(origin_covid19_imgs[i])
                    st.text('Output Lung segmentation')
                    st.image(output_seg_lungs)
                with infection:
                    st.text('Infection mask')
                    st.image(infection_covid19_masks[i])
                    st.text('Output Infected segmentation')
                    st.image(output_seg_infected)
                with lung:
                    st.text('Lung image')
                    st.image(lung_covid19_masks[i])
                    st.text('Final output')
                    st.image(illustrate_im)
            with non_covid:
                output_seg_lungs, output_seg_infected, illustrate_im, inference_time = inference(non_covid_imgs[i], compiled_model_onnx)
                st.text(f"Inference time: {inference_time} s")
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(non_covid_imgs[i])
                    st.text('Output Lung segmentation')
                    st.image(output_seg_lungs)
                with infection:
                    st.text('Infection mask')
                    st.image(non_covid_infection_masks[i])
                    st.text('Output Infected segmentation')
                    st.image(output_seg_infected)
                with lung:
                    st.text('Lung image')
                    st.image(non_covid_lung_masks[i])
                    st.text('Final output')
                    st.image(illustrate_im)
            with normal:
                output_seg_lungs, output_seg_infected, illustrate_im, inference_time = inference(normal_imgs[i], compiled_model_onnx)
                st.text(f"Inference time: {inference_time} s")
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(normal_imgs[i])
                    st.text('Output Lung segmentation')
                    st.image(output_seg_lungs)
                with infection:
                    st.text('Infection mask')
                    st.image(normal_infection_mask_imgs[i])
                    st.text('Output Infected segmentation')
                    st.image(output_seg_infected)
                with lung:
                    st.text('Lung image')
                    st.image(normal_lung_mask_imgs[i])
                    st.text('Final output')
                    st.image(illustrate_im)
                    
with input_image:
    st.title('Input image')
    img = st.file_uploader('Upload an image', type = ['pnj', 'jpg','png'])
    if img is not None:
        img = Image.open(img)
        st.image(img)
        img = np.array(img)
        output_seg_lungs, output_seg_infected, illustrate_im, inference_time = inference(img, compiled_model_onnx)
        st.text(f"Inference time: {inference_time} s")
        img1, img2, img3 = st.columns(3)
        with img1:
            st.text('Output Lung segmentation')
            st.image(output_seg_lungs)
        with img2:
            st.text('Output Infected segmentation')
            st.image(output_seg_infected)
        with img3:
            st.text('Final output')
            st.image(illustrate_im)