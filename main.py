import streamlit as st
import cv2
import numpy as np
import cv2
import glob
import torch
from openvino.runtime import Core
import torchvision.transforms as transforms
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
    
    st.title("Some sample Image")
    images = st.tabs(["Sample image 1", "Sample image 2", "Sample image 3", "sample image 4"])
    for i, image in enumerate(images):
        with image:
            covid, non_covid, normal = st.tabs(["Covid image", "Non vovid image", "Normal image"])
            with covid:
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(origin_covid19_imgs[i])
                with infection:
                    st.text('Infection mask')
                    st.image(infection_covid19_masks[i])
                with lung:
                    st.text('Lung image')
                    st.image(lung_covid19_masks[i])
            with non_covid:
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(non_covid_imgs[i])
                with infection:
                    st.text('Infection mask')
                    st.image(non_covid_infection_masks[i])
                with lung:
                    st.text('Lung image')
                    st.image(non_covid_lung_masks[i])
            with normal:
                original, infection, lung = st.columns(3)
                with original:
                    st.text('Original Lung image')
                    st.image(normal_imgs[i])
                with infection:
                    st.text('Infection mask')
                    st.image(normal_infection_mask_imgs[i])
                with lung:
                    st.text('Lung image')
                    st.image(normal_lung_mask_imgs[i])
                    
                    
with input_image:
    st.title('Input image')
    im = st.file_uploader('chosse a image', type = ['pnj', 'jpg','png'])
    if im is not None:
        im = cv2.imread(im)
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