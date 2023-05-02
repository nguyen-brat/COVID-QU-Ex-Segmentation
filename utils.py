import cv2
import numpy as np

img = cv2.imread('datasets\Infection Segmentation Data\Test\COVID-19\images\covid_1579.png')
temp = img[:, :, 0][:, :, None]
temp = np.repeat(temp, 3, axis=-1)
print(temp.shape)