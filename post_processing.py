import cv2
import numpy as np

def noise_remove(im):
    kernel = np.ones((5, 5), np.uint8)
    im_re = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel) 
    contours, hierarchy = cv2.findContours(im_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <10:
            cv2.fillPoly(im_re, pts =[cnt], color=(0))
    return im_re

def post_processing(outputs_classification, output_lungs, output_infected):
    class_revert_cvt = { 0:'Normal',1: 'COVID-19',2:'Non-COVID'}
    
    if outputs_classification.tolist()[0] == 1:
        output_infected = noise_remove(output_infected)
        output_lungs = noise_remove(output_lungs)
        illustrate_im = cv2.cvtColor(output_lungs.copy(),cv2.COLOR_GRAY2RGB)
        output_infected = cv2.bitwise_and(output_infected,output_lungs, mask = None)
        infected_ratio = 100*np.count_nonzero(output_infected)/(np.count_nonzero(output_lungs)+1e-5)
        outputs_classification = class_revert_cvt[outputs_classification.tolist()[0]]
        
        contours, hierarchy = cv2.findContours(output_infected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(illustrate_im, contours, -1, (0, 255, 0), 1)
        illustrate_im = cv2.putText(illustrate_im, f'Infected ratio: {infected_ratio:.4f}%',(5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        illustrate_im = cv2.putText(illustrate_im, f'Predicted: {outputs_classification}',(5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        return outputs_classification, output_lungs, output_infected, infected_ratio, illustrate_im
    else:
        output_infected = np.zeros_like(output_infected) 
        output_lungs = noise_remove(output_lungs)
        illustrate_im = cv2.cvtColor(output_lungs.copy(),cv2.COLOR_GRAY2RGB)
        outputs_classification = class_revert_cvt[outputs_classification.tolist()[0]]
        illustrate_im = cv2.putText(illustrate_im, f'Infected ratio: 0%',(5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        illustrate_im = cv2.putText(illustrate_im, f'Predicted: {outputs_classification}',(5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        return outputs_classification, output_lungs, output_infected, 0, illustrate_im