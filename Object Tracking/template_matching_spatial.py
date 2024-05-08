import cv2 as cv
import numpy as np
import os
import scipy.fft as sft
from skimage import io, color, transform, feature
import matplotlib.pyplot as plt
from PIL import Image
from source import Helper
h = Helper()

def template_matching(sticker, folder_path):
    # Get list of image files in the folder
    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("No image files found in the folder.")
        exit()
        
    for image_file in image_files:
        # Read the image
        image = io.imread(image_file, as_gray=True)
        # Convert image to numpy array
        image = np.array(image)
        
        result = feature.match_template(image, sticker)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        
        # highlight matched region
        hsticker, wsticker = sticker.shape
        image_window = cv.rectangle(image,(x,y),(x+wsticker, y+hsticker), (0,255,0), 4)
        cv.imshow('Image', image_window)
        #rect = plt.Rectangle((x, y), wsticker, hsticker, edgecolor='r', facecolor='none')
        
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 1200, 720)
       
        cv.imshow('Image', image_window)
        cv.waitKey(0)

    cv.destroyAllWindows()
    
    
if __name__ == "__main__":
    
    sticker = io.imread('stickers/sticker6.png', as_gray = True)
    # Path to the folder containing images
    folder_path = '/Users/sallyliu/Desktop/CSCI 452/Project/Image_Processing_Project/video_circle'
    template_matching(sticker, folder_path)