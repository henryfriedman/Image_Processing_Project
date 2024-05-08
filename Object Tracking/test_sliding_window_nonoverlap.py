import cv2 as cv
import numpy as np
import os
import scipy.fft as sft
from skimage import io, color, transform
import matplotlib.pyplot as plt
from PIL import Image
from source import Helper
h = Helper()

def tracking(sticker, folder_path):
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
        _,_,padded_img =  h.pad_image(sticker,image)
        padded_img= np.array(padded_img)
        
        best_position, best_match_box = h.find_location_sliding_window(sticker,padded_img, step_x = sticker.shape[1], step_y = sticker.shape[0])
    
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 1200, 720)

        y,x = best_position
        img = padded_img.copy()
        sticker_height = sticker.shape[0]
        sticker_width = sticker.shape[1]
        image_window = cv.rectangle(padded_img,(x,y),(x+sticker_width, y+sticker_height), (0,255,0), 4)
        cv.imshow('Image', image_window)
    
        cv.waitKey(0)

    cv.destroyAllWindows()
    
    
if __name__ == "__main__":
    
    sticker = io.imread('stickers/sticker6.png', as_gray = True)
    # Path to the folder containing images
    folder_path = '/Users/sallyliu/Desktop/CSCI 452/Project/Image_Processing_Project/video_circle'
    tracking(sticker, folder_path)