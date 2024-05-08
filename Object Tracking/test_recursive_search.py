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
        
        bounds = (0, 0, image.shape[0], image.shape[1])
        selected_part, selected_bounds = h.find_location_recursive(sticker,image, bounds, min_mse = float('inf'))
    
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 1200, 720)

        y,x,_,_ = selected_bounds
        img = image.copy()
        part_height = selected_part.shape[0]
        part_width = selected_part.shape[1]
        image_window = cv.rectangle(image,(x,y),(x+part_width, y+part_height), (0,255,0), 4)
        cv.imshow('Image', image_window)
        cv.waitKey(0)

    cv.destroyAllWindows()
    
    
if __name__ == "__main__":
    
    sticker = io.imread('stickers/sticker6.png', as_gray = True)
    # Path to the folder containing images
    folder_path = '/Users/sallyliu/Desktop/CSCI 452/Project/Image_Processing_Project/video_circle'
    tracking(sticker, folder_path)