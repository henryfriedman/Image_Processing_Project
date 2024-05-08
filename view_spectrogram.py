import cv2 as cv
import numpy as np
import os
import scipy.fft as sft
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import Image
from source import Helper
h = Helper()


def view_spectrogram(sticker,folder_path):

    # Get list of image files in the folder
    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("No image files found in the folder.")
        exit()

    # Load the first image to get dimensions
    first_image = cv.imread(image_files[0], 0)
    height, width = first_image.shape

    for image_file in image_files:
        # Read the image
        image = io.imread(image_file, as_gray=True)
        # Convert image to numpy array
        image = np.array(image)

        # Perform spectrogram
        composite_dft= h.create_spectrogram(sticker, image)
        _,_,pad_image = h.pad_image(sticker,image)
        spectrogram = cv.convertScaleAbs(composite_dft)
        concatenated_image = np.hstack((spectrogram, pad_image))
        
        cv.namedWindow('Spectrogram', cv.WINDOW_NORMAL)
        cv.resizeWindow('Spectrogram', 1200, 720)
        cv.imshow('Spectrogram',concatenated_image)
    
        cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":
    
    sticker = io.imread('stickers/right_triangle.png', as_gray = True)
    # Path to the folder containing images
    folder_path = '/Users/sallyliu/Desktop/CSCI 452/Project/Image_Processing_Project/rotating_30_60_90_chirps_video'
    view_spectrogram(sticker, folder_path)