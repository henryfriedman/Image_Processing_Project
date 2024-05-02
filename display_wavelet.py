import cv2 as cv
import numpy as np
import os
import pywt

# Path to the folder containing images
folder_path = 'triangles_vid'

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
    image = cv.imread(image_file, 0)
    coeffs = pywt.dwt2(image, 'db2')
    LL, (LH, HL, HH) = coeffs
    wavelet_image = np.concatenate((np.concatenate((LL, LH), axis=1),
                                np.concatenate((HL, HH), axis=1)),
                                axis=0)
    wavelet_image = cv.normalize(wavelet_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imshow('Wavelet Image', wavelet_image)
    #cv.imshow('Video', image)
    cv.waitKey(0)

cv.destroyAllWindow()
