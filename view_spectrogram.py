import cv2 as cv
import numpy as np
import os
from skimage import io, color
from PIL import Image
from source import Helper

h = Helper()
sticker = np.asarray(Image.open('Zeyi_Draft/triangle.png'))
gray_sticker = color.rgba2rgb(sticker)
gray_sticker = color.rgb2gray(gray_sticker)

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

    # Perform spectrogram
    result= h.create_spectrogram(gray_sticker, image, show = False)
    spectrogram = result["Composite DFT"]
    _,_,pad_image = h.pad_image(gray_sticker,image)
    concatenated_image = np.hstack((spectrogram, pad_image))
    
    cv.namedWindow('Spectrogram', cv.WINDOW_NORMAL)
    cv.resizeWindow('Spectrogram', 1200, 720)

    cv.imshow('Spectrogram', concatenated_image)
    cv.waitKey(0)

cv.destroyAllWindows()
