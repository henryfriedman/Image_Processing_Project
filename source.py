from skimage import io, color, data, draw, exposure, feature,metrics, filters, measure, morphology, util, segmentation
import numpy as np

import matplotlib.pyplot as plt
import scipy.fft as sft
import scipy.signal as sps
from PIL import Image

class Helper:
    
    def __init__(self) -> None:
        pass
    
    def get_dft(self,img):
        return sft.fft2(img)
    
    def get_dft_magnitude(self,dft):
        """
        Returns the log10-scaled magnitude of the DFT shifted to have low frequencies at the center
        Param: dft (the complex DFT of an image)
        """   
        dft_mag = np.log10(1 + np.abs(sft.fftshift(dft)))
        return dft_mag
    
    def display_img(self,img):
        """
        Display the image img both as 2D and 3D
        Param: img (a float image with values between 0 and 1)
        """
        I, J = img.shape
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img, cmap='viridis')
        ax1.set_title("Original Image")
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        j = np.arange(0, J, 1)
        i = np.arange(0, I, 1)
        jj, ii = np.meshgrid(j, i)
        dft = self.get_dft(img)
        dft_mag = self.get_dft_magnitude(dft)
        ax2.plot_surface(jj, ii, dft_mag, rstride=1, cstride=1, cmap='viridis', antialiased=False)
        ax2.set_title("3D Plot")
        ax3 = fig.add_subplot(1, 3, 3)
        inv = np.real(sft.ifft2(dft))
        ax3.imshow(inv, cmap='gray')
        ax3.set_title("Inverse DFT")
    
    def pad_image(self, sticker, image):
        # Define the size of the search box
        box_height = sticker.shape[0]
        box_width = sticker.shape[1]
        
        # Calculate the required padding
        pad_height = 0 if image.shape[0] % box_height == 0 else box_height - image.shape[0] % box_height
        pad_width = 0 if image.shape[1] % box_width == 0 else box_width - image.shape[1] % box_width
        
        # Apply padding to the image
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='reflect')
        
        return box_height, box_width, padded_image
    
    def dft_by_search_box(self,sticker,image): 
        """Get the DFT of each searching box from the image

        Args:
            sticker (np.array): the sticker
            image  (np.array): the original image

        Returns:
            dft_by_box
        """
          
        # Apply padding to the image
        box_height, box_width, padded_image = self.pad_image(sticker, image)
        
        # Initialize an array to store the DFT results
        dft_by_box = []
        
        # Loop over the padded image
        for y in range(0, padded_image.shape[0], box_height):
            for x in range(0, padded_image.shape[1], box_width):
                # Extract the search box from the padded image
                search_box = padded_image[y:y + box_height, x:x + box_width]
                
                # Compute the DFT of each search box
                search_box_dft = self.get_dft(search_box)
                search_box_dft_mag = self.get_dft_magnitude(search_box_dft)
                # Store the DFT result
                dft_by_box.append(search_box_dft_mag)
        
        return dft_by_box
    
    def create_spectrogram(self, sticker, image):
        """ Create a 2D DFT spectrogram of the image
            By combining the DFTs of each searching box

        Args:
            sticker (np.array): the sticker
            image  (np.array): the original image
        
        Returns:
            composite_dft: the complete 2D spectrogram of the image
        """
        # Get dft of each searching box from the image
        dft_by_box = self.dft_by_search_box(sticker, image)
        # Pad the image
        box_height, box_width, padded_image = self.pad_image(sticker,image)
        
        # Create an empty composite array
        composite_dft = np.zeros_like(padded_image)

        total_height = composite_dft.shape[0]
        total_width = composite_dft.shape[1]
        
        num_boxes_vertically = total_height//box_height
        num_boxes_horizontally = total_width//box_width
        
        # Populate the composite array with DFT results
        for i in range(num_boxes_vertically):
            for j in range(num_boxes_horizontally):
                # Calculate the index in the list
                idx = i * num_boxes_horizontally + j
                dft_box = dft_by_box[idx]
          
                # Place the DFT result in the correct position in the composite array
                composite_dft[
                    i * box_height: (i + 1) * box_height,
                    j * box_width: (j + 1) * box_width
                ] = dft_box
        
        return composite_dft
       
            
    def show_spectrogram(self, composite_dft, sticker, grid = False):
        if grid == False:
            plt.figure(figsize=(10, 5))
            plt.imshow(composite_dft, cmap = "gray")
            plt.colorbar()
            plt.title('2-D Spectrogram Representation')
      
        elif grid == True:
           self.show_grid(sticker,composite_dft)
           
    
    def show_grid(self, sticker, image):
        """Add grid line to the image based on the sticker size
        
        Args:
            sticker (np.array): the sticker
            image  (np.array): the target image 
        """
        # Pad the image
        box_height, box_width, padded_image = self.pad_image(sticker,image)
        num_boxes_vertically = padded_image.shape[0]//box_height
        num_boxes_horizontally = padded_image.shape[1]//box_width
        
        plt.figure(figsize=(10, 5))
        plt.imshow(padded_image, cmap='gray')
        plt.colorbar()
        plt.title('Padded Image with Grid Lines')
        
        # Calculate the boundaries for the vertical lines (edges of the boxes)
        for i in range(1, num_boxes_horizontally):
            plt.axvline(x=i * box_width, color='orange', linestyle='-', linewidth=2)

        # Calculate the boundaries for the horizontal lines (edges of the boxes)
        for j in range(1, num_boxes_vertically):
            plt.axhline(y=j * box_height, color='orange', linestyle='-', linewidth=2)

        
    def find_location_mse(self, sticker, image):
        """Find the location of the sticker based on mse value

        Args:
            sticker (np.array): the sticker
            image (np.array): the target image

        Returns:
            best_box: Target box number
            min_mse: The minimum value of mse
            sorted_box_idx: Sorted box numbers based on mse values from min to max 
            sorted_mse: Sorted mse values from min to max
        """
        
        sticker_dft = self.get_dft(sticker)
        sticker_dft_mag = self.get_dft_magnitude(sticker_dft)
        mse_results = []
        best_box = -1
        min_mse = np.inf
        
        dft_by_box = self.dft_by_search_box(sticker, image)
        i = 0
        for dft_mag in dft_by_box:
            # Compute the MSE between the two magnitude spectra
            mse = np.mean((dft_mag - sticker_dft_mag) ** 2)
            mse_results.append(mse)
           
            if(mse < min_mse):
                min_mse = mse
                best_box = i+1
            i += 1
        
        sorted_mse_idx = np.argsort(mse_results)
        sorted_box_idx = [i + 1 for i in sorted_mse_idx]
        sorted_mse = np.sort(mse_results)
        
        return best_box, min_mse, sorted_box_idx, sorted_mse
            
            
    def find_location_ssim(self, sticker, image):
        
        sticker_dft = self.get_dft(sticker)
        sticker_dft_mag = self.get_dft_magnitude(sticker_dft)
        ssim_results = []
        best_box = -1
        max_ssim = 0
        
        dft_by_box = self.dft_by_search_box(sticker, image)
        i = 0
        for dft_mag in dft_by_box:
            # Compute the MSE between the two magnitude spectra
            ssim = metrics.structural_similarity(dft_mag, sticker_dft_mag, data_range = dft_mag.max()-dft_mag.min())
            ssim_results.append(ssim)
           
            if(ssim > max_ssim):
                max_ssim = ssim
                best_box = i+1
            i += 1
        
        sorted_ssim_idx = np.argsort(ssim_results)
        # Add 1 to all index to represent box number beginning from 1
        sorted_box_idx = [i + 1 for i in sorted_ssim_idx]
        sorted_ssim = np.sort(ssim_results)
        
        return best_box, max_ssim, sorted_box_idx, sorted_ssim
    
        
        
