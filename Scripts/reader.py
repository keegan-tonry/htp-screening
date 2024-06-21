import os, pims
import imageio.v3 as iio
import numpy as np
from nd2reader import ND2Reader

def read_file(file_path, accept_dim = False):
    acceptable_formats = ('.tiff', '.tif', '.nd2')
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        return None

    def check_first_frame_dim(file):
        min_intensity = np.min(file[0])
        mean_intensity = np.mean(file[0])
        if 0.5 * mean_intensity <= min_intensity:
            return True
        else:
            return False

    
    def bleach_correction(im):
        min_px_intensity = np.min(im)
        num_frames=len(im) #num frames in video
        corrected_frames=np.zeros_like(im) #empty for storing corrected values same number as original frames
        #MEAN INTENSITY OF FIRST FRAME
        i_frame_data = im[0] - min_px_intensity #adjusted intensity values of the first frame where min intensity has been normalized to ~zero
        initial_mean_intensity = np.mean(i_frame_data)
        #bleach correction for each frame
        for i in range(num_frames):
            frame_data=im[i] - min_px_intensity
            #find normalization factor relative to the first frame
            normalization_factor=initial_mean_intensity / np.mean(frame_data)
            corrected_frames[i] = normalization_factor * frame_data + min_px_intensity
        return corrected_frames

    
    def convert_to_array(file):
        num_images = file.sizes['t']
        num_channels = file.sizes['c']
        height = file.metadata['height']
        width = file.metadata['width']
        images = np.zeros((num_images, height, width, num_channels))
        for i in range(num_channels):
            for j in range(num_images):
                frame = np.array(file.get_frame_2D(c=i, t=j))
                images[j, :, :, i] = frame
        return images
    
    if file_path.endswith('.tiff') or file_path.endswith('.tif'):
        file = iio.imread(file_path)
        if len(file.shape) == 3:
            file = np.reshape(file, (file.shape + (1,)))    
        channels = file.shape[3]

    elif file_path.endswith('.nd2'):
        try:
            file_nd2 = ND2Reader(file_path)
        except:
            return None
        file = convert_to_array(file_nd2)
        channels = len(file_nd2.metadata['channels'])

    file = bleach_correction(file)
    if accept_dim == False and check_first_frame_dim(file) == True:
        print(file_path + 'is too dim, skipping to next file...')
        return None
    else:
        return file