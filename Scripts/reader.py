import os, pims
import imageio.v3 as iio
import numpy as np
from nd2reader import ND2Reader

def read_file(file_path):
    acceptable_formats = ('.tiff', '.tif', '.nd2')
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        return None
    
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

    return file
