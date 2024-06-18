import imageio.v3 as iio
from nd2reader import ND2Reader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import fft2, ifft2
from scipy.interpolate import Akima1DInterpolator
from scipy import optimize

#Find the first point at which a spline interpolant for a set of correlator values descends below a certain threshold
def findRoot(xValues, yValues,threshold):
    interpolator = Akima1DInterpolator(xValues, yValues)
    def fun(arg):
        return interpolator([arg])[0]-threshold
        
    return optimize.root_scalar(fun,bracket=[min(xValues),max(xValues)]).root

def check_flow(file, channel, decay_threshold = 1/np.exp(1), min_corr_len = 25, min_fraction = 0.5, frame_stride = 1, downsample = 8, return_graph=False):
    #Conversion from a pixel to length in microns
    pix_size = 2.4859
    #Width of annuli for binning of displacement vectors
    bin_width = 2.4859
    #Width of annuli in pixels
    pixel_bin_width = np.ceil(bin_width / pix_size)
    #Length at which to stop computing correlators
    max_len = 100
    #Length in pixels at which to cut off correlators
    max_pixel_len = np.rint(max_len / pix_size)
    #Cutoff magnitude to consider a vector to be null; also helps to avoid divide-by-zero errors
    flt_tol = 1e-10

    def normalVectors(velocities):
        #Find velocity directions
        def normalize(vector):
            magnitude = np.linalg.norm(vector)
            if(magnitude > flt_tol):
                return np.array([vector[0]/magnitude,vector[1]/magnitude])
            else:
                return np.array([0,0])
                
        normals = np.zeros_like(velocities)
        for i in range(0, velocities.shape[0]):
            for j in range(0, velocities.shape[1]):
                normals[i][j] = normalize(velocities[i][j])
    
        return normals

    images = file[:,:,:,channel]

    xindices = np.arange(0, images[0].shape[0], downsample)
    yindices = np.arange(0, images[0].shape[1], downsample)

    radii = np.zeros((len(xindices),len(yindices)))
    for i in range(0,64):
        for j in range(0,64):
            radii[i][j] = np.sqrt(i**2 + j**2)

    #For each consecutive pair
    corrLens = np.zeros(int(np.floor((len(images)-1)/frame_stride)))
    pos = 0
    for i in range(0,len(images)-frame_stride,1):
        flow = cv.calcOpticalFlowFarneback(images[i], images[i+frame_stride], None, 0.5, 3, 20, 3, 5, 1.2, 0)
        directions = normalVectors(flow[xindices][:,yindices])
        dirX = directions[:,:,0]
        dirY = directions[:,:,1] 
        xFFT = fft2(dirX)
        xConv = np.real(ifft2(np.multiply(xFFT,np.conjugate(xFFT))))
        yFFT = fft2(dirY)
        yConv = np.real(ifft2(np.multiply(yFFT,np.conjugate(yFFT))))
        convSum = np.add(xConv,yConv)
        means = np.zeros(len(range(0,int(max_pixel_len),int(pixel_bin_width))))
        for i in range(0,int(max_pixel_len),int(pixel_bin_width)):
            means[i] = convSum[(radii > i -.5*pixel_bin_width)&(radii < i + .5*pixel_bin_width)].mean()/convSum[0,0]
        corrLens[pos] = pix_size*findRoot(range(0,len(means)),means,decay_threshold)
        pos = pos + 1
    
    if(len(corrLens[corrLens>min_corr_len])/len(corrLens) > min_fraction):
        verdict = "Flow detected"
    else:
        verdict = "Flow not detected"
    fig, ax = plt.subplots(figsize=(12,10))
    plt.plot(range(0,len(corrLens)),corrLens)
    
    return verdict, fig

def read_file(file_path):
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
            channels = 1
        channels = file.shape[3]
        filetype = 'tif'

    elif file_path.endswith('.nd2'):
        file_nd2 = ND2Reader(file_path)
        file = convert_to_array(file_nd2)
        channels = len(file.metadata['channels'])

    else:
        raise TypeError("Please input valid file type ('.nd2', '.tiff', '.tif')")

    return channels, file