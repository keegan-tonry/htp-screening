from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from nd2reader import ND2Reader
import math, pims, yaml, gc, csv, os, glob, pickle

import numpy as np
from numpy.polynomial import Polynomial

from skimage import measure, io
from skimage.measure import label, regionprops


def track_void(image, threshold, step):
    def binarize(frame, offset_threshold):
        avg_intensity = np.mean(frame)
        threshold = avg_intensity * (1 + offset_threshold)
        new_frame = np.where(frame < threshold, 0, 1)
        return new_frame
        
    def find_largest_void(frame, find_void = True):      
        if find_void:
            frame = np.invert(frame)
        labeled, a = label(frame, connectivity= 2, return_num =True) # identify the regions of connectivity 2
        regions = regionprops(labeled) # determines the region properties of the labeled
        largest_region = max(regions, key = lambda r: r.area) # determines the region with the maximum area
        return largest_region.area # returns largest region area
    
    #def find_largest_void(frame):
        #return max(find_largest_void_mid(frame, find_void = True), find_largest_void_mid(frame, find_void = False))
    
    void_lst = []
    
    for i in range(0, len(image), step):
        new_frame = binarize(image[i], threshold)
        void_area = find_largest_void(new_frame)
        void_lst.append(void_area)
    return void_lst

def check_resilience(file, channel, R_offset=0, percent_threshold_loss = 0.9, percent_threshold_gain = 1.1, frame_step=1, frame_start_percent=0.9, frame_stop_percent=1, verbose=False):
    #Note for parameters: frame_step (stepsize) used to reduce the runtime. 
    image = file[:,:,:,channel]
    frame_initial_percent = 0.05

    fig, ax = plt.subplots(figsize = (5,5))

    # Error Checking: Empty Image
    if (image == 0).any():
        verdict = "Data not available for this channel."
        return verdict, fig
    
    largest_void_lst = track_void(image, R_offset, frame_step)
    start_index = int(len(largest_void_lst) * frame_start_percent)
    stop_index = int(len(largest_void_lst) * frame_stop_percent)
    start_initial_index = int(len(largest_void_lst)*frame_initial_percent)

    percent_gain_initial_list = np.mean(largest_void_lst[0:start_initial_index])
    percent_gain_list = np.array(largest_void_lst)/percent_gain_initial_list
    
    ax.plot(np.arange(start_index, stop_index), percent_gain_list[start_index:stop_index])
    ax.set_xlabel("Frames")
    ax.set_ylabel("Proportion of orginal void size")
    #Calculate
    
    avg_percent_change = np.mean(largest_void_lst[start_index:stop_index])/percent_gain_initial_list
    max_void_size = max(largest_void_lst)/(len(image[0,0,:])*len(image[0,:,0]))
    #print(avg_percent_change)
    #print("max void size")
    #print(max_void_size)
    #Give judgement
    if avg_percent_change >= percent_threshold_loss and avg_percent_change <= percent_threshold_gain or max_void_size < 0.10:
        verdict = "Persistance possibly detected."
        r_value = 1
    else:
        verdict = "Persistance not detected"
        r_value = 0
    
    max_void_value = int(max_void_size*10)
    
    return verdict, fig, r_value, max_void_value

def main():
    file = read_file(sys.argv[1])
    channel = read_file(sys.argv[2])
    verdict, fig, r_value, void_value = check_resilience(file, channel)

if __name__ == "__main__":
    main()
    
