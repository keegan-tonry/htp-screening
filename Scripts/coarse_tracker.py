import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
# from numpy.polynomial import Polynomial, polyroots
from nd2reader import ND2Reader
from scipy.interpolate import splrep, sproot, BSpline
import scipy
import scipy.signal as signal

def calculate_mean_mode(frame):
    mean_intensity = np.mean(frame)
    mode_intensity = mode(frame.flatten(), keepdims=False).mode
    return mean_intensity, mode_intensity

def analyze_frames(image, threshold_percentage):
    first_frame = image[0]
    last_frame = image[len(image) - 1]
        
    # calculate mean and mode for first frame
    mean_first_frame, mode_first_frame = calculate_mean_mode(first_frame)
    # calculate mean and mode for last frame
    mean_last_frame, mode_last_frame = calculate_mean_mode(last_frame)            
                   
     # Calculate the difference between mean and mode for both frames
    diff_first_frame = abs(mean_first_frame - mode_first_frame)
    diff_last_frame = abs(mean_last_frame - mode_last_frame)

    # Calculate the percentage increase
    percentage_increase = ((diff_last_frame - diff_first_frame) / diff_first_frame) * 100

    # Output "coarsening" if the difference for the last frame is x% larger than for the first frame
    if percentage_increase > threshold_percentage:
        return "Coarsening detected."
    else:
        return "Coarsening not detected"

def check_coarse(file, channel, first_frame = 0, last_frame = None, threshold_percentage = 0.7, verbose=False):
    extrema_bounds_list = []
    extrema_bounds_idx_list = []
    areas_list = []
    extrema_len_list = []
    extrema_height_list = []

    im = file[:,:,:,channel]

    # Set last_frame to last frame of movie if unspecified
    if last_frame == None: 
        last_frame = len(im) - 1

    fig, ax = plt.subplots(figsize=(5,5))

    if (im == 0).all(): # If image is blank, then end program early
        verdict = "Data not available for this channel."
        return verdict, fig

    max_px_intensity = 1.1*np.max(im)
    min_px_intensity = np.min(im)
    bins_width = 3
    poly_deg = 40
    poly_len = 10000
    
    near_zero_limit = 0.01
    minimum_area = 0.010
    
    i_frame_data = im[first_frame] - min_px_intensity
    f_frame_data = im[last_frame] - min_px_intensity
    f_norm = np.mean(i_frame_data) / np.mean(f_frame_data)
    f_frame_data = f_norm * f_frame_data

    fig, ax = plt.subplots(figsize=(5,5))
    set_bins = np.arange(0, max_px_intensity, f_norm * bins_width)
    bins_num = len(set_bins)
    i_count, bins = np.histogram(i_frame_data.flatten(), bins=set_bins, density=True)
    f_count, bins = np.histogram(f_frame_data.flatten(), bins=set_bins, density=True)
    center_bins = (bins[1] - bins[0])/2
    plt_bins = bins[0:-1] + center_bins
    ax.plot(plt_bins, i_count, '^-', ms=4, c='darkred', alpha=0.2, label= "frame " + str(first_frame+1)+" dist")
    ax.plot(plt_bins, f_count, 'v-', ms=4, c='darkorange',   alpha=0.2, label= "frame " + str(last_frame+1)+" dist")
    
    count_diff = f_count - i_count
    ax.plot(plt_bins, count_diff, 'D-', ms=2, c='red', label = "difference btwn")
    
    p_cutoff = 1e-5
    initial_spline = splrep(plt_bins, i_count, s = 0.00005)
    in_cutoff = np.max(np.where(BSpline(*initial_spline)(plt_bins) >= p_cutoff))
    ax.axvline(x = in_cutoff)
    minimum_area = 0.01 * float(BSpline.basis_element(initial_spline[0]).integrate(0, in_cutoff))

    initial_spline = splrep(plt_bins, i_count, s = 0.00005)

    ax.plot(plt_bins, i_count, '^-', ms=4, c='darkred', alpha=0.2, label= "frame " + str(first_frame+1)+" dist")
    ax.plot(plt_bins, f_count, 'v-', ms=4, c='darkorange',   alpha=0.2, label= "frame " + str(last_frame+1)+" dist")
    count_diff = f_count - i_count
    ax.plot(plt_bins, count_diff, 'D-', ms=2, c='red', label = "difference btwn")
    
    ax.plot(plt_bins, BSpline(*initial_spline)(plt_bins), c='magenta', label='initial_fit')
    
    
    # ### get range for local extrema of interest ###
    
    spline = splrep(plt_bins, f_count - i_count, s = 0.00005)
    t = spline[0]
    
    cumulative_count_diff = np.zeros_like(count_diff)
    for i in range(len(cumulative_count_diff)):
        cumulative_count_diff[i] = np.sum(count_diff[:i])
    filtered_ccd = scipy.ndimage.gaussian_filter1d(cumulative_count_diff, 8)
    ax.plot(filtered_ccd, c = 'darkgreen', label = 'CDF')
    
    peaks_max = signal.argrelextrema(filtered_ccd, np.greater, order = 20)
    peaks_min = signal.argrelextrema(filtered_ccd, np.less, order = 20)
    areas = np.append(np.abs(filtered_ccd[peaks_max][0]), np.abs(filtered_ccd[peaks_max][0] - filtered_ccd[peaks_min][0]))

    verdict = analyze_frames(im, threshold_percentage)

    ax.axhline(0, color='dimgray', alpha=0.6)
    ax.set_xlabel("Pixel intensity value")
    ax.set_ylabel("Probability")
    ax.set_xlim(0,max_px_intensity + 5)
    ax.legend()
    
    return verdict, fig, areas

def main():
    file = read_file(sys.argv[1])
    channel = sys.argv[2]
    results = check_coarse(file, channel)

if __name__ == "__main__":
    main()
