import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
# from numpy.polynomial import Polynomial, polyroots
from nd2reader import ND2Reader
from scipy.interpolate import splrep, sproot, BSpline

def calculate_mean_mode(frame):
    mean_intensity = np.mean(frame)
    mode_intensity = mode(frame.flatten(), keepdims=False).mode
    return mean_intensity, mode_intensity

def analyze_frames(image, threshold_percentage, start_index, end_index):
    first_frame = image[start_index]
    last_frame = image[end_index]

    mean_first_frame, mode_first_frame = calculate_mean_mode(first_frame)
    mean_last_frame, mode_last_frame = calculate_mean_mode(last_frame) 

    diff_first_frame = abs(mean_first_frame - mode_first_frame)
    diff_last_frame = abs(mean_last_frame - mode_last_frame)

    percentage_increase = ((diff_first_frame - diff_last_frame) / diff_last_frame) * 100

    if percentage_increase > threshold_percentage:
        return "dim"
    else:
        return "not dim"

def check_coarse(file, channel, first_frame = 0, last_frame = None, verbose=False):
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
    set_bins = np.arange(0, max_px_intensity, bins_width)
    bins_num = len(set_bins)
    i_count, bins = np.histogram(i_frame_data.flatten(), bins=set_bins, density=True)
    f_count, bins = np.histogram(f_frame_data.flatten(), bins=set_bins, density=True)
    center_bins = (bins[1] - bins[0])/2
    plt_bins = bins[0:-1] + center_bins
    ax.plot(plt_bins, i_count, '^-', ms=4, c='darkred', alpha=0.2, label= "frame " + str(first_frame+1)+" dist")
    ax.plot(plt_bins, f_count, 'v-', ms=4, c='darkorange',   alpha=0.2, label= "frame " + str(last_frame+1)+" dist")
    
    count_diff = f_count - i_count
    ax.plot(plt_bins, count_diff, 'D-', ms=2, c='red', label = "difference btwn")
    
    prob_cutoff = np.max(np.where(i_count >= 1e-4))
    ax.axvline(x = prob_cutoff)
    initial_spline = splrep(plt_bins, i_count, s = 0.00005)
    minimum_area = 0.01 * float(BSpline.basis_element(initial_spline[0]).integrate(0, prob_cutoff))
    
    ax.plot(plt_bins, BSpline(*initial_spline)(plt_bins), c='magenta', label='initial_fit')
    
    i_count = i_count[i_count < prob_cutoff]
    f_count = f_count[f_count < prob_cutoff]
    
    # ### get range for local extrema of interest ###
    
    spline = splrep(plt_bins, f_count - i_count, s = 0.00005)
    t = spline[0]
    ax.plot(plt_bins, BSpline(*spline)(plt_bins), c='blue', label='diff_fit')
    
    areas = []
    roots = np.array(sproot(spline))
    roots = roots[roots < prob_cutoff]
    bas = BSpline.basis_element(t)
    area_1 = 0 if len(roots) == 0 else float(bas.integrate(a=0, b=roots[0]))
    if area_1 > minimum_area: 
        areas.append(area_1)
    for i in range(len(roots) - 1):
        area = float(bas.integrate(roots[i], roots[i+1]))
        if area > minimum_area:
            areas.append(area)

    if len(areas) >= 2:
        verdict = "Coarsening likely detected."

    else:
        verdict = "Coarsening not detected."

    ax.axhline(0, color='dimgray', alpha=0.6)
    # plt.title(title)
    ax.set_xlabel("Pixel intensity value")
    ax.set_ylabel("Probability")
    ax.set_xlim(0,max_px_intensity + 5)
    ax.legend()
    return verdict, fig

    # j = 0
    # c_list = ['cyan', 'magenta', 'lime', 'blue', 'deeppink', 'seagreen', 'red', 'green',
    #           'yellow','orange','purple','pink','brown','gray','black','white',]
    # for i in range(len(x_ranges_list)-1):
    #     x1, x1_idx = zero_intersect_in_range(x_poly, y_poly, x_ranges_list[i][0], x_ranges_list[i][1],
    #                                          near_zero_limit, verbose)
    #     x2, x2_idx = zero_intersect_in_range(x_poly, y_poly, x_ranges_list[i+1][0], x_ranges_list[i+1][1],
    #                                          near_zero_limit, verbose)
    #     area = area_under_curve(x_poly, y_poly, x1_idx, x2_idx)
    #     if np.abs(area) > minimum_area:

    #         ax.axvline(x2, color = c_list[j], linestyle = '--', alpha = 0.7)
    #         ax.axvline(x1, color = c_list[j], linestyle = ':') #, alpha = 0.6)
    #         extrema_bounds_list.append([round(float(x1), 2), round(float(x2), 2)])
    #         extrema_bounds_idx_list.append([x1_idx, x2_idx])

    # ### find area above (or below?) extrema of interest ###
    #         areas_list.append(area.round(6))
    #         ax.fill_between(x_poly[x1_idx:x2_idx], y_poly[x1_idx:x2_idx],
    #                         color= c_list[j], alpha = 0.8, label="area %i = %1.3e" %((j+1), area))
    #         if area < 0:
    #             extrema_h = (np.sort(y_poly[x1_idx:x2_idx]))[0]
    #         else:
    #             extrema_h = (np.sort(y_poly[x1_idx:x2_idx]))[-1]
    #         ax.axhline(extrema_h, color = c_list[j], linestyle = '--', alpha = 0.7)
    #         extrema_height_list.append(extrema_h.round(8))
    #         extrema_len_list.append((x2 - x1).round(1))
    #         j += 1

    # results = [bins_num, extrema_bounds_list, extrema_bounds_idx_list, extrema_len_list, extrema_height_list, areas_list]

    if ((len(areas_list) >= 2) and (areas_list[-1] > 0)):
        verdict = "Coarsening likely detected."
    else:
        verdict = "Coarsening not detected."

    ax.axhline(0, color='dimgray', alpha=0.6)
    # plt.title(title)
    ax.set_xlabel("Pixel intensity value")
    ax.set_ylabel("Probability")
    ax.set_xlim(0,max_px_intensity + 5)
    ax.legend()
    return verdict, fig

def main():
    file = read_file(sys.argv[1])
    channel = sys.argv[2]
    results = check_coarse(file, channel)

if __name__ == "__main__":
    main()
