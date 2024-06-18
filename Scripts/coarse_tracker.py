import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from numpy.polynomial import Polynomial
from nd2reader import ND2Reader

### function to find the corresponding x-value with the y-value nearest to a given value
def find_nearest(yarray, xarray, value):
    yarray = np.asarray(yarray)
    xarray = np.asarray(xarray)
    idx = (np.abs(yarray - value)).argmin()
    return xarray[idx]

### function to find the index of the x-value nearest to a given value
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

### function to find the area under the curve bounded by x-value *indices* i_idx and f_idx
def area_under_curve(x_vals, y_vals, i_idx, f_idx):
    total = 0
    dx = (x_vals[f_idx] - x_vals[i_idx]) / (len(x_vals[i_idx:f_idx]))
    y_vals_in_range = y_vals[i_idx:f_idx]
    for i in range(len(x_vals[i_idx:f_idx])):
        total += (y_vals_in_range[i] * dx)
    return total

### function to find two x-values nearest to where a curve crosses the x-axis, (+y) --> (-y) or (-y) --> (+y)
def zero_intersect(y_arr, x_arr):
    x_ranges_list = []
    zero_cross_range_idx1 = np.where(np.diff(np.sign(y_arr)))[0]
    zero_cross_range_idx2 = (np.where(np.diff(np.sign(y_arr)))[0])+1
    for i in range(len(zero_cross_range_idx2)):
        idx1 = zero_cross_range_idx1[i]
        idx2 = zero_cross_range_idx2[i]
        x_range = [x_arr[idx1], x_arr[idx2]]
        x_ranges_list.append(x_range)
    return x_ranges_list

### function to specify the nearest x-value given the range found in the 'zero_intersect' function
def zero_intersect_in_range(x_arr, y_arr, x1, x2, near_zero_limit, verbose):
    i_idx = find_nearest_idx(x_arr, x1)
    f_idx = find_nearest_idx(x_arr, x2)
    y_max_idx = y_arr.argmax()
    y_max = y_arr[y_max_idx]
    close_xvals = []
    corresponding_yvals = []
    found = False
    if verbose == True:
        print("searching for zero intersect in x-val range [%i, %i]"%(x_arr[i_idx], x_arr[f_idx]))
    for i in range(len(y_arr[i_idx:f_idx])):
        if (np.abs(0-y_arr[i+i_idx]))<near_zero_limit:
            found = True
            if verbose == True:
                print("diff near zero at: %4.2f, diff = %1.1e" %(x_arr[i+i_idx], 0-y_arr[i+i_idx]))
            close_xvals.append(x_arr[i+i_idx])
            corresponding_yvals.append(y_arr[i+i_idx])
    if len(close_xvals)!=0:
        nearest = find_nearest(corresponding_yvals, close_xvals, 0)
        idx = find_nearest_idx(x_arr, nearest)
        return nearest, idx
    elif found == False:
        if verbose == True:
            print("none found")
        return 0, 0


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

    fig, ax = plt.subplots(figsize=(20,20))

    if (im == 0).all(): # If image is blank, then end program early
        verdict = "Data not available for this channel."
        return verdict, fig

    max_px_intensity = 1.1*np.max(im)
    bins_width = 3
    poly_deg = 5
    poly_len = 10000

    near_zero_limit = 0.01
    minimum_area = 0.022

    i_frame_data = im[first_frame]
    f_frame_data = im[last_frame]

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

    ### polynomial fit for a clean line ###
    poly_series = Polynomial.fit(plt_bins, count_diff, deg=poly_deg)  # 5th order polynomial curve fit
    x_poly, y_poly = poly_series.linspace(poly_len)
    ax.plot(x_poly, y_poly, c='k', alpha=0.6, label="polyfit")

    ### get range for local extrema of interest ###
    x_ranges_list = zero_intersect(y_poly, x_poly)

    j = 0
    c_list = ['cyan', 'magenta', 'lime', 'blue', 'deeppink', 'seagreen', 'red', 'green',
              'yellow','orange','purple','pink','brown','gray','black','white',]
    for i in range(len(x_ranges_list)-1):
        x1, x1_idx = zero_intersect_in_range(x_poly, y_poly, x_ranges_list[i][0], x_ranges_list[i][1],
                                             near_zero_limit, verbose)
        if verbose == True:
            print(" --> first_zero_x = %4.2f, at index [%i] \n" % (x1, x1_idx))

        x2, x2_idx = zero_intersect_in_range(x_poly, y_poly, x_ranges_list[i+1][0], x_ranges_list[i+1][1],
                                             near_zero_limit, verbose)
        if verbose == True:
            print(" --> second_zero_x = %4.2f, at index [%i] \n" %(x2, x2_idx))

        area = area_under_curve(x_poly, y_poly, x1_idx, x2_idx)
        if verbose == True:
            print("* area check for x range [%4.2f, %4.2f] = %1.3e * \n" %(x1_idx, x2_idx, area))
        if np.abs(area) > minimum_area:

            ax.axvline(x2, color = c_list[j], linestyle = '--', alpha = 0.7)
            ax.axvline(x1, color = c_list[j], linestyle = ':') #, alpha = 0.6)
            extrema_bounds_list.append([round(float(x1), 2), round(float(x2), 2)])
            extrema_bounds_idx_list.append([x1_idx, x2_idx])

    ### find area above (or below?) extrema of interest ###
            areas_list.append(area.round(6))
            ax.fill_between(x_poly[x1_idx:x2_idx], y_poly[x1_idx:x2_idx],
                            color= c_list[j], alpha = 0.8, label="area %i = %1.3e" %((j+1), area))
            if area < 0:
                extrema_h = (np.sort(y_poly[x1_idx:x2_idx]))[0]
            else:
                extrema_h = (np.sort(y_poly[x1_idx:x2_idx]))[-1]
            ax.axhline(extrema_h, color = c_list[j], linestyle = '--', alpha = 0.7)
            extrema_height_list.append(extrema_h.round(8))
            extrema_len_list.append((x2 - x1).round(1))
            j += 1

    results = [bins_num, extrema_bounds_list, extrema_bounds_idx_list, extrema_len_list, extrema_height_list, areas_list]

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
    #return title, (np.array(results)).round(4)
    return verdict, fig

def main():
    file = read_file(sys.argv[1])
    channel = sys.argv[2]
    results = check_flow(file, channel, filetype)

if __name__ == "__main__":
    main()