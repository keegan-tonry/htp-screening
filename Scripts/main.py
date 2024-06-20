from reader import read_file
import os, csv, sys
from resilience_tracker import check_resilience
from flow_tracker import check_flow
from coarse_tracker import check_coarse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec


def execute_htp(filepath, channel_select=-1, resilience=True, flow=True, coarse=True, verbose=True):
    def check(channel, resilience, flow, coarse):
        if resilience == True:
            r, rfig = check_resilience(file, channel, verbose)
        else:
            r = "Resilience not tested"
        if flow == True:
            f, ffig = check_flow(file, channel, verbose)
        else:
            f = "Flow not tested"
        if coarse == True:
            c, cfig = check_coarse(file, channel, verbose)
        else:
            c = "Coarseness not tested."

        figpath = Path(filepath).stem + '_channel' + str(channel) + '_graphs.png'
        if verbose == True:
            fig = plt.figure()
            gs = gridspec.GridSpec(1,3)

            ax1 = rfig.axes[0]
            ax1.remove()
            ax1.figure = fig
            fig.add_axes(ax1)
            ax1.set_subplotspec(gs[0, 0])

            ax2 = ffig.axes[0]
            ax2.remove()
            ax2.figure = fig
            fig.add_axes(ax2)
            ax2.set_subplotspec(gs[0, 1])

            ax3 = cfig.axes[0]
            ax3.remove()
            ax3.figure = fig
            fig.add_axes(ax3)
            ax3.set_subplotspec(gs[0, 1])

            plt.savefig(figpath)

        plt.close(rfig)
        plt.close(ffig)
        plt.close(cfig)
            
        return [channel, r, f, c]
    
    file = read_file(filepath)

    if (isinstance(file, np.ndarray) == False):
        return None

    channels = min(file.shape)
    print('Total Channels:', channels)
    
    if (isinstance(channel_select, int) == False) or channel_select > channels:
        raise ValueError("Please give correct channel input (-1 for all channels, 0 for channel 1, etc)")
    
    rfc = []
    
    if channel_select == -1:
        for channel in range(channels):
            print('Channel:', channel)
            rfc.append(check(channel, resilience, flow, coarse))
    
    else: 
        rfc.append(check(channel_select, resilience, flow, coarse))

    return rfc

def process_directory(root_dir, channel):
    if os.path.isfile(root_dir):
        all_data = []
        file_path = root_dir
        filename = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        rfc_data = execute_htp(file_path)
        if rfc_data == None:
            raise TypeError("Please input valid file type ('.nd2', '.tiff', '.tif')")
        all_data.append([filename])
        all_data.extend(rfc_data)
        all_data.append([])

        if all_data:
            headers = ['Channel', 'Resilience', 'Flow', 'Coarseness']
            output_filepath = os.path.join(dir_name, "summary.csv")
            with open(output_filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for entry in all_data:
                    if isinstance(entry, list) and len(entry) == 1:
                        # Write the file name
                        csvwriter.writerow(entry)
                    elif entry:
                        # Write the headers if entry contains channel data
                        csvwriter.writerow(headers)
                        headers = []  # Ensure headers are only written once per file
                        csvwriter.writerow(entry)
                    else:
                        # Write an empty row
                        csvwriter.writerow([])
    
    for dirpath, dirnames, filenames in os.walk(root_dir):

        dirnames[:] = [d for d in dirnames if d != "Resilience analysis" and d != "contraction_analysis"]
        all_data = []

        for filename in filenames:
            if filename.starts_with('._'):
                continue
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            rfc_data = execute_htp(file_path, channel)
            if rfc_data == None:
                continue
            all_data.append([file_path])
            all_data.extend(rfc_data)
            all_data.append([])

        if all_data:
            headers = ['Channel', 'Resilience', 'Flow', 'Coarseness']
            output_file_path = os.path.join(root_dir, "summary.csv")
            with open(output_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for entry in all_data:
                    if isinstance(entry, list) and len(entry) == 1:
                        # Write the file name
                        csvwriter.writerow(entry)
                        csvwriter.writerow(headers)  # Write headers after the filename
                    elif entry:
                        csvwriter.writerow(entry)
                    else:
                        csvwriter.writerow([])  # Write an empty row

def main():
    dir_name = sys.argv[1]
    channel = -1 if len(sys.argv) == 2 else sys.argv[2]
    process_directory(dir_name, channel)

if __name__ == "__main__":
    main()