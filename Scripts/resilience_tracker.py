import numpy as np

def binarize(image, threshold):
    new_image = np.zeros(image.shape)
    for i, frame in enumerate(file):
        avg_intensity = np.mean(frame)
        threshold_val = avg_intensity * (1 + R_offset)
        for y in range(file.shape[2]):
          for x in range(file.shape[1]):
              if (frame[x][y] < threshold_val):
                  new_image[i][x][y] = 0
              else:
                  new_image[i][x][y] = 1
    return new_im

def check_resilience(channel, file):
    return None