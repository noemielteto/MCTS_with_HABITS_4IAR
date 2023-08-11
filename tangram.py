import math
import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

class TangramShape:
    def __init__(self, pixels, name):
        self.pixels = pixels
        self.name   = name

    def __repr__(self):
        return 'Tangram instance ' + str([self.name])

    def getDisplacedPixels(self, x, y, angle):

        pixels = copy.deepcopy(self.pixels)

        if angle:

            filled_coords = np.where(pixels)
            x_border = min(filled_coords[0])
            y_border = max(filled_coords[1])
            pixels = pixels[x_border:, :y_border+1]
            pixels = scipy.ndimage.rotate(pixels,angle=angle)

        if y:
            pixels = np.roll(pixels, -y, axis=0)
        if x:
            pixels = np.roll(pixels, x, axis=1)

        return pixels

    def place(self, x, y, angle):
        self.pixels = self.getDisplacedPixels(x, y, angle)

    def intersects(self, pixels):
        return np.max(self.pixels + pixels)==2

    def draw(self, color=None, pause=False, save_name=False, scale=1):

        arr = copy.deepcopy(self.pixels)
        row_mask = np.all(arr == 0, axis=1)
        col_mask = np.all(arr == 0, axis=0)
        # select the rows and columns that are not all zero
        arr = arr[~row_mask][:, ~col_mask]
        # create masked array so that empty space shows up transparent
        mask = np.ma.masked_where(arr == 0, arr)

        f,ax = plt.subplots(1, 1, figsize=(mask.shape[1]*scale, mask.shape[0]*scale))

        if color is not None:
            ax.imshow(mask, cmap=ListedColormap([color]), interpolation='none')
        else:
            ax.imshow(mask)

        ax.axis('off')
        plt.tight_layout()

        if pause:
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        elif save_name:
            plt.savefig(save_name, transparent=True, dpi=10)
            plt.close('all')

        else:
            plt.show()

class TangramSilhouette:
    def __init__(self, pixels, color='grey'):
        self.pixels = pixels
        self.color = color

    def outOfBounds(self, shape_pixels):
        if np.isin(2, self.pixels + shape_pixels*2):
            return True
        else:
            return False

    def draw(self, save_name):
        f,ax = plt.subplots(1,1)
        ax.imshow(self.pixels)
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()
