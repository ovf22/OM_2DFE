import matplotlib.pyplot as plt
import numpy as np
import sys


def image_ROI_crop_center(data_crop, x_center=0, y_center=0,
                          width=100, height=100, cropping=False):
    """ Crop the tomography data according to the cutting parameters

    Parameters
    ----------
    data_crop : Tomography data file [Array] \n
    x_left : X-coordinate on left side of image \n
    y_bottom : X-coordinate on left side of image \n
    width : Width of cropped image. Default is 100 \n
    height : Height of cropped image. Default is 100 \n
    cropping : Used for activating cropping of ROI, if True
    then the image is cropped to the designated coordinates
    and dimensions. Default is False \n

    Returns
    -------
    data : Tomography data from ROI [Array]

    """

    no_slicing = slice(None)
    data_shape = data_crop.shape
    cut_arr = np.array([height, width])

    # If the cropping dimensions are bigger than the img data set,
    # then kill the script
    if np.any(cut_arr > data_shape):
        print('Too much data was removed. Adjust cut parameters')
        sys.exit()

    # If any cutting parameters are chosen, then crop the volume accordingly
    # Else return the full data set
    if np.any(cut_arr > 0):
        print('Cropping data')
        if cropping:
            x_left = x_center - width//2
            x_right = x_center + width//2
            x_slice = slice(x_left, x_right)
        else:
            x_slice = no_slicing
        if cropping:
            y_top = y_center + height//2
            y_bottom = y_center - height//2
            y_slice = slice(y_bottom, y_top)
        else:
            y_slice = no_slicing

        data = data_crop[y_slice, x_slice]
    else:
        data = data_crop
        print('No cropping performed')
    return data


def image_plot(data, color='gray', title='', fig_height=6,
               fig_path='', fig_name='test_fig'):
    ''' Plot image


    Parameters
    ----------
    data : image data \n
    color : colormap \n
    title : plot title \n
    fig_height : figure height \n
    fig_path : figure directoty \n
    fig_name : figure name

    Returns
    -------
    None.

    '''
    ds = data.shape
    fig, ax1 = plt.subplots(1, 1, figsize=(fig_height*ds[1]/ds[0], fig_height))
    ax1.imshow(data.squeeze(), cmap=color)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set(xlabel='x-axis  [Pixel index]',
            ylabel='y-axis  [Pixel index]',
            title=title)
    plt.tight_layout()
    plt.savefig(fig_path + fig_name + '.png', dpi='figure')
