from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


def rot_mat_2D(phi):
    """ Calculate rotation matrix for rotation about the y and z axis.

    Parameters
    ----------
    phi : Rotation angle around z-axis [float - radians]\n

    Returns
    -------
    R : Rotation matrix [2x2 Array]

    """
    # cos and sin to phi and theta
    cp, sp = np.cos(phi), np.sin(phi)

    # Rotation matrix around z-axis
    R = np.array([[cp, -sp],
                  [sp, cp]])
    return R


def orient_average_2D(vec):
    """ Calculate the average orientation of the image data set.

    Parameters
    ----------
    vec : Eigenvectors from structure tensor analysis [Array]

    Returns
    -------
    vec_new : Corrected eigenvectors for global misalignment of image.

    """
    # Calculate average vector
    vec_avg = (np.average(vec.reshape(2, -1), axis=1)
               / np.linalg.norm(np.average(vec.reshape(2, -1), axis=1)))

    phi_avg = -np.arctan(vec_avg[1] / vec_avg[0])
    print('The mean orientations (phi_xy) =\
          (%1.2f\u00B0)' % np.degrees(phi_avg))

    # Calculate rotation matrix
    R = rot_mat_2D(phi_avg)

    # Calculate new orientation vectors
    vec_new = (R@vec.reshape(2, -1)).reshape(vec.shape)

    return vec_new


def plot_hist(data, bins=180, limits=90, alpha=0.5, title=None,
              fig_name='Misalignment_hist', fig_path=''):
    """ Plot histogram of orientation data.

    Parameters
    ----------
    data : Orientation data [Array of angles]\n
    bins : Number of bins for histogram. The default is 180 [int]\n
    limits : Histogram range defined by [-limits, limits].
    The default is 90 [float]\n
    title : Plot title [str]\n
    fig_name : Name of saved figure without extension [str].
    Default is Misalignment_hist\n
    fig_path : Directory for saving figure [str].
    Default is working directory

    Returns
    -------
    None.

    """
    data_dev = np.mean(np.abs(data))
    data_mean = np.average(data)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # plot histogram of original vector orientations
    ax.hist(data.ravel(), bins=bins, range=[-limits, limits],
            alpha=alpha, color='b', density=True)

    data_mean_str = (r'Mean $\phi_{xy}$ ='
                     + str(format(np.abs(data_mean), '1.0f'))
                     + '$^\\circ$')
    data_abs_mean_str = (r'Mean $|\phi_{xy}|$ = $\pm$ '
                         + str(format(data_dev, '1.2f'))
                         + '$^\\circ$')

    ax.axvline(data_mean, color='k', linestyle='solid',
               linewidth=1.5, label=data_mean_str)
    ax.axvline(data_dev, color='k', linestyle='dotted',
               linewidth=1.5, label=data_abs_mean_str)
    ax.axvline(-data_dev, color='k', linestyle='dotted', linewidth=1.5)

    # add plot formatting
    ax.set_xlabel('Fiber elevation angle, $\\phi_{xy}$ [$^\\circ$]')
    ax.set_ylabel('Fraction [-]')
    ax.set_xlim([-limits, limits])
    plt.legend(loc=2)
    if title:
        ax.set_title(title)
    plt.savefig(fig_path + fig_name + '.png')
    plt.show()


def fig_with_colorbar(data, misalignment, title='', fig_height=6,
                      alpha=0.5, cmap=None, vmin=None, vmax=None,
                      variable='$\\phi_{xy}$ [$^\\circ$]',
                      fig_name='Misalignment_overlay',
                      fig_path=''):
    """ Creates a figure with data, fiber misalignment overlay and color bar.

    Parameters
    ----------
    data : image data [Array]\n
    misalignment : Fiber misalignment 2D overlay [Array]\n
    title : Figure title [str]. Default is empty string\n
    alpha : overlay plot bleeding parameter [float]. The default is 0.5\n
    cmap : Color map for fiber misalignment overlay plot [str]\n
    vmin : Minimum limit for fiber misalignment overlay plot.
    The default is None.\n
    vmax : Minimum limit for fiber misalignment overlay plot.
    The default is None.\n
    fig_name : Name of saved figure without extension [str].
    Default is Tomo_fig\n
    fig_path : Directory for saving figure [str].
    Default is working directory

    Returns
    -------
    None.

    """
    ds = data.shape
    fig, ax = plt.subplots(1, 1, figsize=(fig_height*ds[1]/ds[0], fig_height))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)

    # plot gray scale tomogram image
    ax.imshow(data, cmap='gray')
    ax.set_ylim(ax.get_ylim()[::-1])

    # plot reg overlay of fibermisalignment
    im = ax.imshow(misalignment, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)

    # add colorbar for misalignment range
    clb = fig.colorbar(im, cax=cax, orientation='vertical',
                       ticks=[vmin, vmin/2, 0, vmax/2, vmax])
    clb.ax.set_title(variable)

    clb.ax.set_yticklabels(['$<$'+str(vmin), str(vmin/2), '$0$',
                            str(vmax/2), '$>$'+str(vmax)])
    # add plot formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x-axis [Pixel index]')
    ax.set_ylabel('y-axis [Pixel index]')
    plt.savefig(fig_path + fig_name + '.png')
    plt.show()
