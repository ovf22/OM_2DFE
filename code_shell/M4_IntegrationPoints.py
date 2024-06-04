import matplotlib.pyplot as plt
import numpy as np


def load_AbqData(file_name, start_line, end_line, start_offset, end_offset):
    """ Load abaqus data from file.

    Parameters
    ----------
    file_name: Abaqus output file name [str] \n
    start_line: Line in file containing string just before
    the data starts [str] \n
    end_line: Line in file containing string just after the data ends [str] \n
    start_offset: Number of lines after the start_line to the data [int] \n
    end_offset: Number of lines after the data to end_line [int]

    Returns
    -------
    ip_data: Date from integration points [Array] \n
    ip_indices: element and integration point indices [Array]

    """
    with open(file_name, 'r', encoding='cp1257') as f:
        lines = f.readlines()

    # Get start of integration point (IP) list.
    # Get number of first line that contains...
    start_ip = next(i for i, line in enumerate(lines) if start_line in line)
    start_ip += start_offset  # Add offset.
    # Find stop of IPs.
    # Get the first line searching backwards that contains...
    end_ip = len(lines) - next(i for i, line in enumerate(lines[::-1])
                               if end_line in line)
    end_ip -= end_offset  # Subtract offset.

    # Get IP lines.
    ip_lines = lines[start_ip:end_ip]
    # Create array with IPs.
    ip_data = np.array([l.split() for l in ip_lines], dtype=float)
    ip_indices = ip_data[:, :2].astype(int)
    ip_data = ip_data[:, 2:]
    return ip_data, ip_indices


def Int_point_plotting2D(data, ip_coords, ip_data_set, Nplots,
                         vmin=[-10], vmax=[10], unit='unit',
                         variable='Var', deci=0, cmap='coolwarm',
                         fig_name='Test', fig_path='', fig_title='',
                         figsize=(9, 6), sp_title=None):
    ''' Plot field variables from integration points on top of image data

    Parameters
    ----------
    data: Image data [Array] \n
    ip_coords: Integration point coordinates scaled to same dimension and
    coordinate syatem as the image data [Array]. \n
    ip_data: Data from integration points used for scatter plotting [Array] \n
    Nplots: Number of subplots [int] \n
    vmin: Minimum limit in colorplot [float] \n
    vmax: Maximum limit in colorplt [float] \n
    variable: Variable of ip_data [str] \n
    unit: Unit for variabel [str] \n
    deci: Number of decimals for colorbar ticks [int] \n
    cmap : colormap for scatterplot [str] \n
    fig_name : figure name [str] \n
    fig_path : figure directory for saving [str] \n
    fig_title : figure name [str] \n
    figsize : figure size (tuple) \n
    sp_title : subplots title [list]

    Returns
    -------
    None.

    '''
    x_min, x_max = np.min(ip_coords[:, 0]), np.max(ip_coords[:, 0])
    y_min, y_max = np.min(ip_coords[:, 1]), np.max(ip_coords[:, 1])

    fig = plt.figure(figsize=figsize)

    for i in range(Nplots):
        if ip_data_set.ndim == 1:
            ip_data = ip_data_set
        else:
            ip_data = ip_data_set[:, i]
        ax = fig.add_subplot(1, Nplots, i+1)
        ax.set_box_aspect((y_max-y_min)/(x_max-x_min))

        ax.set(xlabel='x-axis [Pixel index]',
               ylabel='y-axis [Pixel index]')

        ax.imshow(data[:, :], cmap='gray')
        ax.set_ylim(ax.get_ylim()[::-1])
        sc2D = ax.scatter(ip_coords[:, 0], ip_coords[:, 1],
                          c=ip_data, cmap=cmap,
                          vmin=vmin[i], vmax=vmax[i], s=35, alpha=.6)
        clb = fig.colorbar(sc2D, shrink=0.5, pad=0.1,
                           ticks=[vmin[i], vmin[i]/2,
                                  0, vmax[i]/2, vmax[i]])

        clb.ax.set_title(variable + ' [' + unit + ']', pad=15)

        clb.ax.set_yticklabels(['$<$'+str(round(vmin[i], deci)),
                                str(round(vmin[i]/2, deci)), '0',
                                str(round(vmax[i]/2, deci)),
                                '$>$'+str(round(vmax[i], deci))])

        if sp_title is not None:
            ax.title.set_text(sp_title[i])
            ax.title.set_size(16)

    fig.suptitle(fig_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path + fig_name + '.png')
