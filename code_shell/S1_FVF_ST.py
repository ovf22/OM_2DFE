import cv2
import numpy as np
import os
from scipy.stats import norm
from scipy.signal import fftconvolve
from sklearn.mixture import GaussianMixture
from structure_tensor import eig_special_2d, structure_tensor_2d

import M1_ImgHandling as M1IH
import M2_Alignment as M2A


# File names and directory
job_name = 'shell_sample_name'

sample_name = 'CFPP_xy'
file_name = sample_name + '.tif'
sample_npz = sample_name + '.npz'

data_path = '../data/'
result_path = '../results/'+job_name+'_files/figures/'

data_file_path = os.path.join(data_path, file_name)

initial_cropping = True

# Load tiff data.
data_full = cv2.imread(data_file_path, cv2.IMREAD_GRAYSCALE)
data_full_shape = data_full.shape
print('Tiff data shape = ', data_full_shape)

# Plot image
M1IH.image_plot(data_full, title='Full image', fig_height=4,
                fig_name='fig1_'+job_name+'_full_microscopy_image',
                fig_path=result_path)

# Read meta data.
PIXEL_SIZE = 0.58167611940656179
print("Pixel size = ", np.round(PIXEL_SIZE, 2), " micrometers")

# Set known fiber diameter in micro meters.
FIBER_DIAMETER = 7

# %% Crop data
if initial_cropping:
    ROI = shell_crop
    data = M1IH.image_ROI_crop_center(data_full,
                                      x_center=ROI[0], y_center=ROI[1],
                                      width=ROI[2], height=ROI[3],
                                      cropping=True)

    # Plot cropped image
    M1IH.image_plot(data, title='Cropped image', fig_height=4,
                    fig_name='fig2_'+job_name+'_cropped_microscopy_image',
                    fig_path=result_path)
else:
    data = data_full.copy()

# %% Fiber volume fraction distribution
# Determine upper and lower threshold for binarization of image
# Define histogram for dataset
hist, bin_edges = np.histogram(data.ravel(), bins=256, density=True)

# Fit a trimodal distribution to the image data
gmm = GaussianMixture(n_components=3)
gmm.fit(data.reshape(-1, 1))

# Statistical properties of the constituents (matrix, boundaries, fiber)
gmm_means = gmm.means_
gmm_stds = gmm.covariances_**0.5
gmm_weights = gmm.weights_

# Sort properties in ascending order
gmm_sort = np.argsort(gmm_means, axis=0)
gmm_means = gmm_means[gmm_sort]
gmm_stds = gmm_stds[gmm_sort]
gmm_weights = gmm_weights[gmm_sort]

# Prepare trimodal distributions for plotting and determination of lower threshold
x_dist = np.arange(0, 256, 1)

y1 = norm.pdf(x_dist, gmm_means[0, 0, 0], gmm_stds[0, 0, 0]) * gmm_weights[0]
y2 = norm.pdf(x_dist, gmm_means[1, 0, 0], gmm_stds[1, 0, 0]) * gmm_weights[1]
y3 = norm.pdf(x_dist, gmm_means[2, 0, 0], gmm_stds[2, 0, 0]) * gmm_weights[2]

# Define lower threshold - Intersection between normal distributions of matrix and boundaries
thres_lower = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()

# Define measured FVF from a burnoff test
FVF_measured = 0.67

# Initialize threshold values and properties for estimating the upper threshold
Vf = 1.
dVf = 1 - FVF_measured
LT = thres_lower[0]
UT = LT + 50
break_while = False

# Define ramping function
fx_init = np.linspace(data.min(), data.max(), data.max()-data.min()+1)
hist, bin_edges = np.histogram(data.ravel(), bins=fx_init.size, density=True)

# Start while loop for finding the upper threshold which satifies the condition for global
while Vf > FVF_measured - 0.2:
    # scaling functions for matrix, composite and fibers ranging from 0 to 1
    fxm = np.zeros(fx_init[fx_init < LT].size)
    fxc = np.linspace(0, 1, fx_init[(fx_init >= LT) & (fx_init <= UT)].size + 1)
    fxf = np.ones(fx_init[fx_init > UT].size-1)
    fx = np.concatenate((fxm, fxc, fxf))

    # Stop the while loop if the condition is true
    if break_while == True:
        break

    # if the integral of the scaling function, fx, and the histogram is close to the measured FVF,
    # stop the while loop and define the upper threshold
    if abs(np.sum(hist * fx) - FVF_measured) > dVf:
        UT -= 1
        break_while = True
    else:
        dVf = abs(np.sum(hist * fx) - FVF_measured)
        Vf = np.sum(hist * fx)
        UT += 1

# Define segmented image (matrix and fibers) with transition between constituents
data_bin = np.copy(data).astype(float)
data_bin -= LT
data_bin[data_bin <= 0.] = 0.
data_bin[(data_bin > 0.) & (data_bin <= (UT-LT))] /= (UT-LT)
data_bin[data_bin >= (UT-LT)] = 1.

# Define function for estimating the FVF based on a kernel size
def FVF(data, kern_mul, FIBER_DIAMETER, PIXEL_SIZE):
    """ Function used for estimating the local fiber volume fraction based on a binarized image.


    Parameters
    ----------
    data : Binarized dataset [2D array of floats] \n
    kern_mul: Kernel multiplier used for controling the kernel window size [int] \n
    FIBER_DIAMETER: Fiber diameter used for defining the kernel window size [float] \n
    PIXEL_SIZE: Image pixel size used for defining the kernel window size [float] \n

    Returns
    -------
    fvf_data : Fiber volume fraction distribution with same array dimensions as the input data [2D array of floats] \n
    k_dim: kernel window size in pixels

    """
    # k_param: Fiber diameter measured in pixels multiplied by kernel_multiplier
    k_param = int(FIBER_DIAMETER / PIXEL_SIZE) * kern_mul
    # k_dim: 1d kernel size
    k_dim = 2*k_param+1
    # kern: 2d kernel window
    kern=np.ones([k_dim, k_dim]) / (k_dim) ** 2

    pad_dim = k_dim//2
    data_pad = np.pad(data_bin, ((pad_dim, pad_dim), (pad_dim, pad_dim)), 'reflect')
    fvf_data = fftconvolve(data_pad.astype('float64'), kern, mode='same')
    fvf_data = fvf_data[pad_dim:-pad_dim, pad_dim:-pad_dim]
    fvf_data[fvf_data<0] = 0
    fvf_data[fvf_data>1] = 1
    
    return fvf_data, k_dim


# Estimate FVF based on averaging kernel window.
if shell_kernel < 99:
    # If the kernel multiplier is set <99, then a variable FVF is used.
    FVF_img, k_dim = FVF(data_bin, shell_kernel, FIBER_DIAMETER, PIXEL_SIZE)
else:
    # If the kernel multiplier is set <=99, then a constant FVF is assumed.
    FVF_inter, k_dim = FVF(data_bin, 4, FIBER_DIAMETER, PIXEL_SIZE)
    FVF_img = np.ones(data.shape) * FVF_inter.mean()

print('Lower threshold ', str(LT))
print('Upper threshold ', str(UT))
print('Mean FVF ', str(FVF_img.mean()))
print('Std FVF ', str(FVF_img.std()))
print('Min/max FVF ', str(FVF_img.min()), '/', str(FVF_img.max()))

# %% Structure tensor analysis
# Set structure tensor parameters
r = FIBER_DIAMETER / 2 / PIXEL_SIZE
sigma = 0.5
rho = 4 * round(np.sqrt(r**2 / 2), 2)

# Copy volume and cast it to 64-bit floating point.
data_f = data.astype(np.float64)

# Calculate S.
S = structure_tensor_2d(data_f, sigma, rho)

truncate = 4
kernel_radius = int(max(sigma, rho) * truncate + 0.5)

print('kernel_radius:', kernel_radius)

S = S[:, kernel_radius:-kernel_radius,
      kernel_radius:-kernel_radius]
S.shape

data_s = data_f[kernel_radius:-kernel_radius,
                kernel_radius:-kernel_radius]

FVF_s = FVF_img[kernel_radius:-kernel_radius,
                kernel_radius:-kernel_radius]

print('The mean Fiber Volume Fraction is: ' + str(round(np.mean(FVF_s), 3)))

# %% Eigenvaluedecomposition
val, vec = eig_special_2d(S)

# Clear some memory.
del data_f
del S

# Smallest eigenvalue corresponds to fiber direction.
# Eigenvectors are returned as vec=[y,x], this is flipped back to vec=[x,y]
vec = np.flip(vec, axis=[0])

# Align all vectors along the positive x-axis
vec_r = vec * np.sign(vec[0])

# Update eigenvectors based on average orientation
vec_new = M2A.orient_average_2D(vec_r)

# Calculate orientations
phi_new = np.arcsin(vec_new[1]) * 180 / np.pi

# Histogram of orientations
M2A.plot_hist(phi_new, limits=5, bins=360, alpha=0.5,
              fig_name='fig3_'+job_name+'_Misalignment_hist',
              fig_path=result_path)


# %% - Overlay plots
# Fiber misalignments
M2A.fig_with_colorbar(
    data_s[:, :],
    phi_new[:, :],
    'Fiber misalignment relative to global fiber direction',
    cmap='coolwarm',
    alpha=0.5,
    vmin=-5,
    vmax=5,
    fig_name='fig4_'+job_name+'_Fiber_misalignment_overlay',
    fig_path=result_path)

# Fiber volume fractions
M2A.fig_with_colorbar(
    data_s[:, :],
    FVF_s[:, :],
    'Fiber Volume Fraction distribution',
    cmap='PRGn',
    alpha=0.5,
    vmin=0,
    vmax=1,
    variable='FVF [-]',
    fig_name='fig5_'+job_name+'_FVF_overlay',
    fig_path=result_path)

# %% Define ramping function for scaling field variables near boundary conditions.
def scaling_data_fade(data, PADDING, PIXEL_SIZE, FVF_measured=None):
    """ Function used for scaling field variables at the start and end along
        the x-axis. This is used to avoid stress concentrations at boundary
        conditions.


    Parameters
    ----------
    data : Dataset being scaled at start and end of x-axis
    PADDING : Length of padding at start and end [int]
    PIXEL_SIZE : Micrograph pixel size size [float] \n
    FVF_measured : Measured Fiber Volume Fraction [float] \n
        Default value is None. This value is only used if the field variable
        being scaled is the FVF. \n

    Returns
    -------
    data_new : Updated dataset with with scaling along the
        start and end of the x-axis

    """
    data_copy = data.copy()
    PADDING_idx = int(PADDING/PIXEL_SIZE)

    scale1d = np.linspace(1, 0, PADDING_idx)
    beta = 2
    scale1dS = 1 / (1 + (scale1d[1: -1] / (1 - scale1d[1: -1]))**(-beta))
    scale1dS = np.insert(scale1dS, [0, scale1dS.size], [1, 0])

    scale2d = np.repeat(scale1dS[:, np.newaxis], data_copy.shape[0], axis=1)

    if FVF_measured is not None:
        scale1dSrev = np.flip(scale1dS)

        scale2drev = np.repeat(scale1dSrev[:, np.newaxis],
                               data_copy.shape[0], axis=1)

        print(scale2drev.shape)
        print(data_copy[:, -PADDING_idx:].shape)

        data_right = data_copy[:, -PADDING_idx:] * scale2d.T + \
            scale2drev.T * FVF_measured
        data_left = data_copy[:, :PADDING_idx] * np.flip(scale2d.T, 1) + \
            np.flip(scale2drev.T, 1) * FVF_measured

    else:
        data_right = data_copy[:, -PADDING_idx:] * scale2d.T
        data_left = data_copy[:, :PADDING_idx] * np.flip(scale2d.T, 1)

    data_new = np.concatenate((data_left,
                               data[:, PADDING_idx:-PADDING_idx],
                               data_right), axis=1)
    return data_new

# Define the length of the padding region where field variables are scaled to nominal values
PADDING = 2*70
# Change field variable data near the boundary using the scaling function
phi_padded = scaling_data_fade(phi_new, PADDING, PIXEL_SIZE)
FVF_padded = scaling_data_fade(FVF_s, PADDING, PIXEL_SIZE, FVF_measured=FVF_s.mean())

# Define model dimensions used in Abaqus
L_MODEL = len(data_s[0, :])*PIXEL_SIZE
T_MODEL = len(data_s[:, 0])*PIXEL_SIZE

MODEL_DIM = [L_MODEL, T_MODEL]
MODEL_DIM_FILE = job_name+'_ImgDim.txt'


# Save model dimensions to file. The file is used in S2_Box.py
np.savetxt(MODEL_DIM_FILE, MODEL_DIM, delimiter=';')

# %% Save variables and constants for mapping orientations to integration
# Variables for S3_mapping.py
np.savez(job_name+'_MAP_VAR.npz', data=data_s, PIXEL_SIZE=PIXEL_SIZE,
         MODEL_DIM=MODEL_DIM, phi=np.transpose(phi_padded, (1, 0)),
         FVF=np.transpose(FVF_padded, (1, 0)))
