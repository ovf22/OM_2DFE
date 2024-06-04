import cv2
import numpy as np
import os

from structure_tensor import eig_special_2d, structure_tensor_2d

import M1_ImgHandling as M1TH
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
M1TH.image_plot(data_full, title='Full image', fig_height=4,
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
    data = M1TH.image_ROI_crop_center(data_full,
                                      x_center=ROI[0], y_center=ROI[1],
                                      width=ROI[2], height=ROI[3],
                                      cropping=True)

    # Plot cropped image
    M1TH.image_plot(data, title='Cropped image', fig_height=4,
                    fig_name='fig2_'+job_name+'_cropped_microscopy_image',
                    fig_path=result_path)
else:
    data = data_full.copy()

# %% FVF distribution - Otsu's thresholding
ret, thresh = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Control size of kernel window
kernel_multiplier = shell_kernel

k_param = int(FIBER_DIAMETER / PIXEL_SIZE) * kernel_multiplier
k_dim = 2 * k_param + 1  # Kernel size
kern = np.ones([k_dim, k_dim]) / (k_dim) ** 2  # Kernel window

# Estimate FVF based on averaging kernel window.
if kernel_multiplier < 99:
    FVF_img = cv2.filter2D(thresh/255, -1, kern)
else:
    # If the kernel multiplier is set <=99, then a constant FVF is assumed.
    FVF_img = np.ones(data.shape) * 0.685

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

# Clear some memory.
del data_f

# %% Eigenvaluedecomposition
val, vec = eig_special_2d(S)

del S
val = val.astype(np.float32)
vec = vec.astype(np.float32)

# Smallest eigenvalue corresponds to fiber direction.
# Eigenvectors are returned as vec=[y,x], this is flipped back to vec=[x,y]
vec = np.flip(vec, axis=[0])

# Align all vectors along the positive x-axis
vec_r = vec * np.sign(vec[0])

# Update eigenvectors based on average orientation
vec_new = M2A.orient_average_2D(vec_r)

# Calculate orientations
phi_new = np.arcsin(vec_new[1])*180/np.pi

# Histogram of orientations
M2A.plot_hist(phi_new, limits=5, bins=360, alpha=0.5,
              fig_name='fig3_'+job_name+'_Misalignment_hist',
              fig_path=result_path)


# %% - Overlay plot of fiber misalignment
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


L_MODEL = len(data_s[0, :])*PIXEL_SIZE
T_MODEL = len(data_s[:, 0])*PIXEL_SIZE

MODEL_DIM = [L_MODEL, T_MODEL]
MODEL_DIM_FILE = job_name+'_ImgDim.txt'


# Save model dimensions to file. The file is used in S2_Box.py
np.savetxt(MODEL_DIM_FILE, MODEL_DIM, delimiter=';')

# %% Save variables and constants for mapping orientations to integration
# Variables for S3_mapping.py
np.savez(job_name+'_MAP_VAR.npz', data=data_s, PIXEL_SIZE=PIXEL_SIZE,
         MODEL_DIM=MODEL_DIM, phi=np.transpose(phi_new, (1, 0)),
         FVF=np.transpose(FVF_s, (1, 0)))
