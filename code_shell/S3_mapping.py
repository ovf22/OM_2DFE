import numpy as np
import pandas as pd
from scipy import ndimage

import M4_IntegrationPoints as M4IP

job_name = 'shell_sample_name'
result_path = '../results/'+job_name+'_files/figures/'

MAP_VAR = np.load(job_name+'_MAP_VAR.npz')
PIXEL_SIZE = MAP_VAR['PIXEL_SIZE']
MODEL_DIM = MAP_VAR['MODEL_DIM']  # Length, Thickness
phi = MAP_VAR['phi']
fvf = MAP_VAR['FVF']
data = MAP_VAR['data']

# %% Import integration points
ip_coords, ip_indices = M4IP.load_AbqData(file_name=job_name+'_IP2.dat',
                                          start_line='THE FOLLOWING TABLE IS '
                                          'PRINTED AT THE INTEGRATION POINTS '
                                          'FOR ELEMENT TYPE',
                                          end_line='THE ANALYSIS HAS '
                                          'BEEN COMPLETED',
                                          start_offset=6,
                                          end_offset=3)

# %% Map orientations onto integrations points
# Rescale to match pixel scale.
pixel_size_mm = PIXEL_SIZE / 1000
ip_data_coords = ip_coords / pixel_size_mm

# Get orientation at the coordinates with nearest neighbor interpolation
# Values outside the data are set to 0.
ip_phi = ndimage.map_coordinates(phi, ip_data_coords.T, order=0, cval=np.nan)
ip_fvf = ndimage.map_coordinates(fvf, ip_data_coords.T, order=0, cval=np.nan)

# %% 2D scatter plot of orientations and FVF distribution
M4IP.Int_point_plotting2D(data, ip_data_coords, ip_phi,
                          Nplots=1, vmin=[-5], vmax=[5],
                          unit='$^\\circ$',
                          variable='$\\phi_{xy}$',
                          fig_name='fig6_'+job_name+'_IP-2D misalignment',
                          fig_path=result_path)

M4IP.Int_point_plotting2D(data, ip_data_coords, ip_fvf,
                          Nplots=1, vmin=[0.], vmax=[1.],
                          unit='-',
                          variable='$FVF$',
                          cmap='PRGn',
                          fig_name='fig7_'+job_name+'_IP-2D FVF',
                          fig_path=result_path)


# %% Create dataframe for easy group iteration.
df = pd.DataFrame(ip_indices, columns=['ele_id', 'ip_id'])

# Get max ids.
element_max_index = df['ele_id'].max()
ip_max_index = df['ip_id'].max()

# Set phi values.
df['PHI'] = np.radians(ip_phi)
df['FVF'] = ip_fvf

for column in df.columns[2:]:
    # Open writable file.
    with open(f'{job_name}_{column}.f', 'w') as output:
        # Write header.
        output.write(f'      real*8 {column}0({ip_max_index},{element_max_index})\n')
        # For each element...
        for ele_id, group in df.groupby('ele_id'):
            ip_ids = group['ip_id']
            # Create element header.
            line = f'      DATA ({column}0(I,{ele_id}), I={ip_ids.min()},{ip_ids.max()})/ '
            # Get phis for the element.
            angles = [f'{angle:.6f}' for angle in group[column]]
            for i in range(3, len(angles), 3):
                # Only have three phis per line.
                angles[i] = '\n     &  ' + angles[i]
            # Add commas.
            line += ', '.join(angles)

            output.write(line + '/\n')
