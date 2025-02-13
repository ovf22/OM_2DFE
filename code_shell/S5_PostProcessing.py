import matplotlib.pyplot as plt
import numpy as np

import M4_IntegrationPoints as M4IP

job_name = 'shell_sample_name'
result_path = '../results/'+job_name+'_files/figures/'

MAP_VAR = np.load(job_name+'_MAP_VAR.npz')
PIXEL_SIZE = data = MAP_VAR['PIXEL_SIZE']
data = MAP_VAR['data']

# %% Load integration point coordinates
ip_coords, _ = M4IP.load_AbqData(file_name='Out-'+job_name+'_INTCOOR.dat',
                                 start_line='THE FOLLOWING TABLE IS '
                                 'PRINTED AT THE INTEGRATION POINTS '
                                 'FOR ELEMENT TYPE',
                                 end_line='THE ANALYSIS HAS BEEN COMPLETED',
                                 start_offset=6,
                                 end_offset=3)

pixel_size_mm = PIXEL_SIZE / 1000
ip_data_coords = ip_coords / pixel_size_mm

# %% Load stress components at integration points
S_local, _ = M4IP.load_AbqData(file_name='Out-'+job_name+'_S_local.out',
                               start_line='Field Output reported at '
                               'integration points for part',
                               end_line='Minimum',
                               start_offset=5,
                               end_offset=3)

E_local, _ = M4IP.load_AbqData(file_name='Out-'+job_name+'_E_local.out',
                               start_line='Field Output reported at '
                               'integration points for part',
                               end_line='Minimum',
                               start_offset=5,
                               end_offset=3)

# %% Plot stresses in local and global coordinate systems
stresses = ['S11', 'S22', 'S12']
SLmax = [max(idx) for idx in zip(*S_local)]
SLmin = [min(idx) for idx in zip(*S_local)]

M4IP.Int_point_plotting2D(data, ip_data_coords, S_local,
                          S_local.shape[1],
                          vmin=SLmin, vmax=SLmax,
                          unit='MPa', variable='$\\sigma_{ij}$', deci=0,
                          fig_name='fig8_'+job_name+'_local_S',
                          fig_path=result_path,
                          fig_title='Stress in local coordinate system',
                          figsize=(25, 6),
                          sp_title=stresses)

E_local *= 100
strains = ['LE11', 'LE22', 'LE12', 'LE_max_principal']
LEmax = [max(idx) for idx in zip(*E_local)]
LEmin = [min(idx) for idx in zip(*E_local)]

M4IP.Int_point_plotting2D(data, ip_data_coords, E_local,
                          E_local.shape[1],
                          vmin=LEmin, vmax=LEmax,
                          unit='%', variable='$\\epsilon_{ij}$', deci=2,
                          fig_name='fig9_'+job_name+'_local_E',
                          fig_path=result_path,
                          fig_title='Strain in local coordinate system',
                          figsize=(25, 6),
                          sp_title=strains)

# %% Plot stress-strain
def FuncRange(x, xRange):
    """ Save data from list within a defined range.

    Parameters
    ----------
    x: Data for extracting range of data [Array] \n
    xRange: Defined range to extract data [list] \n

    Returns
    -------
    ia: Data point from start of range [float] \n
    ib: Data point from end of range [float]

    """
    ia = 0
    ib = len(x)
    for i in range(len(x)):
        if x[i] > xRange[1]:
            ib = i - 1
            break
    for i in range(ib):
        if x[i] < xRange[0]:
            ia = i
    ib = ib + 1  # in python the range [ia:ib] will not include ib
    return ia, ib


#  Import load-displacement data
LD_file = 'Out-'+job_name+'_load-disp.out'

disp, load = np.loadtxt(open(LD_file, 'r'), delimiter=' ', usecols=(0, 1),
                        dtype=float, unpack=True)

# Import dimensions
length, thickness = np.loadtxt(job_name + '_ImgDim.txt') * 1e-3
A = thickness * 1.0

# Calculate stress-strain
strain = - disp / length * 100
stress = - load / A

stress_max = - np.min(load) / A
strain_max = strain[np.argmin(load)]

# Estimate elastic modulus
StrainRange = [0.05, 0.25]
(ia, ib) = FuncRange(strain, StrainRange)
pEmod = np.polyfit(strain[ia:ib], stress[ia:ib], 1)
Emod = pEmod[0] / 10   # Emod in GPa
Emod_str = r'E-modulus =' + str(format(Emod, '1.2f')) + '$GPa$'

# Plot stress-strain curve
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = plt.plot(strain, stress, 'r--', linewidth=2, label=Emod_str)
ax = plt.plot(strain_max, stress_max, 'rx', markersize=12)

plt.xlim([0, np.max(strain)*1.05])
plt.ylim([0, np.max(stress)*1.05])
plt.grid()
plt.legend()
plt.xlabel('-Strain [$\\%$]')
plt.ylabel('-Stress [MPa]')

plt.savefig(result_path + 'fig10_' + job_name + '_Stress_strain.png')
