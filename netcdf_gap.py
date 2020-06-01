import netCDF4 as nc
import numpy as np
from constants import ha_to_ev

# List of number of dataset containing eigenvalues.
ndtset = [9,8,8]

# Last occupied band 
valence = 14

root = ['/RQexec/broussev/calculs/GaAs/volume_variation/bandstructure/set1/OUT/eigk2_o_DS2',
'/RQexec/broussev/calculs/GaAs/volume_variation/bandstructure/set2/OUT/eigk2_o_DS2',
'/RQexec/broussev/calculs/GaAs/volume_variation/bandstructure/set3/OUT/eigk2_o_DS2']

nlist = len(ndtset)

gap_energy = np.zeros((np.sum(ndtset)))

out_root = 'gap_energy_GaAs'

############################################

index = 0
for j in range(nlist):
    for i in range(ndtset[j]):
    
        fname = '{}{}_EIG.nc'.format(root[j],i+1)
    
        with nc.Dataset(fname,'r') as ncdata:
    
            eigenvalues = ncdata.variables['Eigenvalues'][:,:,:]
            units = ncdata.variables['Eigenvalues'].getncattr('units')
            #print(np.shape(eigenvalues))
    
            cond_arr = eigenvalues[0,:,valence]
            val_arr = eigenvalues[0,:,valence-1]
            gap = min(cond_arr) - max(val_arr)

            gap_energy[index] = gap
            index += 1
    
            print('For dataset {}, gap energy = {} Ha = {} eV'.format(i+1,gap,gap*ha_to_ev))

out_fname = '{}.nc'.format(out_root)

with nc.Dataset(out_fname, 'w') as dts:

    dts.createDimension('number_of_datasets',np.sum(ndtset))
    data = dts.createVariable('gap_energy','d',('number_of_datasets'))
    data[:] = gap_energy
    data.units = units
