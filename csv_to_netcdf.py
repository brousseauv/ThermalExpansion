#! usr/bin/env python

import netCDF4 as nc
import numpy as np

'''
This is a sample file to convert .csv files for experimental data imported with PlotDigitizer,
to the netcdf format used in compare_experimental_data.py

Feel free to modify for other imput formats.

'''

__author__ = "Veronique Brousseau-Couture"

######################################################
# rootname for the .csv / .nc file

root = 'GaAs_alphaT'

# units for x and y axis of experimental data
units = ['K','10^6 K^-1']

######################################################

fname = '{}.csv'.format(root)
out_fname = '{}.nc'.format(root)

with open(fname,'r') as f:

    for i in range(4):
        f.readline()

    ndata = np.int(f.readline())

    line = f.readline().split(',')
    ax1 = line[0]
    ax2 = line[1].split('\n')[0]

    data1 = np.zeros((ndata))
    data2 = np.zeros((ndata))

    for i in range(ndata):
        line = f.readline().split(',')
        data1[i] = np.float(line[0])
        data2[i] = np.float(line[1])

f.close()

with nc.Dataset(out_fname,"w") as dts:

    dts.createDimension('number_of_points',ndata)
    
    data = dts.createVariable('ax1','d', ('number_of_points'))
    data[:] = data1
    data.long_name = ax1
    data.units = units[0]

    data = dts.createVariable('ax2','d', ('number_of_points'))
    data[:] = data2
    data.long_name = ax2
    data.units = units[1]




