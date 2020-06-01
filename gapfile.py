from __future__ import print_function

__author__ = "Veronique Brousseau-Couture"

import os
import warnings
import netCDF4 as nc
import numpy as np

from ElectronPhononCoupling import EpcFile

class GapFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the OUT.nc file and read it
        fname = fname if fname else self.fname

        super(GapFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.gap_energy = root.variables['gap_energy'][:]
            self.gap_energy_units = root.variables['gap_energy'].getncattr('units')

    
