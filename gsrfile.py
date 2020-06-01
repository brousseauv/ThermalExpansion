from __future__ import print_function

__author__ = "Veronique Brousseau-Couture"

import os
import warnings
import netCDF4 as nc
import numpy as np

from ElectronPhononCoupling import EpcFile

class GsrFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the OUT.nc file and read it
        fname = fname if fname else self.fname

        super(GsrFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.etotal = root.variables['etotal'][:]
            self.natom = len(root.dimensions['number_of_atoms'])
    
            self.rprim = root.variables['primitive_vectors'][:]
            self.rprim = np.reshape(self.rprim, (3,3))
    
            self.volume = self.cell_volume()

    def cell_volume(self):

        v = np.linalg.det(self.rprim)

        return abs(v)        
    
