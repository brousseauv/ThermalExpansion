from __future__ import print_function

__author__ = "Veronique Brousseau-Couture"

import os
import warnings
import netCDF4 as nc
import numpy as np

from ElectronPhononCoupling import EpcFile

class OutFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the OUT.nc file and read it
        fname = fname if fname else self.fname

        super(OutFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            status = root.get_variables_by_attributes(name='etotal')
            if status == []:
                self.etotal = root.variables['etotal2'][:]
            else:
                self.etotal = root.variables['etotal'][:]
            self.acell = root.variables['acell'][:]
            self.natom = int(root.variables['natom'][:])
    
            self.rprim = root.variables['rprim'][:]
            self.rprim = np.reshape(self.rprim, (3,3))
    
            self.symrel = root.variables['symrel'][:]
            self.nsym = int(root.variables['nsym'][:])
            self.symrel = np.reshape(self.symrel, (self.nsym,3,3)) # TEST THIS!!!!! Compare output results with the code I wrote in zpr_analyser. And maybe use this there.
    
            self.volume = self.cell_volume()
            self.weights = root.variables['wtk'][:]

    def cell_volume(self):

        lattice = np.einsum('i,ij->ij',self.acell,self.rprim)
        v = np.linalg.det(lattice)

        return abs(v)        
    
