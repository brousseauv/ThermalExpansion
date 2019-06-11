#! \usr/bin/env python

from __future__ import print_function

__author__ = "Veronique Brousseau-Couture"

import os
import warnings
import netCDF4 as nc
import numpy as np

from constants import gpa_to_habo3
from ElectronPhononCoupling import EpcFile

class ElasticFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the OUT.nc file and read it
        fname = fname if fname else self.fname

        super(ElasticFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.stiffness_relaxed = root.variables['elastic_constants_relaxed_ion'][:,:]
            self.stiffness_clamped = root.variables['elastic_constants_clamped_ion'][:,:]
            self.compliance_relaxed = root.variables['compliance_constants_relaxed_ion'][:,:]/gpa_to_habo3
            self.compliance_clamped = root.variables['compliance_constants_clamped_ion'][:,:]/gpa_to_habo3

        
