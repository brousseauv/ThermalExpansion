#! usr/bin/env python
# -*- coding: utf-8 -*-

# Author:  Veronique Brousseau-Couture <veronique.brousseauc@gmail.com>

import numpy as np
import os
import sys
import netCDF4 as nc
import warnings
import itertools as itt

from ElectronPhononCoupling import DdbFile

###################################

class ThermalExpansion(object):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    units = 'eV'
    temp = None
    ntemp = None

    def __init__(self,

        ddb_flists = None,
        out_flists = None,

        units = 'eV',
        temp = np.arange(0,300,50),
        ntemp = None,
        rootname = 'te.out'

        **kwargs):


        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != len(ddb_flists):
            raise Exception('ddb_flists and out_flists must be the same length!')

        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists

        self.units = units
        self.temp = temp
        self.ntemp = len(self.temp) 

# Define the Gibbs free energy (same as Helmholtz but add the PV term
# Do I treat both as the same case, or add a flag "pressure" True/False?



# Also, add a check with the vibrational free energy formula from the book, and check that it is the same as abinit's output...
# get the DDB from text? how does it come out from anaddb ? I would prefer to work with netcdf files...


# get total E, total vibrational energy for each P,T,V(a,b,c)... write the code in a general way, so it can work with any symmetry
# minimize G as a function of a/b/c to get the equilibrium lattice parameters

# add a function to get the grüneisen mode parameters


# I could also call functions from EPC code... it owuld work well! And allow me to 


############################################################
# Also, split this into different files?


# Create a directory if it does not exist
def create_directory(fname):

    dirname = os.path.dirname(fname)
    if not dirname:
        return
    if not os.path.exists(dirname):
        os.system('mkdir -p ' + dirname)

# Main function
def compute(
        #Input files
        ddb_flists = None,
        out_flists = None,
        rootname = 'te.out',

        #Parameters
        units = 'eV',
        temp = None,
        ntemp = None,

        #Options

        **kwargs):

    # Compute corrected bandstructures for ZPM and/or temperature-dependent corrections
    calc = ThermalExpansion(#passed arguments)
            out_flists = out_flists, 
            ddb_flists = ddb_flists,

            rootname = rootname,
            senergy = senergy,

            temp = temperature,
            units = units,

            **kwargs)
    
    # Write output file
    calc.write_freeenergy()
    calc.write_acell()

    return calc


###########################


