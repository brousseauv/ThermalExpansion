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
from outfile import OutFile

###################################

class VariableContainer:pass
    
cst = VariableContainer()
cst.ha_to_ev = np.float(27.211396132)
cst.ev_to_ha = np.float(1./cst.ha_to_ev)
cst.kb_haK = 3.1668154267112283e-06 # Boltzmann constant
cst.kb_evK = cst.kb_haK*cst.ha_to_ev
print(cst.kb_evK)


class HelmholtzFreeEnergy(object):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    units = 'eV'
    temperature = None
    ntemp = None

    def __init__(self,

        ddb_flists = None,
        out_flists = None,

        units = 'eV',
        temperature = np.arange(0,300,50),
        ntemp = None,
        rootname = 'te.out',

        **kwargs):


        print('Computing Helmoltz free energy')
        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != np.shape(ddb_flists)[0]:
            raise Exception('ddb_flists and out_flists must be the same length!')

        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists

        self.units = units
        self.temperature = temperature
        self.ntemp = len(self.temperature) 

        # set parameter space dimensions
        nvol, nqpt = np.shape(self.ddb_flists)
        free_energy = np.zeros((nvol,3,self.ntemp))

        # Loop on all volumes
        for v in range(nvol):

            # Open OUTfile
            self.gs = OutFile(out_flists[v])

            # get E
            self.E = self.gs.etotal[0]
            print(self.E)

            # initialize F_0, F_T 
            F_0 = 0.
            F_T = 0.

            # for each qpt:
            for i in range(nqpt):

                # open the ddb file
                ddb = DdbFile(self.ddb_flists[v][i])
                nmode = 3*ddb.natom

                # Check if qpt is Gamma
                is_gamma = ddb.is_gamma

                # diagonalize the dynamical matrix and get the eigenfrequencies
                if is_gamma:
                    ddb.compute_dynmat(asr=True)
                else:
                    ddb.compute_dynmat()
                print(ddb.omega)
                # get F0 contribution
                # get Ftherm contribution
            
        # Check how many lattice parametersv are inequivalent and reduce matrices 
        # see EPC module, qptanalyser function get_se_indices and reduce_array
        # Minimize F



# Also, add a check with the vibrational free energy formula from the book, and check that it is the same as abinit's output...

# get the DDB from text? how does it come out from anaddb ? I would prefer to work with netcdf files...


# get total E, total vibrational energy for each P,T,V(a,b,c)... write the code in a general way, so it can work with any symmetry
# minimize G as a function of a/b/c to get the equilibrium lattice parameters

# add a function to get the grüneisen mode parameters



class GibbsFreeEnergy(object):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    units = 'eV'
    temperature = None
    ntemp = None

    def __init__(self,

        ddb_flists = None,
        out_flists = None,

        units = 'eV',
        temperature = np.arange(0,300,50),
        ntemp = None,
        rootname = 'te.out',

        **kwargs):


        print('Computing Gibbs free energy')
        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if np.shape(out_flists)[0] != np.shape(ddb_flists)[0]:
            raise Exception('ddb_flists and out_flists must have the same number of pressures!')
        if np.shape(out_flists)[1] != np.shape(ddb_flists)[1]:
            raise Exception('ddb_flists and out_flists must have the same number of volumes!')

        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists

        self.units = units
        self.temperature = temperature
        self.ntemp = len(self.temperature) 

        # Loop on all pressures
            # Loop on all volumes
                # get E
                # get PV
                # for each qpt:
                    # diagonaalize the dynamical matrix and get the eigenfrequencies
                    # get F0 contribution
                    # get Ftherm contribution
                
            # Check how many lattice parametersv are inequivalent and reduce matrices
            # Minimize G


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
        temperature = None,
        ntemp = None,
        gibbs = False, # Default value is Helmoltz free energy, at P=0 (or, at constant P)

        #Options

        **kwargs):

    # Choose appropriate type of free energy 
    if gibbs:
        calc = GibbsFreeEnergy(
                out_flists = out_flists, 
                ddb_flists = ddb_flists,
    
                rootname = rootname,
    
                temperature = temperature,
                units = units,
    
                **kwargs)
 
    else:
        calc = HelmholtzFreeEnergy(
                out_flists = out_flists, 
                ddb_flists = ddb_flists,
    
                rootname = rootname,
    
                temperature = temperature,
                units = units,
    
                **kwargs)
    

    # Write output file
#    calc.write_freeenergy()
        # write gibbs or helmholtz, equilibrium acells (P,T), list of temperatures, pressures, initial volumes
        # in netcdf format, will allow to load the data for plotting
#    calc.write_acell()
        # write equilibrium acells, in ascii file
    return calc


###########################


