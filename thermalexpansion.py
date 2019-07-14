#! usr/bin/env python
# -*- coding: utf-8 -*-

# Author:  Veronique Brousseau-Couture <veronique.brousseauc@gmail.com>

import numpy as np
from scipy.optimize import curve_fit
import os
import sys
import netCDF4 as nc
import warnings
import itertools as itt

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import constants as cst


from ElectronPhononCoupling import DdbFile
from outfile import OutFile
from gsrfile import GsrFile
from gapfile import GapFile
from elasticfile import ElasticFile
#from zpr_plotter import EXPfile
import eos as eos

from matplotlib import rc
#rc('text', usetex = True)
#rc('font', family = 'serif', weight = 'bold')

###################################

    
tolx = 1E-16
tol12 = 1E-12
tol6 = 1E-6
tol20 = 1E-20

#class EXPfile(CDFfile):
#
#    def __init__(self, *args, **kwargs):
#
#        super(EXPfile, self).__init__(*args,**kwargs)
#
#        self.xaxis = None
#        self.yaxis = None
#        self.ndata = None
#
#    def read_nc(self, fname=None):
#
#        fname = fname if fname else self.fname
#        super(EXPfile, self).read_nc(fname)
#
#        with nc.Dataset(fname,'r') as ncdata:
#
#            self.xaxis = ncdata.variables['ax1'][:]
#            self.yaxis = ncdata.variables['ax2'][:]
#            self.ndata = len(self.xaxis)
#            self.xaxis_units = ncdata.variables['ax1'].getncattr('units')
#            self.yaxis_units = ncdata.variables['ax2'].getncattr('units')

class FreeEnergy(object):

    def __init__(self,

        rootname = 'te.out',
        units = 'eV',
        
        **kwargs):
        
       self.rootname = rootname
       self.units = units

    def get_f0(self,freq):

        # F_0 = Sum (hbar omega)/2
        return np.sum(freq)/2

    def get_fthermal(self,freq,nmode):

        # F_T = kbT Sum ln(1-e^(hbar omega/kbT)), where Zfactor = ln(1-e^(hbar omega/kbT))
        Zfactor = np.zeros((nmode,self.ntemp))

        for imode, omega in enumerate(freq):

            Zfactor[imode,:] = self.get_Zfactor(omega) 

        #Sum on modes
        Zfactor = np.einsum('vt->t',Zfactor)

        return cst.kb_haK*self.temperature*Zfactor

    def get_Zfactor(self, omega):
        
        z = np.zeros(self.ntemp)

        for t, T in enumerate(self.temperature):
            
            # Prevent dividing by 0
            if T<tol20 :
                continue

            x = omega/(cst.kb_haK*T)
            # prevent divergence of the log 
            if x < tolx:
                continue

            z[t] = np.log(1 - np.exp(-x))

        return z    

    def set_weights(self,wtq, normalize=True):
        # Set the qpt weights
        if normalize:
            self.wtq = np.array(wtq)/np.sum(wtq)
        
        else:
            self.wtq = np.array(wtq)

    def ha2molc(self,f0,ft,v):

        x = (f0*np.ones((self.ntemp)) + ft)*cst.ha_to_ev*cst.ev_to_j*cst.avogadro
        for t,T in enumerate(self.temperature):
            #print('T = {:>3d}K : F_0+F_T = {: 13.11e} J/molc'.format(T,x[t]))
            print('T = {:>3d}K : F_0+F_T = {: 13.11e} J/molc = {: 13.11e} Ha'.format(T,x[t],f0+ft[t]))

        fname = '{}_anaddb.dat'.format(self.rootname)

        if v==0:
            action = 'w'
        else:
            action = 'a'

        with open(fname, action) as f:
            f.write('Thermal Free energy, for comparison with anaddb output, for volume index {}\n\n'.format(v+1))
            f.write('{:>8}  {:>15}  {:>15}\n'.format('T (K)','F_0+F_T (Ha)','F_0+F_T (J/molc)'))
            for t, T in enumerate(self.temperature):
                f.write('{:>8d}  {:>14.11e}  {:>14.11e}\n'.format(T,f0+ft[t],x[t]))

            f.write('\n')

        f.close()
            

    def reduce_acell(self,acell):
        # Check which lattice parameter are independent
        x = np.ones((3))*acell[0]
        check = np.isclose(acell,x)

        arr = [0]
        for i in range(2):
            if not check[i+1]:
                arr.append(i+1)

        return arr 

    def get_bose(self,omega,temp):

        # This returns the Bose-Einstein function value for given omega and T, and prevents division by zero
        bose = np.zeros(len(temp))

        for t, T in enumerate(temp):

            if T<tol6:
                continue

            x = omega/(cst.kb_haK*T)

            if x<tolx:
                continue

            bose[t] = 1./(np.exp(x)-1)

        return bose

    def get_specific_heat(self,omega,temp):

        # this returns the phonon specific heat
        cv = np.zeros(len(temp))

        for t,T in enumerate(temp):

            if T<tol6:
                continue

            x = omega/(cst.kb_haK*T)

            if x<tolx:
                continue

            cv[t] = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2

        return cv

    def get_bulkmodulus_from_elastic(self,elast):

        if self.symmetry=='cubic':

            B0 = (elast[0,0]+2*elast[0,1])/3.

        if self.symmetry=='hexagonal':

            c11 = elast[0,0]
            c12 = elast[0,1]
            c13 = elast[0,2]
            c33 = elast[2,2]

            B0 = ( (c11+c12)*c33-2*c13**2)/(c11+c12+2*c33-4*c13)

        return B0
 
class HelmholtzFreeEnergy(FreeEnergy):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    wtq = [1.0]
    temperature = None

    check_anaddb = False

    def __init__(self,

        rootname,
        units,
        symmetry,

        ddb_flists = None,
        out_flists = None,

        wtq = [1.0],
        temperature = np.arange(0,300,50),

        check_anaddb = False,

        bulk_modulus = None,
        bulk_modulus_units = None,

        **kwargs):


        print('Computing Helmoltz free energy')
        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != np.shape(ddb_flists)[0]:
            raise Exception('ddb_flists and out_flists must have the same number of volumes')


        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists
        self.symmetry = symmetry
        if not self.symmetry:
            raise Exception('Symmetry type must be specified')

        super(HelmholtzFreeEnergy,self).__init__(rootname,units)
        self.check_anaddb = check_anaddb

        self.temperature = temperature
        self.ntemp = len(self.temperature) 

        if bulk_modulus_units == 'GPa':
            self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
        elif bulk_modulus_units == 'HaBo3':
            self.bulk_modulus = bulk_modulus
        else:
            raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')

        # set parameter space dimensions
        nvol, nqpt = np.shape(self.ddb_flists)
        self.free_energy = np.zeros((nvol,self.ntemp))
        self.qred = np.zeros((nqpt,3))

        self.volume = np.empty((nvol,4)) # 1st index = data index, 2nd index : total cell volume, (a1,a2,a3)

        # Check that all qpt lists have the same lenght, and that it is equal to the number of wtq
        for v in range(nvol):
            if len(ddb_flists[v][:]) != len(wtq):
                raise Exception('all ddb lists must have the same number of files, and this number should be equal to the number of qpt weights')

        self.set_weights(wtq)

        # Loop on all volumes
        for v in range(nvol):

           # Open OUTfile
            gs = OutFile(out_flists[v])
            self.volume[v,0] = gs.volume
            self.volume[v,1:] = gs.acell

            # Check how many lattice parametersv are inequivalent and reduce matrices 
            # see EPC module, qptanalyser function get_se_indices and reduce_array
            # I do the acell equivalence check only one. There should be no need to check this, as all datasets must describe the same material!

        # what would be the right atol (absolute tolerance) for 2 equivalent lattice parameters? 1E-4, is it too loose?
            if v==0: # REDONDANT SI JE SPECIFIE EXPLICITEMENT LE TYPE DE SYMETRIE... IL VA FALLOIR FAIRE LES 7 GROUPES ?? 
                self.distinct_acell = self.reduce_acell(self.volume[v,1:])
                nmode = 3*gs.natom
                self.natom = gs.natom
                self.omega = np.zeros((nvol,nqpt,nmode))

            # get E
            E = gs.etotal[0]

            # initialize F_0, F_T 
            F_0 = 0.
            F_T = np.zeros((self.ntemp))

            # Add entropy term?? 

            # for each qpt:
            for i in range(nqpt):

                # open the ddb file
                ddb = DdbFile(self.ddb_flists[v][i])
                if  v==0:
                    self.qred[i,:] = ddb.qred
                #nmode = 3*ddb.natom

                # Check if qpt is Gamma
                is_gamma = ddb.is_gamma

                # diagonalize the dynamical matrix and get the eigenfrequencies
                if is_gamma:
                    ddb.compute_dynmat(asr=True)
                else:
                    ddb.compute_dynmat()
                        ##### CHECK WITH GABRIEL IF I SHOULD HAVE LOTO SPLITTING AT GAMMA (WHERE DOES HIS CODE TREAT THE ELECTRIC FIELD PERTURBAITON IN THE DDB AT GAMMA???)

                # Store frequencies for Gruneisen parameters
                self.omega[v,i,:] = ddb.omega
                # get F0 contribution
                F_0 += self.wtq[i]*self.get_f0(ddb.omega) 
                # get Ftherm contribution
                F_T += self.wtq[i]*self.get_fthermal(ddb.omega,nmode)
                
            # Sum free energy = E + F_0 + F_T
            self.free_energy[v,:] = (E+F_0)*np.ones((self.ntemp)) + F_T

        if self.check_anaddb:
            # Convert results in J/mol-cell, to compare with anaddb output
            self.ha2molc(F_0,F_T,v)

        # Minimize F
        #Here, I have F[nvol,T] and also the detailed acell for each volume
        #But I do not have a very detailed free energy surface. I should interpolate it on a finer mesh, give a model function? 
        # For Helmholtz, what is usually done is to fit the discrete F(V,T) = F(a,T), F(b,T), F(c,T)... (each separately) with a parabola, one I have the fitting parameters I can
            # easily get the parabola's minimum.

        # To check the results, add the fitting parameters in the output file. So the fitting can be plotted afterwards.

        # I will have to think about what to do when there is also pressure... do I just use a paraboloid for fitting and minimizing?
        # That would be the main idea. If there is 1 independent acell, it is a parabola (x^2), if there are 2 it is a paraboloid (x^2 + y^2), if there are 3 it would be a paraboloic "volume" (x^2 +
        # y^2 + z^2)
        
        # Minimize F, according to crystal symmetry
        #self.temperature_dependent_acell = self.minimize_free_energy()
#        self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
#        self.acell_via_gruneisen = self.get_acell(nqpt,nmode)
        
# add a function to get the grüneisen mode parameters. This will require to store the frequencies for computation after all volumes have been read and analysed.
# for the Gruneisen, I need the derivative od the frequencies vs volume.

    def minimize_free_energy(self):

        plot = False

        if plot:
            import matplotlib.pyplot as plt
        
        if self.symmetry == 'cubic':
            
            fit = np.zeros((self.ntemp))

            for t, T in enumerate(self.temperature):
                afit = np.polyfit(self.volume[:,1],self.free_energy[:,t],2)
                fit[t] = -afit[1]/(2*afit[0])

                if plot:
                    xfit = np.linspace(9.50,12.0,100)
                    yfit = afit[0]*xfit**2 + afit[1]*xfit + afit[2]
                    plt.plot(self.volume[:,1],self.free_energy[:,t],marker='o')
                    plt.plot(xfit,yfit)

            if plot:
                plt.show()

            fit = np.expand_dims(fit,axis=0)
            return fit

#    def get_gruneisen(self, nqpt, nmode,nvol):
#
#        plot = True 
#
#        if plot :
#            import matplotlib.pyplot as plt
#            fig,arr = plt.subplots(1,2,figsize = (12,6), sharey = False,squeeze=False)
#
#        if self.symmetry == 'cubic':
#            
#            gru = np.zeros((nqpt,nmode))
#
#            for q,v in itt.product(range(nqpt),range(nmode)):
#
#                if q==0 and v<3:
#                # put Gruneisen at zero for acoustic modes at Gamma
#                    gru[q,v] = 0
#                else:
#                    gru[q,v] = -1*np.polyfit(np.log(self.volume[:,1]), np.log(self.omega[:,q,v]),1)[0]
#               
#            # correct divergence at q-->0
#            # this would extrapolate Gruneisen at q=0 from neighboring qpts
#            #x = [np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:])]
#            #for v in range(3):
#            #    y = [gru[1,v],gru[2,v]]
#            #    gru[0,v] = np.polyfit(x,y,1)[1]
#            
#            gru2 = self.gruneisen_from_dynmat(nqpt,nmode,nvol)
#            self.gru2 = gru2
#
#            #print('slope')
#            #print('delta omega {}'.format(self.omega[2,1,:]-self.omega[0,1,:]))
#            #print('omega0 {}'.format(self.omega[1,1,:]))
#            #print('gruneisen {}'.format(gru[1,:]))
# 
#            if plot :
#                #x = np.array([1,2,3])
#                #x = [np.linalg.norm(self.qred[0,:]),np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:]),np.linalg.norm(self.qred[3,:])]
#                #for v in range(nmode):
#                #    plt.plot(x,gru[:4,:],marker='o')
#                
#                # plot mode Gruneisen vs frequency
#                col = ['red','orange','yellow','green','blue','purple']
#                for v in range(nmode):
#                    arr[0][0].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru[:,v],color=col[v],marker = 'o',linestyle='None')
#                    arr[0][0].set_xlabel('Frequency (meV)')
#                    arr[0][0].set_ylabel('Mode Gruneisen')
#
#                    arr[0][1].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru2[:,v],color=col[v],marker = 'o',linestyle='None')
#                    arr[0][1].set_xlabel('Frequency (meV)')
#                    arr[0][1].set_ylabel('Mode Gruneisen')
#                    arr[0][0].set_title(r'Slope ln$\omega$ vs lnV') 
#                    arr[0][1].set_title(r'Dynamical matrix') 
#                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
#                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')
#
#
##            if plot:
##                for c in range(nqpt): 
##                    for i in range(nmode/2):
##                        plt.plot(np.log(self.volume[:,1]),np.log(self.omega[:,c,i]),marker='x')
##                        plt.xlabel('ln V')
##                        plt.ylabel('ln omega')
#
#                plt.savefig('gruneisen_GaAs2.png')
#                plt.show()
#
#            return gru 
            
#    def get_acell(self, nqpt, nmode):
#
#        # Evaluate acell(T) from Gruneisen parameters
#        if self.symmetry == 'cubic':
#            
#            plot = False
#            # First, get alpha(T)
#            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t
#            for t in range(self.ntemp):
#                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
#            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
#            bose = 1./(np.exp(x)-1)
#            bose[0,:3,:] = 0 # Check what Gabriel did)
#            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],bose)
#            # fix this properly later!!! 
#            cv[0,:3,:] = 0
#
#            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen)/(self.volume[1,0]*self.bulk_modulus)
#            alpha2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2)/(self.volume[1,0]*self.bulk_modulus)
#
#            # Then, get a(T)
#            integral = 1./(self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen)
#            a = self.volume[1,1]*(integral + 1)
#
#            if plot:
#                import matplotlib.pyplot as plt
#                fig,arr = plt.subplots(1,2,figsize=(15,5),sharey=False)
#                arr[0].plot(self.temperature,alpha*1E6) 
#                arr[1].plot(self.temperature,alpha2*1E6) 
#                arr[0].set_ylabel(r'$\alpha$ ($10^{-6}$ K$^{-1}$)')
#                arr[0].set_xlabel(r'Temperature (K)')
#                arr[1].set_xlabel(r'Temperature (K)')
#                arr[0].set_title(r'Slope ln$\omega$ vs lnV') 
#                arr[1].set_title(r'Dynamical matrix') 
#
#
#                plt.savefig('alpha_GaAs.png')
#                plt.show() 
#            
#            
#            return a
#    def gruneisen_from_dynmat(self,nqpt,nmode,nvol):
#
#        # This computes the gruneisen parameters from the change in the dynamical matrix
#        # like phonopy and anaddb
#            
#        # for now, I reopen all files, but later on change the loop ordering (anyway, it should not do everything linearly, but rather use functions depending of input parameters
#
#        gru = np.zeros((nqpt,nmode))
#        dplus =  2
#        d0 = 1
#        dminus = 0 # put this in a function
#        dV = self.volume[dplus,1] - self.volume[dminus,1]
#        V0 = self.volume[d0,1]
#
#        for i in range(nqpt):
#            dD = np.zeros((nmode,nmode),dtype=complex)
#            for v in range(nvol):
#
#                # open the ddb file
#                ddb = DdbFile(self.ddb_flists[v][i])
#
#                # Check if qpt is Gamma
#                is_gamma = ddb.is_gamma
#
#                # this is taken from Gabriel's code, directly. for testing purposes.
#                amu = np.zeros(ddb.natom)
#                for ii in np.arange(ddb.natom):
#    
#                    jj = ddb.typat[ii]
#                    amu[ii] = ddb.amu[jj-1]
#
#                dynmat = ddb.get_mass_scaled_dynmat_cart()
#
#                if v==dplus:
#                    dD += dynmat
#                if v==dminus:
#                    dD -= dynmat
#                if v==d0:
#                    eigval, eigvect = np.linalg.eigh(dynmat)
#
#                    for ii in np.arange(ddb.natom):
#                        for dir1 in np.arange(3):
#                            ipert = ii*3 + dir1
#                            eigvect[ipert] = eigvect[ipert]*np.sqrt(cst.me_amu/amu[ii])
#                    
#                    if is_gamma:
#                        eigval[0] = 0.0
#                        eigval[1] = 0.0
#                        eigval[2] = 0.0
#
#                    for ieig,eig in enumerate(eigval):
#                        if eig < 0.0:
#                            warnings.warn('Negative eigenvalue changed to 0')
#                            eigval[ieig] = 0.0
#            #if i==1:
#            #    print(dD)
#            dD_at_q = []
#            
#            for eig in eigvect:
#
#                dD_at_q.append(np.vdot(np.transpose(eig), np.dot(dD,eig)).real)   
##                if i==1:
#                #    print(eig)
#               #     print(np.dot(dD,eig))
#                    #print(dD_at_q)
#                    
#
#            dD_at_q = np.array(dD_at_q)
#
#            for v in range(nmode):
#                if i==0 and v<3:
#                    gru[i,v] = 0
#                else:
#                    gru[i,v] = -V0*dD_at_q[v]/(2*eigval[v].real*dV)
#            if i==1:
#
#                print('Dynamical matrix')
#                print('delta omega^2 {}'.format(dD_at_q))
#                print('omega0 {}'.format(eigval[:]))
#                print('gruneisen {}'.format(gru[i,:]))
#
#
#        return gru


class Gruneisen(FreeEnergy):

    #Input files
    ddb_flists = None
    out_flists = None
    elastic_fname = None

    #Parameters
    wtq = [1.0]
    temperature = None
    pressure = 0.0

    check_anaddb = False

    def __init__(self,

        rootname,
        units,
        symmetry,

        ddb_flists = None,
        out_flists = None,
        elastic_fname = None,

        wtq = [1.0],
        temperature = np.arange(0,300,50),

        check_anaddb = False,

        bulk_modulus = None,
        pressure = 0.0,
        pressure_units = None,
        bulk_modulus_units = None,

        **kwargs):


        print('Computing Gruneisen parameters')
        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != np.shape(ddb_flists)[0]:
            raise Exception('ddb_flists and out_flists must have the same number of volumes')


        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists
        self.elastic_fname = elastic_fname
        self.symmetry = symmetry
        if not self.symmetry:
            raise Exception('Symmetry type must be specified')

        if self.symmetry=='hexagonal':
            if not self.elastic_fname:
                raise Exception('For hexagonal system, elastic compliance tensor (computed with anaddb) must be provided as a .nc file.')

        super(Gruneisen,self).__init__(rootname,units)
        self.check_anaddb = check_anaddb

        self.temperature = temperature
        self.ntemp = len(self.temperature) 
        
        self.pressure_units = pressure_units
        self.pressure = pressure

        if self.pressure_units == 'GPa':
            self.pressure_gpa = self.pressure
            self.pressure = self.pressure*cst.gpa_to_habo3


        if bulk_modulus:
            if bulk_modulus_units == 'GPa':
                self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
            elif bulk_modulus_units == 'HaBo3':
                self.bulk_modulus = bulk_modulus
            else:
                raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')

        # set parameter space dimensions
        '''why the hell does the shape sometimes work and sometimes not???'''
#        nvol, nqpt = np.shape(self.ddb_flists)
        nvol, nqpt = len(self.ddb_flists), len(self.ddb_flists[0])
        self.free_energy = np.zeros((nvol,self.ntemp))
        self.gibbs_free_energy = np.zeros((nvol,self.ntemp))

        self.qred = np.zeros((nqpt,3))

        self.volume = np.empty((nvol,4)) # 1st index = data index, 2nd index : total cell volume, (a1,a2,a3)

        # Check that all qpt lists have the same lenght, and that it is equal to the number of wtq
        for v in range(nvol):
            if len(ddb_flists[v][:]) != len(wtq):
                raise Exception('all ddb lists must have the same number of files, and this number should be equal to the number of qpt weights.\n List index {} has {} entries while there are {} qpt weights.'.format(v,len(ddb_flists[v][:]),len(wtq)))

        self.set_weights(wtq)

        # Loop on all volumes
        for v in range(nvol):

           # Open OUTfile
            gs = OutFile(out_flists[v])
            self.volume[v,0] = gs.volume
            self.volume[v,1:] = gs.acell

            # Check how many lattice parametersv are inequivalent and reduce matrices 
            # see EPC module, qptanalyser function get_se_indices and reduce_array
            # I do the acell equivalence check only one. There should be no need to check this, as all datasets must describe the same material!

        # what would be the right atol (absolute tolerance) for 2 equivalent lattice parameters? 1E-4, is it too loose?
            if v==0: # REDONDANT SI JE SPECIFIE EXPLICITEMENT LE TYPE DE SYMETRIE... IL VA FALLOIR FAIRE LES 7 GROUPES ?? 
                self.distinct_acell = self.reduce_acell(self.volume[v,1:])
                nmode = 3*gs.natom
                self.natom = gs.natom
                self.omega = np.zeros((nvol,nqpt,nmode))

            # get E
            E = gs.etotal[0]

            # initialize F_0, F_T 
            F_0 = 0.
            F_T = np.zeros((self.ntemp))

            # Add entropy term?? 

            # for each qpt:
            for i in range(nqpt):

                # open the ddb file
                ddb = DdbFile(self.ddb_flists[v][i])
                if  v==0:
                    self.qred[i,:] = ddb.qred
                #nmode = 3*ddb.natom

                # Check if qpt is Gamma
                is_gamma = ddb.is_gamma

                # diagonalize the dynamical matrix and get the eigenfrequencies
                if is_gamma:
                    ddb.compute_dynmat(asr=True)
                else:
                    ddb.compute_dynmat()
                        ##### CHECK WITH GABRIEL IF I SHOULD HAVE LOTO SPLITTING AT GAMMA (WHERE DOES HIS CODE TREAT THE ELECTRIC FIELD PERTURBAITON IN THE DDB AT GAMMA???)

                # Store frequencies for Gruneisen parameters
                print(v+1,i+1,ddb.omega)
                self.omega[v,i,:] = ddb.omega
<<<<<<< HEAD
#                if v==1:
#                    print(i+1,ddb.omega)
#
#                    # Manual correction for 0.0gpa
#                    if i+1==26:
#                        self.omega[v,i,0] = 0.2854956226E-04
#                        self.omega[v,i,1] = 0.2854956226E-04
#                    if i+1==56:
#                        self.omega[v,i,0] = 0.3932015304E-04
#                        self.omega[v,i,1] = 0.3932015304E-04


#                    # Manual correction for 0.5gpa
#                    if i+1==26:
#                        self.omega[v,i,0] = 0.4283688046E-04
#                        self.omega[v,i,1] = 0.4283688046E-04
#                    if i+1==56:
#                        self.omega[v,i,0] = 0.5818795222E-04
#                        self.omega[v,i,1] = 0.5818795222E-04

#                    # Manual correction for 1gpa
#                    if i+1==26:
#                        self.omega[v,i,0] = 0.5350676926E-04
#                        self.omega[v,i,1] = 0.5350676926E-04
=======

                '''Manual corrections for ""negative"" frequencies in the DDB along Gamma-A, NOT in phonon dispersion...'''

                if self.pressure_gpa==0.0:

                    if v==0:
                        if i+1==11:
                            self.omega[q,v,0] = 0.1701677510E-04
                            self.omega[q,v,1] = 0.1701677510E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.3081052709E-04
                            self.omega[q,v,1] = 0.3081052709E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.3951433427E-04
                            self.omega[q,v,1] = 0.3951433427E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.4246114942E-04
                            self.omega[q,v,1] = 0.4246114942E-04

                    if v==1:
                        if i+1==26:
                            self.omega[v,i,0] = 0.2854956226E-04
                            self.omega[v,i,1] = 0.2854956226E-04
                        if i+1==56:
                            self.omega[v,i,0] = 0.3932015304E-04
                            self.omega[v,i,1] = 0.3932015304E-04

                    if v==2:
                        if i+1==11:
                            self.omega[q,v,0] = 0.1445752629E-04
                            self.omega[q,v,1] = 0.1445752629E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.2615837711E-04
                            self.omega[q,v,1] = 0.2615837711E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.3352045144E-04
                            self.omega[q,v,1] = 0.3352045144E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.3600703695E-04
                            self.omega[q,v,1] = 0.3600703695E-04

                    if v==3:
                        if i+1==11:
                            self.omega[q,v,0] = 0.1844002580E-04
                            self.omega[q,v,1] = 0.1844002580E-04
                        if i+1==26:
                            self.omega[q,v,0] =0.3321005229E-04 
                            self.omega[q,v,1] = 0.3321005229E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.4239809330E-04
                            self.omega[q,v,1] = 0.4239809330E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.4548321391E-04
                            self.omega[q,v,1] = 0.4548321391E-04

                    if v==5:
                        if i+1==11:
                            self.omega[q,v,0] = 0.1329674604E-04
                            self.omega[q,v,1] = 0.1329674604E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.2417267515E-04
                            self.omega[q,v,1] = 0.2417267515E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.3110661810E-04
                            self.omega[q,v,1] = 0.3110661810E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.3346756983E-04
                            self.omega[q,v,1] = 0.3346756983E-04

                if self.pressure_gpa==0.5:

                    if v==1:
                       if i+1==26:
                           self.omega[v,i,0] = 0.4283688046E-04
                           self.omega[v,i,1] = 0.4283688046E-04
                       if i+1==56:
                           self.omega[v,i,0] = 0.5818795222E-04
                           self.omega[v,i,1] = 0.5818795222E-04

                    if v==2:
                        if i+1==11:
                            self.omega[q,v,0] = 0.2238287077E-04
                            self.omega[q,v,1] = 0.2238287077E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.3991101016E-04
                            self.omega[q,v,1] = 0.3991101016E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.5055067226E-04
                            self.omega[q,v,1] = 0.5055067226E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.5407840848E-04
                            self.omega[q,v,1] = 0.5407840848E-04

                    if v==3:
                        if i+1==11:
                            self.omega[q,v,0] = 0.2778743747E-04
                            self.omega[q,v,1] = 0.2778743747E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.4923183936E-04
                            self.omega[q,v,1] = 0.4923183936E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.6209346782E-04
                            self.omega[q,v,1] = 0.6209346782E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.6634359246E-04
                            self.omega[q,v,1] = 0.6634359246E-04

                    if v==5:
                        if i+1==11:
                            self.omega[q,v,0] = 0.2088010115E-04
                            self.omega[q,v,1] = 0.2088010115E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.3749295751E-04
                            self.omega[q,v,1] = 0.3749295751E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.4775718968E-04
                            self.omega[q,v,1] = 0.4775718968E-04
                        if i+1==56:
                            self.omega[q,v,0] = 0.5119297864E-04
                            self.omega[q,v,1] = 0.5119297864E-04

                if self.pressure_gpa==1.0:

                    if v==1:
                        if i+1==26:
                            self.omega[v,i,0] = 0.5350676926E-04
                            self.omega[v,i,1] = 0.5350676926E-04

                    if v==2:
                        if i+1==11:
                            self.omega[q,v,0] = 0.2815987569E-04
                            self.omega[q,v,1] = 0.2815987569E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.4973369845E-04
                            self.omega[q,v,1] = 0.4973369845E-04

                    if v==3:
                        if i+1==11:
                            self.omega[q,v,0] = 0.3428041003E-04
                            self.omega[q,v,1] = 0.3428041003E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.6015751069E-04
                            self.omega[q,v,1] = 0.6015751069E-04

                    if v==5:
                        if i+1==11:
                            self.omega[q,v,0] = 0.2648013493E-04
                            self.omega[q,v,1] = 0.2648013493E-04
                        if i+1==26:
                            self.omega[q,v,0] = 0.4713674353E-04
                            self.omega[q,v,1] = 0.4713674353E-04
                        if i+1==41:
                            self.omega[q,v,0] = 0.5965927239E-04
                            self.omega[q,v,1] = 0.5965927239E-04

                if self.pressure_gpa==1.5:

                if self.pressure_gpa==3.5:

                if self.pressure_gpa==5.0:

>>>>>>> 4df947079054e40ba60338a58bffdca23e28662e
                # get F0 contribution
                F_0 += self.wtq[i]*self.get_f0(ddb.omega) 
                # get Ftherm contribution
                F_T += self.wtq[i]*self.get_fthermal(ddb.omega,nmode)

        # Add a check for the right volumes ordering?
        # cubic : aminus equil aplus
        #hexagonal : aminus equil aplus cminus equil cplus
                
            self.free_energy[v,:] = (E+F_0)*np.ones((self.ntemp)) + F_T
            self.gibbs_free_energy[v,:] = (E+F_0+self.pressure*self.volume[v,0])*np.ones((self.ntemp)) + F_T 

            if self.check_anaddb:
                print('Thermal free energy for volume {}'.format(v))
                # Convert results in J/mol-cell, to compare with anaddb output
                self.ha2molc(F_0,F_T,v)

        ### End of loop on volumes, all data has been read ###
        ''' TO DO'''
        ''' First fit of Murnaghan EOS from static calc results, to get parameters??? or use them as input???'''

        ''' Fit Murnaghan EOS for Ftot(V) at each T, to get the 'real' P I must add for the Gibbs free energy'''



        # Minimize F
        #Here, I have F[nvol,T] and also the detailed acell for each volume
        #But I do not have a very detailed free energy surface. I should interpolate it on a finer mesh, give a model function? 
        # For Helmholtz, what is usually done is to fit the discrete F(V,T) = F(a,T), F(b,T), F(c,T)... (each separately) with a parabola, one I have the fitting parameters I can
            # easily get the parabola's minimum.

        # To check the results, add the fitting parameters in the output file. So the fitting can be plotted afterwards.

        # I will have to think about what to do when there is also pressure... do I just use a paraboloid for fitting and minimizing?
        # That would be the main idea. If there is 1 independent acell, it is a parabola (x^2), if there are 2 it is a paraboloid (x^2 + y^2), if there are 3 it would be a paraboloic "volume" (x^2 +
        # y^2 + z^2)
        
        # Minimize F, according to crystal symmetry

        ### Add a check for homogenious acell increase (for central finite difference)
        self.equilibrium_volume = self.volume[1,:]
<<<<<<< HEAD
        self.temperature_dependent_acell = self.minimize_free_energy()
        print(np.shape(self.temperature_dependent_acell))
=======
#        self.minimize_free_energy_from_eos()
>>>>>>> 4df947079054e40ba60338a58bffdca23e28662e

        # Read elastic compliance from file
        if self.elastic_fname:
            elastic = ElasticFile(self.elastic_fname)
            self.compliance = elastic.compliance_relaxed
            self.compliance_rigid = elastic.compliance_clamped

            bmod = self.get_bulkmodulus_from_elastic(elastic.stiffness_relaxed)
            bmod2 = self.get_bulkmodulus_from_elastic(elastic.stiffness_clamped)
            print('Bulk modulus from elastic constants = {:>7.3f} GPa'.format(bmod))
            print('Bulk modulus from elastic constants (clamped) = {:>7.3f} GPa'.format(bmod2))

            print('Elastic constants:')
            print('c11 = {}, c33 = {}, c12 = {}, c13 = {} GPa'.format(elastic.stiffness_relaxed[0,0],elastic.stiffness_relaxed[2,2],elastic.stiffness_relaxed[0,1],elastic.stiffness_relaxed[0,2]))
            print('Compliance constants:')
            print('s11 = {}, s33 = {}, s12 = {}, s13 = {} GPa^-1'.format(elastic.compliance_relaxed[0,0],elastic.compliance_relaxed[2,2],elastic.compliance_relaxed[0,1],elastic.compliance_relaxed[0,2]))


        self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
        self.acell_via_gruneisen = self.get_acell(nqpt,nmode)
        self.effective_phonon_pressure = self.get_phonon_effective_pressure(nqpt,nmode)
        
# add a function to get the grüneisen mode parameters. This will require to store the frequencies for computation after all volumes have been read and analysed.
# for the Gruneisen, I need the derivative od the frequencies vs volume.

    def minimize_free_energy_from_eos(self):

        ## Same as minimize_free_energy, but fit a Murnaghan EOS instead of a paraboloid

        if self.symmetry == 'hexagonal':

            # Only independent fit for now...
            fit = np.zeros((2,self.ntemp))

            for t,T in enumerate(self.temperature):

                # First, treat a
                # V0, E0, B0, B0' 
                '''or, B0=9.8, B0'=7.6'''
                p0 = [self.equilibrium_volume[1],self.equilibrium_volume[3], self.gibbs_free_energy[1,t], 8.7*cst.gpa_to_habo3,8.9*cst.gpa_to_habo3]
                popt, pcov= curve_fit(eos.murnaghan_EV_axial, self.volume[:,0], self.gibbs_free_energy[:,t], p0)
#                print('for T={}K, popt= {}'.format(T,popt))
        

    def minimize_free_energy(self):

        plot = False

        if plot:
            import matplotlib.pyplot as plt
        
        if self.symmetry == 'cubic':
            
            fit = np.zeros((self.ntemp))

            for t, T in enumerate(self.temperature):
                afit = np.polyfit(self.volume[:,1],self.free_energy[:,t],2)
                fit[t] = -afit[1]/(2*afit[0])

                if plot:
                    xfit = np.linspace(9.50,12.0,100)
                    yfit = afit[0]*xfit**2 + afit[1]*xfit + afit[2]
                    plt.plot(self.volume[:,1],self.free_energy[:,t],marker='o')
                    plt.plot(xfit,yfit)

            if plot:
                plt.show()

            #### ONLY FOR GaAs !!!!! Rescale expansion to experimental parameter
            fit = fit - np.ones(len(fit))*0.18573269

            fit = np.expand_dims(fit,axis=0)
            return fit

        if self.symmetry == 'hexagonal':
            
            from scipy.optimize import leastsq

            fit = np.zeros((2,self.ntemp))
            fit2d = np.zeros((2,self.ntemp))
            fitg = np.zeros((2,self.ntemp))
            fit2dg = np.zeros((2,self.ntemp))


            if plot:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

            # This delta is the difference between real HGH minimum and PAW minimum, at the current pressure
            #delta =  [0.011000356489930141,0.0844492602578999] #for 0gpa
            #delta = [0.004513105799999195,0.0239153519999995] #for 0.5gpa
            #delta = [0.0,0.0] #for 1gpa
            #delta = [0.0031372478999998066,0.001351464000000746] #for 1.5gpa
            #delta = [0.0,0.0] #for 3gpa
            delta = [0.001538456200000482,-0.0026159529999993936] #for 3.5gpa
            #delta = [0.016926552700000208,-0.0055531139999995816] #for 5gpa
            print('From free energy minimization')
            for t, T in enumerate(self.temperature):
                afit = np.polyfit(self.volume[:3,1],self.free_energy[:3,t],2)
                fit[0,t] = -afit[1]/(2*afit[0])
                cfit = np.polyfit(self.volume[3:,3],self.free_energy[3:,t],2)
                fit[1,t] = -cfit[1]/(2*cfit[0])

                
                fit2, cov2 = leastsq(self.residuals, x0=[afit[0],afit[0],cfit[0],cfit[0],self.free_energy[1,t]], args=(self.volume[:,1],self.volume[:,3],
                    self.free_energy[:,t]),maxfev=4000)
                fit2d[:,t] = fit2[0],fit2[2]
                print('\nT={}'.format(T))
                #print(fit2)
                #print(cov2)
#                print('independent fit')
#                print(fit[:,t])
#                print('2d fit')
#                print(fit2d[:,t])

                # Fit Gibbs free energy
                afitg = np.polyfit(self.volume[:3,1],self.gibbs_free_energy[:3,t],2)
                fitg[0,t] = -afitg[1]/(2*afitg[0])
                cfitg = np.polyfit(self.volume[3:,3],self.gibbs_free_energy[3:,t],2)
                fitg[1,t] = -cfitg[1]/(2*cfitg[0])


                fit2g, cov2g = leastsq(self.residuals, x0=[afitg[0],afitg[0],cfitg[0],cfitg[0],self.gibbs_free_energy[1,t]], args=(self.volume[:,1],self.volume[:,3],
                    self.gibbs_free_energy[:,t]),maxfev=4000)
                fit2dg[:,t] = fit2g[0],fit2g[2]
#                print('Gibbs')
                print(fitg[:,t]-delta)
                print(fit2dg[:,t]-delta)
#                print(self.gibbs_free_energy[:,t])
                



                if plot:
                    fig = plt.figure()
                    arr = fig.add_subplot(111,projection='3d')
                    arr.plot(self.volume[:3,1],self.volume[:3,3],self.gibbs_free_energy[:3,t],marker='o',color='k',linestyle='None') #at T=0
                    arr.plot(self.volume[3:,1],self.volume[3:,3],self.gibbs_free_energy[3:,t],marker='o',color='b',linestyle='None') #at T=0

                    xmesh = np.linspace(0.99*self.volume[0,1],1.01*self.volume[2,1],200)
                    ymesh = np.linspace(0.99*self.volume[3,3],1.01*self.volume[5,3],200)
                    xmesh,ymesh = np.meshgrid(xmesh,ymesh)
                    zmesh = self.paraboloid(xmesh,ymesh,p0=fit2g)
                    zlim = arr.get_zlim3d()
                    arr.plot_wireframe(xmesh,ymesh,zmesh)
                    xx = np.ones((10))
                    arr.plot(fit2dg[0,t]*xx,fit2dg[1,t]*xx,np.linspace(0.99999*zlim[0],1.00001*zlim[1],10),color='magenta',linewidth=2,zorder=3)

                    out='FIG/{}_{}K.png'.format(self.rootname,T)
                    create_directory(out)
                    plt.savefig(out)
                    plt.show()
                    plt.close()

            self.independent_fit = fit
            self.fit2d = fit2d 
            self.fitg = fitg
            self.fit2dg = fit2dg

            return fit

    def paraboloid(self, x,y, p0):

        a0 = p0[0]
        A = p0[1]
        c0 = p0[2]
        C = p0[3]
        B = p0[4]

        z = (x-a0)**2/A**2 + (y-c0)**2/C**2 + B

        return z

    def residuals(self,params,x,y,z):
        # params = [a0,A,c0,C]
        return z - self.paraboloid(x,y,params)
 

    def get_gruneisen(self, nqpt, nmode,nvol):

        plot = False

        if plot :
            import matplotlib.pyplot as plt
            fig,arr = plt.subplots(1,2,figsize = (12,6), sharey = False,squeeze=False)

        if self.symmetry == 'cubic':
            
            gru = np.zeros((nqpt,nmode))
            self.gruvol = np.zeros((nqpt,nmode))

            for q,v in itt.product(range(nqpt),range(nmode)):

                if q==0 and v<3:
                # put Gruneisen at zero for acoustic modes at Gamma
                    gru[q,v] = 0
                else:
#                    gru[q,v] = -1*np.polyfit(np.log(self.volume[:,1]), np.log(self.omega[:,q,v]),1)[0]
                    # This is the LINEAR gruneisen parameters
#                    gru[q,v] = -self.equilibrium_volume[1]/self.omega[1,q,v]*np.polyfit(self.volume[:,1],self.omega[:,q,v],1)[0]
                    gru[q,v] = -self.equilibrium_volume[1]/self.omega[1,q,v]*(self.omega[2,q,v]-self.omega[0,q,v])/(self.volume[2,1]-self.volume[0,1])

                    # This is the VOLUMIC one (that is, gru(linear)/3)
                    self.gruvol[q,v] = -self.equilibrium_volume[0]/self.omega[1,q,v]*np.polyfit(self.volume[:,0],self.omega[:,q,v],1)[0]
              
            # correct divergence at q-->0
            # this would extrapolate Gruneisen at q=0 from neighboring qpts
            #x = [np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:])]
            #for v in range(3):
            #    y = [gru[1,v],gru[2,v]]
            #    gru[0,v] = np.polyfit(x,y,1)[1]
            
            gru2 = self.gruneisen_from_dynmat(nqpt,nmode,nvol)
            self.gru2 = gru2


            # get the effective pressure from phonons, for a given temperature


            #print('slope')
            #print('delta omega {}'.format(self.omega[2,1,:]-self.omega[0,1,:]))
            #print('omega0 {}'.format(self.omega[1,1,:]))
            #print('gruneisen {}'.format(gru[1,:]))
            if plot :
                #x = np.array([1,2,3])
                #x = [np.linalg.norm(self.qred[0,:]),np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:]),np.linalg.norm(self.qred[3,:])]
                #for v in range(nmode):
                #    plt.plot(x,gru[:4,:],marker='o')
                
                # plot mode Gruneisen vs frequency
                col = ['red','orange','yellow','green','blue','purple']
                for v in range(nmode):
                    arr[0][0].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru[:,v],color=col[v],marker = 'o',linestyle='None')
                    arr[0][0].set_xlabel('Frequency (meV)')
                    arr[0][0].set_ylabel('Mode Gruneisen')

                    arr[0][1].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru2[:,v],color=col[v],marker = 'o',linestyle='None')
                    arr[0][1].set_xlabel('Frequency (meV)')
                    arr[0][1].set_ylabel('Mode Gruneisen')
                    arr[0][0].set_title(r'Slope $\omega$ vs V') 
                    arr[0][1].set_title(r'Dynamical matrix') 
                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')
                    arr[0][0].grid(b=True, which='major')



#            if plot:
#                for c in range(nqpt): 
#                    for i in range(nmode/2):
#                        plt.plot(np.log(self.volume[:,1]),np.log(self.omega[:,c,i]),marker='x')
#                        plt.xlabel('ln V')
#                        plt.ylabel('ln omega')

                plt.savefig('gruneisen_GaAs2.png')
                plt.show()

            return gru


        if self.symmetry == 'hexagonal':
            
            gru = np.zeros((2,nqpt,nmode)) # Gru_a, Gru_c
            self.gru_vol = np.zeros((nqpt,nmode))
            gru2 = np.zeros((2,nqpt,nmode)) #withfinite difference, on the frequencies

            nlarge = np.zeros((2)) #remove after testing large Gruneisens
            large_a = []
            large_c = []

            for q,v in itt.product(range(nqpt),range(nmode)):

                if q==0 and v<3:
                # put Gruneisen at zero for acoustic modes at Gamma
                    gru[:,q,v] = 0
                else:
#                    gru[q,v] = -1*np.polyfit(np.log(self.volume[:,1]), np.log(self.omega[:,q,v]),1)[0]
                    # This is the LINEAR gruneisen parameters
                    gru2[0,q,v] = -self.equilibrium_volume[1]/(2*self.omega[1,q,v])*np.polyfit(self.volume[:3,1],self.omega[:3,q,v],1)[0]
                    gru2[1,q,v] = -self.equilibrium_volume[3]/self.omega[1,q,v]*np.polyfit(self.volume[3:,3],self.omega[3:,q,v],1)[0]
                    gru[0,q,v] = -self.equilibrium_volume[1]/(2*self.omega[1,q,v])*(self.omega[2,q,v]-self.omega[0,q,v])/(self.volume[2,1]-self.volume[0,1])
                    gru[1,q,v] = -self.equilibrium_volume[3]/(self.omega[1,q,v])*(self.omega[5,q,v]-self.omega[3,q,v])/(self.volume[5,3]-self.volume[3,3])


                    # This is the VOLUMIC one (that is, gru(linear)/3)
                    self.gru_vol[q,v] = (gru[0,q,v] + gru[1,q,v])/3.

#                    # Test to evaluate the impact of the very large Gruneisens on the final sum
#                    if np.abs(gru[0,q,v])>5:
#                        gru[0,q,v] = 0.
#                        nlarge[0] += 1
#                        large_a.append([q,v]) 
#                    if np.abs(gru[1,q,v])>5:
#                        gru[1,q,v] = 0.
#                        nlarge[1] += 1
#                        large_c.append([q,v])
#
#
#            print('Number of large Gruneisen parameters put to 0 : {} for gamma_a, {} for gamma_c'.format(nlarge[0],nlarge[1]))
#            print('They are for (q,v):')
#            print('Gamma_a')
#            if len(large_a) != 0:
#                for p in large_a:
#                    print(p)
#            print('Gamma_c')
#            if len(large_c) != 0:
#                for p in large_c:
#                    print(p)



            self.gru2 = gru2
#            print(gru_tot[0,3:])


            if plot :
                #x = np.array([1,2,3])
                #x = [np.linalg.norm(self.qred[0,:]),np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:]),np.linalg.norm(self.qred[3,:])]
                #for v in range(nmode):
                #    plt.plot(x,gru[:4,:],marker='o')
                
                # plot mode Gruneisen vs frequency
                col = ['red','orange','yellow','green','blue','purple','pink','gray','black','orange','cyan','magenta']
                for v in range(nmode):
                    arr[0][0].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru[0,:,v],color=col[v],marker = 'o',linestyle='None')
                    arr[0][0].set_xlabel('Frequency (meV)')
                    arr[0][0].set_ylabel(r'Mode Gruneisen, $\gamma^a$')

                    arr[0][1].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru[1,:,v],color=col[v],marker = 'o',linestyle='None')
                    arr[0][1].set_xlabel('Frequency (meV)')
                    arr[0][1].set_ylabel(r'Mode Gruneisen, $\gamma^c$')
#                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
#                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')
                    arr[0][0].grid(b=True, which='major')
                    arr[0][1].grid(b=True, which='major')

#                    #with linear fit
#                    arr[1][0].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru2[0,:,v],color=col[v],marker = 'o',linestyle='None')
#                    arr[1][0].set_xlabel('Frequency (meV)')
#                    arr[1][0].set_ylabel(r'Mode Gruneisen, $\gamma^a$')
#
#                    arr[1][1].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru2[1,:,v],color=col[v],marker = 'o',linestyle='None')
#                    arr[1][1].set_xlabel('Frequency (meV)')
#                    arr[1][1].set_ylabel('Mode Gruneisen, $\gamma^c$')
##                    arr[0][0].set_title(r'Slope $\omega$ vs V') 
##                    arr[0][1].set_title(r'Dynamical matrix') 
##                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
##                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')
#                    arr[1][0].grid(b=True, which='major')
#                    arr[1][1].grid(b=True, which='major')


#            if plot:
#                for c in range(nqpt): 
#                    for i in range(nmode/2):
#                        plt.plot(np.log(self.volume[:,1]),np.log(self.omega[:,c,i]),marker='x')
#                        plt.xlabel('ln V')
#                        plt.ylabel('ln omega')
                
                plt.suptitle('{}'.format(self.rootname))

                outfile = 'FIG/gruneisen_{}.png'.format(self.rootname)
                create_directory(outfile)

                plt.savefig(outfile)
#                plt.show()

            return gru 

#
            
    def get_acell(self, nqpt, nmode):

        # Evaluate acell(T) from Gruneisen parameters
        if self.symmetry == 'cubic':
            
            plot = True
            # First, get alpha(T)

            # Get Bose-Einstein factor and specific heat Cv
            self.bose = np.zeros((nqpt,nmode, self.ntemp))
            cv = np.zeros((nqpt,nmode,self.ntemp))

            for i,n in itt.product(range(nqpt),range(nmode)):
                self.bose[i,n,:] = self.get_bose(self.omega[1,i,n],self.temperature)
                cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)

            boseplushalf = self.bose + np.ones((nqpt,nmode,self.ntemp))*0.5
#            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t

#            for t in range(self.ntemp):
#                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
#            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
           # bose = self.get_bose() 
            #bose = 1./(np.exp(x)-1)
            #bose[0,:3,:] = 0 # Check what Gabriel did)
            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            hwt_plushalf = np.einsum('qv,qvt->qvt',self.omega[1,:,:],boseplushalf)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen)/(9*self.equilibrium_volume[0]*self.bulk_modulus)
            alphavol=np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruvol)/(self.equilibrium_volume[0]*self.bulk_modulus)
            alpha2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2)/(self.equilibrium_volume[0]*self.bulk_modulus)


            # Then, get a(T)
            integral = 1./(9*self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen)
            a = self.equilibrium_volume[1]*(integral+1)

            integral_plushalf = 1./(9*self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt_plushalf,self.gruneisen)
            a_plushalf = self.equilibrium_volume[1]*(integral_plushalf+1)


            integralvol = 1./(self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruvol)
            vol = self.equilibrium_volume[0]*(integralvol+1)
            avol = (vol*4)**(1./3)


            if plot:
                import matplotlib.pyplot as plt
                fig,arr = plt.subplots(1,2,figsize=(12,5),sharey=False)

                arr[0].plot(self.temperature,alpha*1E6) 
#                arr[1].plot(self.temperature,alpha2*1E6) 
                arr[0].set_ylabel(r'$\alpha$ ($10^{-6}$ K$^{-1}$)')
                arr[0].set_xlabel(r'Temperature (K)')
                arr[1].set_xlabel(r'Temperature (K)')
#                arr[2].set_xlabel(r'Temperature (K)')

#                arr[0].set_title(r'Slope $\omega$ vs V') 
#                arr[1].set_title(r'Dynamical matrix') 
#                arr[0].set_title(r'Free energy minimization')
#                arr[1].set_title(r'alpha(T) via Gruneisen')
#                arr[2].set_title(r'Phonon effective pressure')

#                arr[2].plot(self.temperature, a*cst.bohr_to_ang, 'or')
#                arr[2].plot(self.temperature, avol*cst.bohr_to_ang, 'ok')


                xexp = np.array([26.0,48.86,74.87,81.22,86.93,86.29,88.83,95.18,102.79,109.77,117.39,142.13,201.78,213.20])
                xexp = xexp + 273*np.ones(len(xexp))
                yexp = np.array([5.6498,5.6511,5.6525,5.6528,5.6525,5.6531,5.6532,5.6534,5.6538,5.6542,5.6546,5.6560,5.6580,5.6598])
#                for ja in range(2): 
#                    arr[ja].set_ylabel(r'a (ang)')
#                    arr[ja].plot(xexp,yexp,'k',marker='x',label = 'exp')
#                    arr[ja].set_ylim(5.63,5.91)
#                arr[2].set_xlim((-100,250))
                arr[1].plot(xexp,yexp,'k',marker='x',label = 'exp')

                #Method 1 : F minimisation
                arr[1].plot(self.temperature, self.temperature_dependent_acell[0,:]*cst.bohr_to_ang,'g',marker='o',label='min F(V,T)')               

                #Method 2 : acell and Gruneisen : via alpha(T) and beta(T)
                arr[1].plot(self.temperature, a*cst.bohr_to_ang, 'r',marker='o',label='alpha(T)-Gruneisen')
#                arr[1].plot(self.temperature, avol*cst.bohr_to_ang, 'b',marker='o',label='via beta')
#                arr[1].legend(numpoints=1)

       
                xph_linh= np.array([5.77446322, 5.77415471, 5.77494795, 5.77810078, 5.78277712,5.7881996 , 5.79428249])     
                x2 = np.array([1.0961881163E+01,1.0974802001E+01,1.0988073359E+01,1.1001737272E+01,1.1015787644E+01,1.1030265342E+01,1.1045190060E+01,1.1060508725E+01,1.1076193192E+01,1.1092324059E+01])*cst.bohr_to_ang
                xph_linh = np.concatenate((xph_linh,x2))
                xph_lin= np.array([10.863535,10.8629899,10.864390,10.870100,10.878476,10.888331,10.899036,1.0910336289E+01,1.0921990411E+01,1.0934082962E+01,1.0946819609E+01,1.0959910371E+01,1.0973437050E+01,1.0987176885E+01,1.1001184648E+01,1.1015530899E+01,1.1030235351E+01])*cst.bohr_to_ang     
                xph_volh= np.array([1.0879469336E+01, 1.0879279851E+01, 1.0879768251E+01, 1.0881710025E+01, 1.0884509564E+01,1.0887826691E+01 ,1.0891394373E+01,1.0895126767E+01,1.0898961976E+01,1.0902895471E+01,1.0906903701E+01,1.0910940294E+01,1.0915029696E+01,1.0919129052E+01,1.0923253317E+01,1.0927417939E+01,1.0931636277E+01])*cst.bohr_to_ang     
                xph_vol= np.array([10.8635352, 10.8633511, 10.8638215, 10.8656968, 10.8684532,10.871682 ,10.8751899,1.0878878852E+01,1.0882676519E+01,1.0886567584E+01,1.0890534011E+01,1.0894531329E+01,1.0898555717E+01,1.0902631294E+01,1.0906747733E+01,1.0910868029E+01,1.0915021346E+01])*cst.bohr_to_ang
                          
#                arr[2].plot(xexp,yexp,'bx')
                tt = np.arange(0,801,50)
#                arr[2].plot(tt,xph_linh,'m',marker='o',label='linh')
#                arr[2].plot(tt,xph_lin,'c',marker='o',label='lin')
                arr[1].plot(tt,xph_volh,'m',marker='o',label=r'Peff, n$_{qv}$+1/2')
                arr[1].plot(tt,xph_vol,'y',marker='o',label=r'Peff,n$_{qv}$')
                arr[1].legend(numpoints=1)
                arr[1].set_xlim(0,500)
                arr[1].set_ylabel(r'a (ang)')

                expalpha = EXPfile('GaAs_alphaT.nc')
                expalpha.read_nc()
                arr[0].plot(expalpha.xaxis,expalpha.yaxis,'xk',label='exp')
                arr[0].set_xlim(0,500)
                
                plt.suptitle(r'GaAs')


                plt.savefig('GaAs_TE.png')
                plt.show() 
            
            for t,T in enumerate(self.temperature):
                print('T={}K, a={} bohr, delta a = {} bohr'.format(T,a[t],a[t]-a[0]))

            a = np.expand_dims(a,axis=0)
            self.acell_plushalf = np.expand_dims(a_plushalf,axis=0)
            return a

        if self.symmetry == 'hexagonal':
            
            
            plot = False

           

            # First, get alpha(T)

            # Get Bose-Einstein factor and specific heat Cv
            self.bose = np.zeros((nqpt,nmode, self.ntemp))
            cv = np.zeros((nqpt,nmode,self.ntemp))

            for i,n in itt.product(range(nqpt),range(nmode)):
                self.bose[i,n,:] = self.get_bose(self.omega[1,i,n],self.temperature)
                cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)

            boseplushalf = self.bose + np.ones((nqpt,nmode,self.ntemp))*0.5

#            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t

#            for t in range(self.ntemp):
#                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
#            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
           # bose = self.get_bose() 
            #bose = 1./(np.exp(x)-1)
            #bose[0,:3,:] = 0 # Check what Gabriel did)
            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            hwt_plushalf = np.einsum('qv,qvt->qvt',self.omega[1,:,:],boseplushalf)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            # Get alpha_a,c with compliance 
            alpha_a = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]
            alpha_c = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]

            alpha_a2 = ( (self.compliance_rigid[0,0]+self.compliance_rigid[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]
            alpha_c2 = ( 2*self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance_rigid[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]

            alpha_af = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
                self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]
            alpha_cf = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
                self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]


            # Then, get a(T) and c(T)
            integral_a = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[0,:,:])
            integral_c = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[1,:,:])
            integral_a0 = 0.5*np.einsum('q,qv,qv',self.wtq,self.omega[1,:,:],self.gruneisen[0,:,:])
            integral_c0 = 0.5*np.einsum('q,qv,qv',self.wtq,self.omega[1,:,:],self.gruneisen[1,:,:])
            integral_aplushalf = np.einsum('q,qvt,qv->t',self.wtq,hwt_plushalf,self.gruneisen[0,:,:])
            integral_cplushalf = np.einsum('q,qvt,qv->t',self.wtq,hwt_plushalf,self.gruneisen[1,:,:])


            aterm = (self.compliance[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a = self.equilibrium_volume[1]*(aterm/self.equilibrium_volume[0] + 1)
            daa = aterm/self.equilibrium_volume[0]

            aterm_plushalf = (self.compliance[0,0]+self.compliance[0,1])*integral_aplushalf + self.compliance[0,2]*integral_cplushalf
            aplushalf = self.equilibrium_volume[1]*(aterm_plushalf/self.equilibrium_volume[0] + 1)

            cterm = 2*self.compliance[0,2]*integral_a + self.compliance[2,2]*integral_c
            c = self.equilibrium_volume[3]*(cterm/self.equilibrium_volume[0] + 1)
            dcc = cterm/self.equilibrium_volume[0]
            cterm_plushalf = 2*self.compliance[0,2]*integral_aplushalf + self.compliance[2,2]*integral_cplushalf
            cplushalf = self.equilibrium_volume[3]*(cterm_plushalf/self.equilibrium_volume[0] + 1)

            #daa_slope = np.polyfit(self.temperature[14:],daa[14:],1)
            #print('Delta a/a intersect: {:>8.5e}, new a0 = {} bohr'.format(daa_slope[1],-daa_slope[1]*self.equilibrium_volume[1]+self.equilibrium_volume[1]))
            #dcc_slope = np.polyfit(self.temperature[14:],dcc[14:],1)
            #print('Delta c/c intersect: {:>8.5e}, new c0 = {} bohr'.format(dcc_slope[1],-dcc_slope[1]*self.equilibrium_volume[3]+self.equilibrium_volume[3]))

            a2 = (self.compliance_rigid[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a2 = self.equilibrium_volume[1]*(a2/self.equilibrium_volume[0] + 1)

            c2 = 2*self.compliance_rigid[0,2]*integral_a + self.compliance[2,2]*integral_c
            c2 = self.equilibrium_volume[3]*(c2/self.equilibrium_volume[0] + 1)

            # Test the da/a at T=0, using the 1/2 factor
            da0 = ((self.compliance[0,0]+self.compliance[0,1])*integral_a0 + self.compliance[0,2]*integral_c0)/self.equilibrium_volume[0]
            dc0 = (2*self.compliance[0,2]*integral_a0 + self.compliance[2,2]*integral_c0)/self.equilibrium_volume[0]

            acell = np.array([a,c])
            self.acell_plushalf = np.array([aplushalf,cplushalf])

#            self.acell2 = np.array([a2,c2])

            print('From Gruneisen parameters')
            print('da/a at T=0 = {}, da = {} bohr, a0 = {} bohr'.format(da0,da0*self.equilibrium_volume[1],da0*self.equilibrium_volume[1] + self.equilibrium_volume[1]))
            print('dc/c at T=0 = {}, dc = {} bohr, c0 = {} bohr'.format(dc0,dc0*self.equilibrium_volume[3],dc0*self.equilibrium_volume[3]+ self.equilibrium_volume[3]))

            for t,T in enumerate(self.temperature):
                print('T={}K, a={}, c={}'.format(T,a[t],c[t]))
                print('plushalf, a={}, c={}'.format(aplushalf[t],cplushalf[t]))


            if plot:
                import matplotlib.pyplot as plt
                fig,arr = plt.subplots(2,3,figsize=(15,10),sharey=False)
                arr[0,0].plot(self.temperature,alpha_a*1E6,'r',label='relaxed') 
#                arr[0,0].plot(self.temperature,alpha_a2*1E6,'g',label='rigid') 
#                arr[0,0].plot(self.temperature,alpha_af*1E6,'c:',label='finite') 


#                twin0 = arr[0].twinx()
#                twin0.plot(self.temperature,alpha_c*1E6,'b',label='c') 
                arr[1,0].plot(self.temperature,alpha_c*1E6,'b',label='relaxed') 
#                arr[1,0].plot(self.temperature,alpha_c2*1E6,'g',label='rigid') 
#                arr[1,0].plot(self.temperature,alpha_cf*1E6,'c:',label='finite') 



                arr[0,0].set_ylabel(r'$\alpha_a$ ($10^{-6}$ K$^{-1}$)',color='r')
                arr[1,0].set_ylabel(r'$\alpha_c$ ($10^{-6}$ K$^{-1}$)',color='b')

                arr[0,0].set_xlabel(r'Temperature (K)')
#                arr[0].legend()
                arr[1,0].set_xlabel(r'Temperature (K)')
#                arr[0].set_title(r'Expansion coefficients') 
#                arr[1].set_title(r'Lattice parameters')
                arr[0,1].plot(self.temperature, daa*1E3,'r',label='relaxed')
#                atest = self.temperature*daa_slope[0]+daa_slope[1]
#                ctest = self.temperature*dcc_slope[0]+dcc_slope[1]
#                arr[0,1].plot(self.temperature, atest*1E3,'k:')
#                arr[1,1].plot(self.temperature, ctest*1E3,'k:')
                for a1 in range(2):
                    for a2 in range(2):
                        arr[a1,a2].set_xlim(self.temperature[0],self.temperature[-1])

                arr[0,2].plot(self.temperature, a*cst.bohr_to_ang,'r',label='n')
                arr[0,2].plot(self.temperature, aplushalf*cst.bohr_to_ang,'k:',label='n + 1/2')
                arr[0,2].plot(self.temperature, self.fitg[0,:]*cst.bohr_to_ang, 'o',color='m',label='Gibbs')
#                arr[0,1].plot(self.temperature, a2*cst.bohr_to_ang,'g',label='rigid')


#                twin1 = arr[1].twinx()
                arr[1,1].plot(self.temperature,dcc*1E3,'b',label='relaxed')
                arr[1,2].plot(self.temperature,c*cst.bohr_to_ang,'b',label='n')
                arr[1,2].plot(self.temperature,cplushalf*cst.bohr_to_ang,'k:',label='n + 1/2')
                arr[1,2].plot(self.temperature, self.fitg[1,:]*cst.bohr_to_ang, 'o',color='m',label='Gibbs')
               

#                arr[1,1].plot(self.temperature,c2*cst.bohr_to_ang,'g',label='rigid')


                #arr[2].plot(self.temperature-273, a*cst.bohr_to_ang, 'or')
                arr[0,1].set_ylabel(r'$\Delta$ a/a x 10$^3$',color='r')
                arr[1,1].set_ylabel(r'$\Delta$ c/c x 10$^3$',color='b')

                arr[0,2].set_ylabel(r'a ($\AA$)',color='r')
                arr[1,2].set_ylabel(r' c ($\AA$)',color='b')
                arr[0,2].legend()
                arr[1,2].legend()

                #arr[2].set_xlabel(r'T (Celcius)')
                #arr[2].set_xlim((-100,250))
        
##                xexp = np.array([26.0,48.86,74.87,81.22,86.93,86.29,88.83,95.18,102.79,109.77,117.39,142.13,201.78,213.20])
##                yexp = np.array([5.6498,5.6511,5.6525,5.6528,5.6525,5.6531,5.6532,5.6534,5.6538,5.6542,5.6546,5.6560,5.6580,5.6598])
##                arr[2].plot(xexp,yexp,'bx')
#                aax = np.array([191.834,266.905,378.641,474.619,547.897,561.844,650.808,711.909,750.280])
#                aay = np.array([2.52395,3.10329,3.98113,4.52480,4.87530,4.89260,5.24276,5.69917,5.82156])
#                arr[0,0].plot(aax,aay,'or',label='SH exp')
##                arr[0].set_ylim(0,14)
#                acx = np.array([190.311,268.754,376.849,467.552,474.509,535.571,549.499,629.7,643.704,706.430,744.827])
#                acy = np.array([3.55177,3.60250,3.75748,4.08767,4.01759,4.29598,4.22574,4.34637,4.62579,4.537,4.78092])
#                arr[1,0].plot(acx,acy,'ob',label='SH exp')
#
#
#                # Reeber 1999 data
#                reeber_alphaA = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_alphaAT_Reeber1999.nc')
#                reeber_alphaA.read_nc()
#                reeber_alphaC = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_alphaCT_Reeber1999.nc')
#                reeber_alphaC.read_nc()
#                reeber_a = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_aT_Reeber1999.nc')
#                reeber_a.read_nc()
#                reeber_c = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_cT_Reeber1999.nc')
#                reeber_c.read_nc()
#
#                arr[0,0].plot(reeber_alphaA.xaxis, reeber_alphaA.yaxis,'xr',label='Reeber1999')
#                arr[1,0].plot(reeber_alphaC.xaxis, reeber_alphaC.yaxis,'xb',label='Reeber1999')
#                arr[0,1].plot(reeber_a.xaxis, reeber_a.yaxis,'xr',label='Reeber1999')
#                arr[1,1].plot(reeber_c.xaxis, reeber_c.yaxis,'xb',label='Reeber1999')
#
#                arr[0,1].plot(self.temperature,self.independent_fit[0,:]*cst.bohr_to_ang,'go',markersize=7,label=r'ind\_fit')
#                arr[1,1].plot(self.temperature,self.independent_fit[1,:]*cst.bohr_to_ang,'go',markersize=7,label=r'ind\_fit')
#                arr[0,1].plot(self.temperature,self.fit2d[0,:]*cst.bohr_to_ang,'mo',markersize=5,label='fit2d')
#                arr[1,1].plot(self.temperature,self.fit2d[1,:]*cst.bohr_to_ang,'mo',markersize=5,label='fit2d')

#                roder_a = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_aT_Roder2005.nc')
#                roder_a.read_nc()
#                roder_a1 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_aT_Roder2005_ref1.nc')
#                roder_a1.read_nc()
#                roder_a4 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_aT_Roder2005_ref4.nc')
#                roder_a4.read_nc()
#                roder_a6 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_aT_Roder2005_ref6.nc')
#                roder_a6.read_nc()
#                roder_c = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_cT_Roder2005.nc')
#                roder_c.read_nc()
#                roder_c1 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_cT_Roder2005_ref1.nc')
#                roder_c1.read_nc()
#                roder_c4 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_cT_Roder2005_ref4.nc')
#                roder_c4.read_nc()
#                roder_c6 = EXPfile('/Users/Veronique/Google_Drive/doctorat/work/TI/GaN/TE/data/GaN_cT_Roder2005_ref6.nc')
#                roder_c6.read_nc()
#
#                arr[0,1].plot(roder_a.xaxis, roder_a.yaxis,'vk',label='Roder2005')
#                arr[1,1].plot(roder_c.xaxis, roder_c.yaxis,'vk',label='Roder2005')
#                arr[0,1].plot(roder_a1.xaxis, roder_a1.yaxis,'Pk',label='Roder2005-ref1')
#                arr[1,1].plot(roder_c1.xaxis, roder_c1.yaxis,'Pk',label='Roder2005-ref1')
#                arr[0,1].plot(roder_a4.xaxis, roder_a4.yaxis,'sk',label='Roder2005-ref4')
#                arr[1,1].plot(roder_c4.xaxis, roder_c4.yaxis,'sk',label='Roder2005-ref4')
#                arr[0,1].plot(roder_a6.xaxis, roder_a6.yaxis,'*k',label='Roder2005-ref6')
#                arr[1,1].plot(roder_c6.xaxis, roder_c6.yaxis,'*k',label='Roder2005-ref6')
#
#                a_pph = np.array([6.08470767295951E+00,6.09447074714269E+00,6.12096344024277E+00,6.17026163490221E+00,6.24972442596757E+00,6.37597582992085E+00,6.61966832821061E+00])
#                c_pph = np.array([9.91152199244032E+00,9.92461271952359E+00,9.96261928816134E+00,1.00342327383386E+01,1.01504947767351E+01,1.03357790051093E+01,1.06854643693506E+01])
#
#                a_pph2 = np.array([6.08430775088164E+00,6.08753879176845E+00,6.09607719775408E+00,6.11101137525506E+00,6.13257697790371E+00,6.16062057030277E+00,6.19506037955212E+00,6.23617247814261E+00])
#                c_pph2 = np.array([9.91088030247267E+00,9.91518883463132E+00,9.92735890091316E+00,9.94902862256150E+00,9.98025600778787E+00,1.00210434988623E+01,1.00710005620079E+01,1.01311997309172E+01])
#
#
#                a_pphs = np.array([6.0841191470E+00,6.0842863750E+00,6.0847267566E+00,6.0854796392E+00,6.0865275355E+00,6.0878156316E+00,6.0892889066E+00,6.0909033314E+00,6.0926312505E+00,6.0944437833E+00])
#                c_pphs =np.array([9.9105765546E+00,9.9108211467E+00,9.9114851844E+00,9.9126269523E+00,9.9142150175E+00,9.9161678521E+00,9.9184044903E+00,9.9208672155E+00,9.9235023431E+00,9.9262702625E+00])
#
#                t0 = np.arange(50,351,50)
#                t1 = np.arange(50,401,50)
#                t2 = np.arange(50,501,50)
#                arr[0,1].plot(t0,a_pph*cst.bohr_to_ang,'gh',label='Peff')
#                arr[1,1].plot(t0,c_pph*cst.bohr_to_ang,'gh',label='Peff')
#                arr[0,1].plot(t1,a_pph2*cst.bohr_to_ang,'cP',label='Peff/3')
#                arr[1,1].plot(t1,c_pph2*cst.bohr_to_ang,'cP',label='Peff/3')
#                arr[0,1].plot(t2,a_pphs*cst.bohr_to_ang,'yD',label='Peff lin')
#                arr[1,1].plot(t2,c_pphs*cst.bohr_to_ang,'yD',label='Peff lin')
#             
#
#                deltaa = (a[0]*cst.bohr_to_ang-reeber_a.yaxis[0])*np.ones(self.ntemp)
#                deltac = (c[0]*cst.bohr_to_ang-reeber_c.yaxis[0])*np.ones(self.ntemp)
#
#                arr[0,1].plot(self.temperature,a*cst.bohr_to_ang-deltaa,'r',linestyle='dashed',label='shifted to exp.')
#                arr[1,1].plot(self.temperature,c*cst.bohr_to_ang-deltac,'b',linestyle='dashed',label='shifted to exp.')


                plt.subplots_adjust(left=0.05,right=0.85,hspace=0.05)
                for a1 in range(2):
                    arr[a1,0].legend(numpoints=1)
                    arr[a1,1].legend(numpoints=1,loc=10,bbox_to_anchor=(1.2,0.5))

#                # Set axis color
#                arr[0].spines["left"].set_edgecolor('red')
#                arr[1].spines["left"].set_edgecolor('red')
#                arr[0].tick_params(axis='y', colors='red')
#                arr[1].tick_params(axis='y', colors='red')
#                twin0.spines["left"].set_edgecolor('red')
#                twin1.spines["left"].set_edgecolor('red')
#                twin0.tick_params(axis='y', colors='red')
#                twin1.tick_params(axis='y', colors='red')
#

#                t = self.temperature
#                #Fit parameters from Semiconductors Handbook
#                afit = 3.184 + 0.739E-5*t + 5.92E-9*t**2
#                cfit = 5.1812+1.455E-5*t+4.62E-9*t**2

#                arr[1].plot(t,afit,'r:')
#                twin1.plot(t,cfit,'b:')


                plt.suptitle(r'{}'.format(self.rootname))
                outfile = 'FIG/{}_alpha.png'.format(self.rootname)
                create_directory(outfile)
                plt.savefig(outfile)
                plt.show() 
            
#            for t,T in enumerate(self.temperature):
#                print('T={}K, a={} ang, c={} ang'.format(T,a[t]*cst.bohr_to_ang,c[t]*cst.bohr_to_ang))



            return acell



    def get_phonon_effective_pressure(self,nqpt,nmode):

        # This function computes the phonon "effective pressure", that is, 
        # P_ph(T) ~  -dF_ph/dV

        if self.symmetry == 'cubic':

            boseplushalf = self.bose + 0.5*np.ones((np.shape(self.bose)))
            #RVolumic Gruneisens 
#            pphl = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen)/self.volume[1,0]
#            pphl2 = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],boseplushalf,self.gruneisen)/self.volume[1,0]

#            pph2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen)/self.volume[1,0]
            # Linear Gruneisens
            pph = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen/3.)/self.volume[1,0]
#            print(np.einsum('qv,qvt,qv->t',self.omega[1,:,:],self.bose,self.gruneisen/3.)
#            print(np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen/3.)

            pph2 = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],boseplushalf,self.gruneisen/3.)/self.volume[1,0]


            for t,T in enumerate(self.temperature):
                print('\nT={}K, Pphonon = {} GPa'.format(T,pph[t]*cst.habo3_to_gpa))
                print('plus half: {} GPa = {} ha/bohr^3'.format(pph2[t]*cst.habo3_to_gpa,pph2[t]))
                #print('lin: {} GPa = {} ha/bohr^3'.format(pphl[t]*cst.habo3_to_gpa,pphl[t]))
                #print('lin plushalf: {} GPa = {} ha/bohr^3'.format(pphl2[t]*cst.habo3_to_gpa,pphl2[t]))




#                print('T={}K, Pphonon (+1/2) = {} GPa'.format(T,pph2[t]*cst.habo3_to_gpa))
                #print('T={}K, Pphonon = {} ha/bo^3'.format(T,pph[t]))
                #print('T={}K, Pphonon (+1/2) = {} ha/bo/3'.format(T,pph2[t]))

            return pph

        if self.symmetry == 'hexagonal':

            boseplushalf = self.bose + 0.5*np.ones((np.shape(self.bose)))
            #RVolumic Gruneisens 
#            pph = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],self.bose,self.gruneisen)/self.volume[1,0]
#            pph2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen)/self.volume[1,0]
            # Linear Gruneisens
            sum_a = np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[0,:,:])
            sum_c = np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[1,:,:])

#            pph_a = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[0,:,:])/self.equilibrium_volume[0]
#
#            pph_a2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen[0,:,:]/3.)/self.equilibrium_volume[0]
#
#            pph_c = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[1,:,:]/3.)/self.equilibrium_volume[0]
#            pph_c2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen[1,:,:]/3.)/self.equilibrium_volume[0]
            pph_a = -1*((self.compliance[0,0]+self.compliance[0,1])*sum_a + self.compliance[0,2]*sum_c)/self.equilibrium_volume[0]
            pph_c = -1*(2*self.compliance[0,2]*sum_a + self.compliance[2,2]*sum_c)/self.equilibrium_volume[0]

            pph_a2 =( -1./3)*((self.compliance[0,0]+self.compliance[0,1])*sum_a + self.compliance[0,2]*sum_c)/self.equilibrium_volume[0]
            pph_c2 = (-1./3)*(2*self.compliance[0,2]*sum_a + self.compliance[2,2]*sum_c)/self.equilibrium_volume[0]

            pph_as = -1*sum_a/self.equilibrium_volume[0]
            pph_cs = -1*sum_c/self.equilibrium_volume[0]


            iprint = False
            if iprint:
                print('Phonon effective pressure')
                for t,T in enumerate(self.temperature):
                    print('\nt={}k, pphonon_a = {} gpa = {} ha/bo^3'.format(T,pph_a[t]*cst.habo3_to_gpa,pph_a[t]))
                    print('    pphonon_c = {} gpa = {} ha/bo^3'.format(pph_c[t]*cst.habo3_to_gpa,pph_c[t]))
                    print('    pphonon_a2 = {} gpa = {} ha/bo^3'.format(pph_a2[t]*cst.habo3_to_gpa,pph_a2[t]))
                    print('    pphonon_c2 = {} gpa = {} ha/bo^3'.format(pph_c2[t]*cst.habo3_to_gpa,pph_c2[t]))
                    print('    pphonon_as = {} gpa = {} ha/bo^3'.format(pph_as[t]*cst.habo3_to_gpa,pph_as[t]))
                    print('    pphonon_cs = {} gpa = {} ha/bo^3'.format(pph_cs[t]*cst.habo3_to_gpa,pph_cs[t]))

#                print('T={}K, Pphonon_a (+1/2) = {} GPa'.format(T,pph_a2[t]*cst.habo3_to_gpa))
#                print('T={}K, Pphonon_c = {} GPa'.format(T,pph_c[t]*cst.habo3_to_gpa))

#                print('T={}K, Pphonon_c (+1/2) = {} GPa'.format(T,pph_c2[t]*cst.habo3_to_gpa))

                #print('T={}K, Pphonon = {} ha/bo^3'.format(T,pph[t]))
                #print('T={}K, Pphonon (+1/2) = {} ha/bo/3'.format(T,pph2[t]))

            return np.array([pph_a,pph_c])


    def gruneisen_from_dynmat(self,nqpt,nmode,nvol):

        # This computes the gruneisen parameters from the change in the dynamical matrix
        # like phonopy and anaddb
            
        # for now, I reopen all files, but later on change the loop ordering (anyway, it should not do everything linearly, but rather use functions depending of input parameters

        gru = np.zeros((nqpt,nmode))
        dplus =  2
        d0 = 1
        dminus = 0 # put this in a function
        dV = self.volume[dplus,0] - self.volume[dminus,0]
        V0 = self.volume[d0,0]


#        fg = open('gruneisen_dynmat.dat','w')
        fg = open('dotproduct_dynmat.dat','w')
        for i in range(nqpt):
            dD = np.zeros((nmode,nmode),dtype=complex)
            for v in range(nvol):

                # open the ddb file
                ddb = DdbFile(self.ddb_flists[v][i])

                # Check if qpt is Gamma
                is_gamma = ddb.is_gamma

                # this is taken from Gabriel's code, directly. for testing purposes.
                amu = np.zeros(ddb.natom)
                for ii in np.arange(ddb.natom):
    
                    jj = ddb.typat[ii]
                    amu[ii] = ddb.amu[jj-1]

                dynmat = ddb.get_mass_scaled_dynmat_cart()

                if v==dplus:
                    dD += dynmat
                if v==dminus:
                    dD -= dynmat
                if v==d0:
                    #eigval is omega^2
                    eigval, eigvect = np.linalg.eigh(dynmat)

                    omega = np.sqrt(np.abs(eigval)) * np.sign(eigval)

                    for ii in np.arange(ddb.natom):
                        for dir1 in np.arange(3):
                            ipert = ii*3 + dir1
#                            eigvect[ipert] = eigvect[ipert]*np.sqrt(cst.me_amu/amu[ii])
                            eigvect[ipert] = eigvect[ipert]
                    
                    if is_gamma:
                        eigval[0] = 0.0
                        eigval[1] = 0.0
                        eigval[2] = 0.0

                    for ieig,eig in enumerate(eigval):
                        if eig < 0.0:
                            warnings.warn('Negative eigenvalue changed to 0')
                            eigval[ieig] = 0.0

            # end v in range(nvol)
       
            dD_at_q = []

            for eig in eigvect:

                dD_at_q.append(np.vdot(np.transpose(eig), np.dot(dD,eig)).real)   
#                if i==1:
                #    print(eig)
               #     print(np.dot(dD,eig))
                #print(dD_at_q)
                    

            dD_at_q = np.array(dD_at_q)

            fg.write('\nqpt {}\n'.format(i+1))
#            fg.write('Real part:\n')
            for v in range(nmode):

                fg.write('Eigenvectors\n')
#                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(dD[v,0].real/dV,dD[v,1].real/dV,dD[v,2].real/dV,dD[v,3].real/dV,dD[v,4].real/dV,dD[v,5].real/dV))
                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(eigvect[v,0].real,eigvect[v,1].real,eigvect[v,2].real,eigvect[v,3].real,eigvect[v,4].real,eigvect[v,5].real))
                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(eigvect[v,0].imag,eigvect[v,1].imag,eigvect[v,2].imag,eigvect[v,3].imag,eigvect[v,4].imag,eigvect[v,5].imag))

#                if i==0 and v<3:
                if omega[v] < tol12:
                    gru[i,v] = 0
                else:
                    gru[i,v] = -V0*dD_at_q[v]/(2*np.abs(eigval[v].real)*dV)
#            fg.write('Imaginary part:\n')
#            for v in range(nmode):
#                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(dD[v,0].imag/dV,dD[v,1].imag/dV,dD[v,2].imag/dV,dD[v,3].imag/dV,dD[v,4].imag/dV,dD[v,5].imag/dV))

            fg.write('dD/dV|u>\n')
            for v in range(nmode):
                omat = np.dot(dD/dV,eigvect[v,:])
                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(omat[0].real,omat[1].real,omat[2].real,omat[3].real,omat[4].real,omat[5].real))
                fg.write('{:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}\n'.format(omat[0].imag,omat[1].imag,omat[2].imag,omat[3].imag,omat[4].imag,omat[5].imag))

                fg.write('Dot product:\n')
                dot = np.vdot(np.transpose(eigvect[v,:]),omat)
                fg.write('{:>12.8e}\n'.format(dot.real))     
        

#            if i==1:
#
#                print('Dynamical matrix')
#                print('delta omega^2 {}'.format(dD_at_q))
#                print('omega0 {}'.format(eigval[:]))
#                print('gruneisen {}'.format(gru[i,:]))
                

        # end i in range(nqpt)

        return gru


    def write_acell(self):

        outfile = 'OUT/{}_acell_from_gruneisen.dat'.format(self.rootname)
        nc_outfile = 'OUT/{}_acell.nc'.format(self.rootname)

        #  First, write output in netCDF format
        create_directory(nc_outfile)

        with nc.Dataset(nc_outfile, 'w') as dts:

            dts.createDimension('number_of_temperatures', self.ntemp)
            dts.createDimension('number_of_lattice_parameters', len(self.distinct_acell))


            data = dts.createVariable('acell_from_gruneisen','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_via_gruneisen[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('acell_from_gruneisen_plushalf','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_plushalf[:,:]
            data.units = 'Bohr radius'


            data = dts.createVariable('acell_from_helmholtz','d',('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.temperature_dependent_acell[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('acell_from_gibbs','d',('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.fitg[:,:]
            data.units = 'Bohr radius'


        # Then, write them in ascii file
        create_directory(outfile)

        with open(outfile, 'w') as f:

            f.write('Temperature dependent lattice parameters via Gruneisen parameters\n\n')

            if self.symmetry == 'cubic':

                f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.acell_via_gruneisen[0,t]))

                f.close()


            if self.symmetry == 'hexagonal':

                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.acell_via_gruneisen[0,t],self.acell_via_gruneisen[1,t]))

                f.write('\n\nFrom Gruneisen with n+1/2\n\n')
                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.acellplushalf[0,t],self.acellplushalf[1,t]))


                f.close()

        # Write also results from Free energy minimisation
        outfile = 'OUT/{}_acell_from_freeenergy.dat'.format(self.rootname)

        create_directory(outfile)

        with open(outfile, 'w') as f:

            f.write('Temperature dependent lattice parameters via Helmholtz free energy\n\n')

            if self.symmetry == 'cubic':


                f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t]))

                f.close()


            if self.symmetry == 'hexagonal':

                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fit2d[0,t],self.fit2d[1,t]))

                # Independent fit: fitg, 2D fit: fit2dg
                f.write('\n\nTemperature dependent lattice parameters via Gibbs free energy\n\n')
                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fitg[0,t],self.fitg[1,t]))



                f.close()



class GibbsFreeEnergy(object):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    units = 'eV'
    temperature = None

    def __init__(self,

        ddb_flists = None,
        out_flists = None,

        units = 'eV',
        temperature = np.arange(0,300,50),
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

class Static(object):

    def __init__(self,

            rootname = 'static.out',
            units = 'eV',

            dedp = False,
        
            etotal_flist = None,
            gap_fname = None,             
            initial_params = None,

            static_plot = False,

            **kwargs):

        self.rootname = rootname
        self.units = units
        self.etotal_flist = etotal_flist
        self.gap_fname = gap_fname
        self.dedp = dedp
        self.initial_params = initial_params
        self.static_plot = static_plot

        # Data check
        if not self.etotal_flist:
            raise Exception('Must provide a list of _GSR.nc files containing total energy')
        if not self.gap_fname:
            raise Exception('Must provide a netCDF file containing gap energies. Please use netcdf_gap.py') 

        
        self.nfile = len(self.etotal_flist)

        self.etotal = np.empty((self.nfile))
        self.gap_energy = np.empty((self.nfile))
        self.volume = np.empty((self.nfile))

        gap = GapFile(self.gap_fname)
        self.gap_energy = gap.gap_energy
        if len(self.gap_energy) != self.nfile:
            raise Exception('{} contains {} gap values, while there are {} files in etotal_flist. '.format(self.gap_fname, len(self.gap_energy),self.nfile))

        for n, fname in enumerate(self.etotal_flist):

            gs = GsrFile(fname)
            self.etotal[n] = gs.etotal
            self.volume[n] = gs.volume


        self.bulk_modulus, self.bulk_modulus_derivative, self.equilibrium_volume, self.equilibrium_energy =  self.get_bulk_modulus()

        if self.dedp:
    
            self.pressure, self.dedp_fit, self.dedp = self.get_dedp()


    def get_bulk_modulus(self):

        
        if self.initial_params:
            p0 = self.initial_params
            popt,pcov = curve_fit(eos.murnaghan_EV, self.volume, self.etotal, p0)  

        else:
            p0 = [self.volume[-1],self.etotal[-1],75,3.5]  # Guess for initial parameters
            popt,pcov = curve_fit(eos.murnaghan_EV, self.volume, self.etotal, p0)  

        if self.static_plot:
            plt.plot(self.volume,self.etotal,'ok',linestyle='None',label='data')
            xfit = np.linspace(0.90*self.volume[0], 1.10*self.volume[-1],200)
            yfit = eos.murnaghan_EV(xfit, popt[0],popt[1],popt[2],popt[3])
            plt.plot(xfit,yfit,label='fit')

            plt.xlabel('Unit cell volume (bohr^3)')
            plt.ylabel('Total energy (Ha)')
            plt.legend(numpoints=1)
            plt.title('Murnaghan EOS')
            plt.tight_layout()

            figname = 'figures/{}_murnaghan.png'.format(self.rootname)
            create_directory(figname)
            plt.savefig(figname)

            plt.show()

        print(popt[2]/cst.gpa_to_habo3)
        
        return popt[2], popt[3], popt[0], popt[1]


    def get_dedp(self):

        pdata = eos.murnaghan_PV(self.volume,self.equilibrium_volume, self.bulk_modulus, self.bulk_modulus_derivative)
        #print(pdata*cst.habo3_to_gpa)
        #print(self.gap_energy*cst.ha_to_ev )


        # find which pressure is closer to zero. Make the linear fit with neighboring data
        p0 = np.abs(pdata).argmin()

        fit = np.polyfit(pdata[p0-1:p0+2], self.gap_energy[p0-1:p0+2],1)

        dedp = fit[0]
        print('dedp  = {} meV/GPa'.format(dedp*cst.ha_to_ev*1000/cst.habo3_to_gpa))

        if self.static_plot:

            plt.plot(pdata*cst.habo3_to_gpa, self.gap_energy*cst.ha_to_ev,marker='s',color='k',linestyle='None',label='data')
            plt.plot(pdata[p0-1:p0+2]*cst.habo3_to_gpa,self.gap_energy[p0-1:p0+2]*cst.ha_to_ev,marker='s',color='m',linestyle='None',label='fitting data')
    
            xfit = np.linspace(pdata[0],pdata[-1],100)
            yfit = xfit*fit[0] + fit[1]
            plt.plot(xfit*cst.habo3_to_gpa,yfit*cst.ha_to_ev,label='fit')

            plt.xlabel('Pressure (GPa)')
            plt.ylabel('Gap energy (eV)')
            plt.legend(numpoints=1)
            plt.title('dEgap/dP')
            plt.tight_layout()

            figname = 'figures/{}_dedp.png'.format(self.rootname)
            create_directory(figname)
            plt.savefig(figname)
            plt.show()
        return pdata, fit, dedp  
        
    def write_output(self):

        fname = 'OUT/{}_STATIC.dat'.format(self.rootname)

        create_directory(fname)

        with open(fname,'w') as f:

            f.write('Static lattice properties, from Murnaghan equation of state\n\n')
            f.write('{:<35s} : {:>12.4f} bohr^3\n'.format('Equilibrium volume',self.equilibrium_volume))
            f.write('{:<35s} : {:>12.4f} eV\n'.format('Equilibrium energy',self.equilibrium_energy*cst.ha_to_ev))
            f.write('{:<35s} : {:>12.4f} GPa\n'.format('Bulk modulus',self.bulk_modulus*cst.habo3_to_gpa))
            f.write('{:<35s} : {:>12.4f}\n'.format('Bulk modulus pressure derivative',self.bulk_modulus_derivative))
            f.write('{:<35s} : {:>12.4f} meV/GPa\n'.format('dEgap/dP (P=0)', self.dedp*cst.ha_to_ev*1000/cst.habo3_to_gpa))

        f.close()

    def write_netcdf(self):
        
        fname = 'OUT/{}_STATIC.nc'.format(self.rootname)

        create_directory(fname)

        with nc.Dataset(fname,'w') as dts:

            dts.createDimension('number_of_points',self.nfile)
            dts.createDimension('one',1)
            dts.createDimension('two',2)

            data = dts.createVariable('equilibrium_volume','d',('one'))
            data[:] = self.equilibrium_volume
            data.units = 'bohr^3'
            
            data = dts.createVariable('equilibrium_energy','d',('one'))
            data[:] = self.equilibrium_energy
            data.units = 'hartree'
           
            data = dts.createVariable('bulk_modulus','d',('one'))
            data[:] = self.bulk_modulus*cst.habo3_to_gpa
            data.units = 'GPa'

            data = dts.createVariable('bulk_modulus_derivative','d',('one'))
            data[:] = self.bulk_modulus_derivative
            data.units = 'None'

            data = dts.createVariable('total_energy','d',('number_of_points'))
            data[:] = self.etotal
            data.units = 'hartree'

            data = dts.createVariable('pressure','d',('number_of_points'))
            data[:] = self.pressure*cst.habo3_to_gpa
            data.units = 'GPa'

            data = dts.createVariable('volume','d',('number_of_points'))
            data[:] = self.volume
            data.units = 'bohr^3'

            data = dts.createVariable('gap_energy','d', ('number_of_points'))
            data[:] = self.gap_energy
            data.units = 'hartree'

            data = dts.createVariable('dE_dP','d',('one'))
            data[:] = self.dedp*cst.ha_to_ev*1000/cst.habo3_to_gpa
            data.units = 'meV/GPa'

            data = dts.createVariable('dedp_fit','d',('two'))
            self.dedp_fit[0] = self.dedp_fit[0]/cst.habo3_to_gpa
            data[:] = self.dedp_fit
            data.units = 'hartree/GPa, hartree'            

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
        elastic_fname = None,
        etotal_flist = None,
        gap_fname = None,
        rootname = 'te2.out',

        #Parameters
        wtq = [1.0],
        temperature = None,

        #Options
        gibbs = False, # Default value is Helmoltz free energy, at P=0 (or, at constant P)
        check_anaddb = False,
        units = 'eV',
        symmetry = None,
        bulk_modulus = None,
        bulk_modulus_units = None,
        pressure = 0.0,
        pressure_units = None,

        expansion = True,
        gruneisen = False,
        bulkmodulus = False,
        dedp = False,
        initial_params = None,
        static_plot = False,

        **kwargs):

    # Choose appropriate type of free energy 

    if bulkmodulus:
        
        static_calc = Static(
                    rootname = rootname,
                    symmetry = symmetry,
                    
                    etotal_flist = etotal_flist,
                    gap_fname = gap_fname,
            
                    dedp = dedp,
                    initial_params = initial_params,

                    static_plot = static_plot,

                    **kwargs)

        ## write static output files
        static_calc.write_output()
        static_calc.write_netcdf()
 
    # Compute thermal expansion
    if expansion:
    ## ADD OPTIONS, TO MINIMIZE FREE ENERGY OR USE GRUNEISEN PARAMETERS
        if gibbs:
            calc = GibbsFreeEnergy(
                    out_flists = out_flists, 
                    ddb_flists = ddb_flists,
        
                    rootname = rootname,
                    symmetry = symmetry,
        
                    wtq = wtq,
                    temperature = temperature,
                    units = units,
                    check_anaddb = check_anaddb,
## FIX ME : IF THERE WAS DATA FOR BULK MODULUS, USE IT!! OR, SIMPLY COMPUTE IT FROM DDB VOLUME DATA...                    
                    bulk_modulus = bulk_modulus,
                    bulk_modulus_units = bulk_modulus_units,
    
        
                    **kwargs)
     
        elif gruneisen:
            calc = Gruneisen(
                    out_flists = out_flists, 
                    ddb_flists = ddb_flists,
        
                    rootname = rootname,
                    symmetry = symmetry,
        
                    wtq = wtq,
                    temperature = temperature,
                    units = units,
                    check_anaddb = check_anaddb,
                    elastic_fname = elastic_fname,
                    pressure = pressure,
                    pressure_units = pressure_units,
        
                    bulk_modulus = bulk_modulus,
                    bulk_modulus_units = bulk_modulus_units,
    
                    **kwargs)


        else:
            calc = HelmholtzFreeEnergy(
                    out_flists = out_flists, 
                    ddb_flists = ddb_flists,
        
                    rootname = rootname,
                    symmetry = symmetry,
        
                    wtq = wtq,
                    temperature = temperature,
                    units = units,
                    check_anaddb = check_anaddb,
        
                    bulk_modulus = bulk_modulus,
                    bulk_modulus_units = bulk_modulus_units,
    
                    **kwargs)


    # Write output file
#    calc.write_freeenergy()
        # write gibbs or helmholtz, equilibrium acells (P,T), list of temperatures, pressures, initial volumes
        # in netcdf format, will allow to load the data for plotting
        calc.write_acell()

       # write equilibrium acells, in ascii file
    return
    


###########################


