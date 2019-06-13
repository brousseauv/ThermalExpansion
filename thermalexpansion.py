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
import matplotlib.pyplot as plt
import constants as cst


from ElectronPhononCoupling import DdbFile
from outfile import OutFile
from gsrfile import GsrFile
from gapfile import GapFile
from elasticfile import ElasticFile
from zpr_plotter import EXPfile
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

    def ha2molc(self,f0,ft):

        x = (f0*np.ones((self.ntemp)) + ft)*cst.ha_to_ev*cst.ev_to_j*cst.Na
        for t,T in enumerate(self.temperature):
            #print('T = {:>3d}K : F_0+F_T = {: 13.11e} J/molc'.format(T,x[t]))
            print('T = {:>3d}K : F_0+F_T = {: 13.11e} J/molc = {: 13.11e} Ha'.format(T,x[t],f0+ft[t]))

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

            c11 = 377
            c12=91 
            c13=58 
            c33=405

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
                print(self.distinc_acell)
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
            self.ha2molc(F_0,F_T)

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

            #print(self.free_energy)
            #print(self.volume)
            for t, T in enumerate(self.temperature):
                afit = np.polyfit(self.volume[:,1],self.free_energy[:,t],2)
                fit[t] = -afit[1]/(2*afit[0])

                if plot:
                    xfit = np.linspace(9.50,12.0,100)
                    yfit = afit[0]*xfit**2 + afit[1]*xfit + afit[2]
                    #print(self.volume)
                    plt.plot(self.volume[:,1],self.free_energy[:,t],marker='o')
                    plt.plot(xfit,yfit)

            if plot:
                plt.show()

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

#    check_anaddb = False

    def __init__(self,

        rootname,
        units,
        symmetry,

        ddb_flists = None,
        out_flists = None,
        elastic_fname = None,

        wtq = [1.0],
        temperature = np.arange(0,300,50),

#        check_anaddb = False,

        bulk_modulus = None,
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
#        self.check_anaddb = check_anaddb

        self.temperature = temperature
        self.ntemp = len(self.temperature) 

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
                self.omega[v,i,:] = ddb.omega
                # get F0 contribution
                F_0 += self.wtq[i]*self.get_f0(ddb.omega) 
                # get Ftherm contribution
                F_T += self.wtq[i]*self.get_fthermal(ddb.omega,nmode)

        # Add a check for the right volumes ordering?
        # cubic : aminus equil aplus
        #hexagonal : aminus equil aplus cminus equil cplus
                
            self.free_energy[v,:] = (E+F_0)*np.ones((self.ntemp)) + F_T

#        if self.check_anaddb:
#            # Convert results in J/mol-cell, to compare with anaddb output
#            self.ha2molc(F_0,F_T)

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
        self.temperature_dependent_acell = self.minimize_free_energy()
        self.equilibrium_volume = self.volume[1,:]

        # Read elastic compliance from file
        if self.elastic_fname:
            elastic = ElasticFile(self.elastic_fname)
            self.compliance = elastic.compliance_relaxed
#            self.compliance = elastic.compliance_clamped
    #            print(self.compliance[0,0],self.compliance[0,1],self.compliance[0,2],self.compliance[2,2])

            bmod = self.get_bulkmodulus_from_elastic(elastic.stiffness_relaxed)
            print(bmod)

        self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
        self.acell_via_gruneisen = self.get_acell(nqpt,nmode)
        self.effective_phonon_pressure = self.get_phonon_effective_pressure(nqpt,nmode)
        
# add a function to get the grüneisen mode parameters. This will require to store the frequencies for computation after all volumes have been read and analysed.
# for the Gruneisen, I need the derivative od the frequencies vs volume.

    def minimize_free_energy(self):

        plot = False

        if plot:
            import matplotlib.pyplot as plt
        
        if self.symmetry == 'cubic':
            
            fit = np.zeros((self.ntemp))

            #print(self.free_energy)
            #print(self.volume)
            for t, T in enumerate(self.temperature):
                afit = np.polyfit(self.volume[:,1],self.free_energy[:,t],2)
                fit[t] = -afit[1]/(2*afit[0])

                if plot:
                    xfit = np.linspace(9.50,12.0,100)
                    yfit = afit[0]*xfit**2 + afit[1]*xfit + afit[2]
                    #print(self.volume)
                    plt.plot(self.volume[:,1],self.free_energy[:,t],marker='o')
                    plt.plot(xfit,yfit)

            if plot:
                plt.show()

            return fit

        if self.symmetry == 'hexagonal':

            return None

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
                    gru[q,v] = -self.equilibrium_volume[1]/self.omega[1,q,v]*np.polyfit(self.volume[:,1],self.omega[:,q,v],1)[0]
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


            for q,v in itt.product(range(nqpt),range(nmode)):

                if q==0 and v<3:
                # put Gruneisen at zero for acoustic modes at Gamma
                    gru[:,q,v] = 0
                else:
#                    gru[q,v] = -1*np.polyfit(np.log(self.volume[:,1]), np.log(self.omega[:,q,v]),1)[0]
                    # This is the LINEAR gruneisen parameters
                    gru[0,q,v] = -self.equilibrium_volume[1]/self.omega[1,q,v]*np.polyfit(self.volume[:3,1],self.omega[:3,q,v],1)[0]
                    gru[1,q,v] = -self.equilibrium_volume[3]/self.omega[1,q,v]*np.polyfit(self.volume[3:,3],self.omega[3:,q,v],1)[0]

                    # This is the VOLUMIC one (that is, gru(linear)/3)
                    self.gru_vol[q,v] = (gru[0,q,v] + gru[1,q,v])/3.


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
                    arr[0][0].set_ylabel('Mode Gruneisen, Gamma_a')

                    arr[0][1].plot(self.omega[1,:,v]*cst.ha_to_ev*1000,gru[1,:,v],color=col[v],marker = 'o',linestyle='None')
                    arr[0][1].set_xlabel('Frequency (meV)')
                    arr[0][1].set_ylabel('Mode Gruneisen_gamma_c')
#                    arr[0][0].set_title(r'Slope $\omega$ vs V') 
#                    arr[0][1].set_title(r'Dynamical matrix') 
#                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
#                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')
                    arr[0][0].grid(b=True, which='major')
                    arr[0][1].grid(b=True, which='major')


#            if plot:
#                for c in range(nqpt): 
#                    for i in range(nmode/2):
#                        plt.plot(np.log(self.volume[:,1]),np.log(self.omega[:,c,i]),marker='x')
#                        plt.xlabel('ln V')
#                        plt.ylabel('ln omega')

#                plt.savefig('gruneisen_GaAs2.png')
                plt.show()

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

#            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t

#            for t in range(self.ntemp):
#                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
#            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
           # bose = self.get_bose() 
            #bose = 1./(np.exp(x)-1)
            #bose[0,:3,:] = 0 # Check what Gabriel did)
            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen)/(9*self.equilibrium_volume[0]*self.bulk_modulus)
            alphavol=np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruvol)/(self.equilibrium_volume[0]*self.bulk_modulus)
            alpha2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2)/(self.equilibrium_volume[0]*self.bulk_modulus)

            # Then, get a(T)
            integral = 1./(9*self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen)
            a = self.equilibrium_volume[1]*(integral+1)

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
                arr[1].plot(self.temperature, self.temperature_dependent_acell*cst.bohr_to_ang,'g',marker='o',label='min F(V,T)')               

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
                arr[1].plot(tt,xph_volh,'m',marker='o',label='Peff, n_qv+1/2')
                arr[1].plot(tt,xph_vol,'y',marker='o',label='Peff,n_qv')
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

            return a

        if self.symmetry == 'hexagonal':
            
            
            plot = True

           

            # First, get alpha(T)

            # Get Bose-Einstein factor and specific heat Cv
            self.bose = np.zeros((nqpt,nmode, self.ntemp))
            cv = np.zeros((nqpt,nmode,self.ntemp))

            for i,n in itt.product(range(nqpt),range(nmode)):
                self.bose[i,n,:] = self.get_bose(self.omega[1,i,n],self.temperature)
                cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)

#            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t

#            for t in range(self.ntemp):
#                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
#            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
           # bose = self.get_bose() 
            #bose = 1./(np.exp(x)-1)
            #bose[0,:3,:] = 0 # Check what Gabriel did)
            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            # Get alpha_a,c with compliance 
            alpha_a = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]
            alpha_c = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]

            alpha_a2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:])/(self.equilibrium_volume[0]*self.bulk_modulus)
            alpha_c2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:])/(self.equilibrium_volume[0]*self.bulk_modulus)



            #print(np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]))
            #print(np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))
            # Then, get a(T) and c(T)
            integral_a = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[0,:,:])
            integral_c = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[1,:,:])

            a = (self.compliance[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a = self.equilibrium_volume[1]*(a/self.equilibrium_volume[0] + 1)

            c = 2*self.compliance[0,2]*integral_a + self.compliance[2,2]*integral_c
            c = self.equilibrium_volume[3]*(c/self.equilibrium_volume[0] + 1)

            acell = np.array([a,c])

            if plot:
                import matplotlib.pyplot as plt
                fig,arr = plt.subplots(1,2,figsize=(15,5),sharey=False)
                arr[0].plot(self.temperature,alpha_a*1E6,'r',label='a') 
                arr[0].plot(self.temperature,alpha_a2*1E6,'r:')
                twin0 = arr[0].twinx()
                twin0.plot(self.temperature,alpha_c*1E6,'b',label='c') 
                arr[0].set_ylabel(r'$\alpha_a$ ($10^{-6}$ K$^{-1}$)')
                twin0.set_ylabel(r'$\alpha_c$ ($10^{-6}$ K$^{-1}$)',color='b')
                twin0.plot(self.temperature,alpha_c2*1E6,'b:')

                arr[0].set_xlabel(r'Temperature (K)')
#                arr[0].legend()
                arr[1].set_xlabel(r'Temperature (K)')
                arr[0].set_title(r'Expansion coefficients') 
                arr[1].set_title(r'Lattice parameters')
                arr[1].plot(self.temperature, a*cst.bohr_to_ang,'r')
                twin1 = arr[1].twinx()
                twin1.plot(self.temperature,c*cst.bohr_to_ang,'b')
                #arr[2].plot(self.temperature-273, a*cst.bohr_to_ang, 'or')
                arr[1].set_ylabel(r'a (ang)')
                twin1.set_ylabel(r'c (ang)',color='b')
                #arr[2].set_xlabel(r'T (Celcius)')
                #arr[2].set_xlim((-100,250))
        
#                xexp = np.array([26.0,48.86,74.87,81.22,86.93,86.29,88.83,95.18,102.79,109.77,117.39,142.13,201.78,213.20])
#                yexp = np.array([5.6498,5.6511,5.6525,5.6528,5.6525,5.6531,5.6532,5.6534,5.6538,5.6542,5.6546,5.6560,5.6580,5.6598])
#                arr[2].plot(xexp,yexp,'bx')
                aax = np.array([191.834,266.905,378.641,474.619,547.897,561.844,650.808,711.909,750.280])
                aay = np.array([2.52395,3.10329,3.98113,4.52480,4.87530,4.89260,5.24276,5.69917,5.82156])
                arr[0].plot(aax,aay,'or')
                arr[0].set_ylim(0,14)
                acx = np.array([190.311,268.754,376.849,467.552,474.509,535.571,549.499,629.7,643.704,706.430,744.827])
                acy = np.array([3.55177,3.60250,3.75748,4.08767,4.01759,4.29598,4.22574,4.34637,4.62579,4.537,4.78092])
                twin0.plot(acx,acy,'ob')

                t = self.temperature
                afit = 3.184 + 0.739E-5*t + 5.92E-9*t**2
                cfit = 5.1812+1.455E-5*t+4.62E-9*t**2

                arr[1].plot(t,afit,'r:')
                twin1.plot(t,cfit,'b:')


#                plt.savefig('alpha_GaAs.png')
                plt.show() 
            
            for t,T in enumerate(self.temperature):
                print('T={}K, a={} ang, c={} ang'.format(T,a[t]*cst.bohr_to_ang,c[t]*cst.bohr_to_ang))



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
            pph_a = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],self.bose,self.gruneisen[0,:,:]/3.)/self.volume[1,0]
            pph_aw = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[0,:,:]/3.)/self.volume[1,0]

            pph_a2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen[0,:,:]/3.)/self.volume[1,0]

            pph_c = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[1,:,:]/3.)/self.volume[1,0]
            pph_cw = -1*np.einsum('q,qv,qvt,qv->t',self.wtq,self.omega[1,:,:],self.bose,self.gruneisen[1,:,:]/3.)/self.volume[1,0]
            pph_c2 = -1*np.einsum('qv,qvt,qv->t',self.omega[1,:,:],boseplushalf,self.gruneisen[1,:,:]/3.)/self.volume[1,0]




            for t,T in enumerate(self.temperature):
                print('\nT={}K, Pphonon_a = {} GPa'.format(T,pph_a[t]*cst.habo3_to_gpa))
                print('weighted: {} GPa'.format(T,pph_aw[t]*cst.habo3_to_gpa))

#                print('T={}K, Pphonon_a (+1/2) = {} GPa'.format(T,pph_a2[t]*cst.habo3_to_gpa))
                print('T={}K, Pphonon_c = {} GPa'.format(T,pph_c[t]*cst.habo3_to_gpa))
                print('weighted: {} GPa'.format(T,pph_cw[t]*cst.habo3_to_gpa))

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
#    calc.write_acell()
       # write equilibrium acells, in ascii file
    return
    


###########################


