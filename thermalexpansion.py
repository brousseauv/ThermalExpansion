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
import eos as eos

from matplotlib import rc
#rc('text', usetex = True)
#rc('font', family = 'serif', weight = 'bold')

###################################

    
tolx = 1E-16

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

    def get_Zfactor(self,omega):
        
        z = np.zeros(self.ntemp)

        for t, T in enumerate(self.temperature):
            
            # Prevent dividing by 0
            if T<1E-20 :
                continue

            x = omega/(cst.kb_haK*T)
            # prevent divergence of the log 
            if x < cst.tolx:
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
                arr.append[i+1]

        return arr 
    
 
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
        self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
        self.acell_via_gruneisen = self.get_acell(nqpt,nmode)
        print(self.volume[:,0])
        
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

    def get_gruneisen(self, nqpt, nmode,nvol):

        plot = True 

        if plot :
            import matplotlib.pyplot as plt
            fig,arr = plt.subplots(1,2,figsize = (12,6), sharey = False,squeeze=False)

        if self.symmetry == 'cubic':
            
            gru = np.zeros((nqpt,nmode))

            for q,v in itt.product(range(nqpt),range(nmode)):

                if q==0 and v<3:
                # put Gruneisen at zero for acoustic modes at Gamma
                    gru[q,v] = 0
                else:
                    gru[q,v] = -1*np.polyfit(np.log(self.volume[:,1]), np.log(self.omega[:,q,v]),1)[0]
               
            # correct divergence at q-->0
            # this would extrapolate Gruneisen at q=0 from neighboring qpts
            #x = [np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:])]
            #for v in range(3):
            #    y = [gru[1,v],gru[2,v]]
            #    gru[0,v] = np.polyfit(x,y,1)[1]
            
            gru2 = self.gruneisen_from_dynmat(nqpt,nmode,nvol)
            self.gru2 = gru2

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
                    arr[0][0].set_title(r'Slope ln$\omega$ vs lnV') 
                    arr[0][1].set_title(r'Dynamical matrix') 
                    arr[0][0].plot(self.omega[1,0,v]*cst.ha_to_ev*1000,gru[0,v],marker='d',color='black',linestyle='None')
                    arr[0][0].plot(self.omega[1,16,v]*cst.ha_to_ev*1000,gru[16,v],marker='s',color='black',linestyle='None')


#            if plot:
#                for c in range(nqpt): 
#                    for i in range(nmode/2):
#                        plt.plot(np.log(self.volume[:,1]),np.log(self.omega[:,c,i]),marker='x')
#                        plt.xlabel('ln V')
#                        plt.ylabel('ln omega')

                plt.savefig('gruneisen_GaAs2.png')
                plt.show()

            return gru 
            
    def get_acell(self, nqpt, nmode):

        # Evaluate acell(T) from Gruneisen parameters
        if self.symmetry == 'cubic':
            
            plot = False
            # First, get alpha(T)
            x = np.zeros((nqpt,nmode,self.ntemp)) # q,v,t
            for t in range(self.ntemp):
                x[:,:,t] = self.omega[1,:,:]/(cst.kb_haK*self.temperature[t])
            cv = cst.kb_haK*x**2*np.exp(x)/(np.exp(x)-1)**2
            bose = 1./(np.exp(x)-1)
            bose[0,:3,:] = 0 # Check what Gabriel did)
            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],bose)
            # fix this properly later!!! 
            cv[0,:3,:] = 0

            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen)/(self.volume[1,0]*self.bulk_modulus)
            alpha2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2)/(self.volume[1,0]*self.bulk_modulus)

            # Then, get a(T)
            integral = 1./(self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen)
            a = self.volume[1,1]*(integral + 1)

            if plot:
                import matplotlib.pyplot as plt
                fig,arr = plt.subplots(1,2,figsize=(15,5),sharey=False)
                arr[0].plot(self.temperature,alpha*1E6) 
                arr[1].plot(self.temperature,alpha2*1E6) 
                arr[0].set_ylabel(r'$\alpha$ ($10^{-6}$ K$^{-1}$)')
                arr[0].set_xlabel(r'Temperature (K)')
                arr[1].set_xlabel(r'Temperature (K)')
                arr[0].set_title(r'Slope ln$\omega$ vs lnV') 
                arr[1].set_title(r'Dynamical matrix') 


                plt.savefig('alpha_GaAs.png')
                plt.show() 
            
            
            return a
    def gruneisen_from_dynmat(self,nqpt,nmode,nvol):

        # This computes the gruneisen parameters from the change in the dynamical matrix
        # like phonopy and anaddb
            
        # for now, I reopen all files, but later on change the loop ordering (anyway, it should not do everything linearly, but rather use functions depending of input parameters

        gru = np.zeros((nqpt,nmode))
        dplus =  2
        d0 = 1
        dminus = 0 # put this in a function
        dV = self.volume[dplus,1] - self.volume[dminus,1]
        V0 = self.volume[d0,1]

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
                    eigval, eigvect = np.linalg.eigh(dynmat)

                    for ii in np.arange(ddb.natom):
                        for dir1 in np.arange(3):
                            ipert = ii*3 + dir1
                            eigvect[ipert] = eigvect[ipert]*np.sqrt(cst.me_amu/amu[ii])
                    
                    if is_gamma:
                        eigval[0] = 0.0
                        eigval[1] = 0.0
                        eigval[2] = 0.0

                    for ieig,eig in enumerate(eigval):
                        if eig < 0.0:
                            warnings.warn('Negative eigenvalue changed to 0')
                            eigval[ieig] = 0.0
            #if i==1:
            #    print(dD)
            dD_at_q = []
            
            for eig in eigvect:

                dD_at_q.append(np.vdot(np.transpose(eig), np.dot(dD,eig)).real)   
#                if i==1:
                #    print(eig)
               #     print(np.dot(dD,eig))
                    #print(dD_at_q)
                    

            dD_at_q = np.array(dD_at_q)

            for v in range(nmode):
                if i==0 and v<3:
                    gru[i,v] = 0
                else:
                    gru[i,v] = -V0*dD_at_q[v]/(2*eigval[v].real*dV)
            if i==1:

                print('Dynamical matrix')
                print('delta omega^2 {}'.format(dD_at_q))
                print('omega0 {}'.format(eigval[:]))
                print('gruneisen {}'.format(gru[i,:]))


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
            raise Exception('{} contains {} gap values, while there are {} files in etotal_flist. '.format(self.gap_fname, self.nfile, len(self.gap_energy)))

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


