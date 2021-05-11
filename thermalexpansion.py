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
import constants as cst
''' FIX ME do I still need these two? '''
from scipy.optimize import least_squares,curve_fit, leastsq


from ElectronPhononCoupling import DdbFile
from outfile import OutFile
from gsrfile import GsrFile
from gapfile import GapFile
from elasticfile import ElasticFile
from expfile import EXPfile
import eos as eos
import lmfit as lmfit

from matplotlib import rc
rc('text', usetex = True)
rc('font', family = 'sans-serif', weight = 'bold')

###################################

tolx = 1E-16
tol12 = 1E-12
tol6 = 1E-6
tol20 = 1E-20

class FreeEnergy(object):

    def __init__(self,

        rootname = 'te.out',
        units = 'eV',
        verbose = False,
        
        **kwargs):
        
       self.rootname = rootname
       self.units = units
       self.verbose = verbose

       self.eos_list = ['Murnaghan', 'Birch-Murnaghan']

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

        #    if omega<tol6:
        #        continue

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

        if self.verbose:
            for t,T in enumerate(self.temperature):

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

            if omega<tol6:
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

            if omega<tol6:
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

class Gibbs_from_anaddb(FreeEnergy):

    #Input files
    thermo_flist = None
    out_flists = None

    #Parameters
    wtq = [1.0]
    temperature = None

    equilibrium_index = None

    def __init__(self,

        rootname,
        units,
        symmetry,

        thermo_flist = None,
        out_flists = None,

        wtq = [1.0],
        temperature = np.arange(0,300,50),

        bulk_modulus = None,
        bulk_modulus_units = None,
        pressure = 0.0,
        pressure_units = None,
        eos_type = 'Murnaghan',
        use_axial_eos = False,

        equilibrium_index = None,

        **kwargs):


        print('Reading Gibbs free energy from anaddb output')
        if not thermo_flist:
            raise Exception('Must provide a list of files for thermo_flist')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != len(thermo_flist):
            raise Exception('thermo_flist and out_flists must have the same number of volumes,but thermo_flist has {} and out_flist has {}'.format(len(thermo_flist),len(out_flists)))


        #Set input files
        self.thermo_flist = thermo_flist
        self.out_flists = out_flists
        self.symmetry = symmetry
        if not self.symmetry:
            raise Exception('Symmetry type must be specified')

        super(Gibbs_from_anaddb,self).__init__(rootname,units)

        self.equilibrium_index = equilibrium_index  #Transfer to Python indexing
        if self.equilibrium_index is None:
            raise Exception('Must provide the equilibrium volume index in volume list (in normal indexing)')
        print('Equilibrium data is the {} volume in list'.format(self.equilibrium_index))
        self.equilibrium_index -= 1


        if bulk_modulus is not None:
            if bulk_modulus_units == 'GPa':
                self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
            elif bulk_modulus_units == 'HaBo3':
                self.bulk_modulus = bulk_modulus
            else:
                raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')

        self.pressure_units = pressure_units
        self.pressure = pressure

        if self.pressure_units == 'None':
            raise Exception('Please specify pressure units')

        if self.pressure_units == 'GPa':
            self.pressure_gpa = self.pressure
            self.pressure = self.pressure*cst.gpa_to_habo3
        else:
            self.pressure_gpa = self.pressure*cst.habo3_to_gpa

        print('Computing at external pressure {} GPa'.format(self.pressure_gpa))

        #Define EOS type
        self.eos_type = eos_type

        if self.eos_type not in self.eos_list:
            raise Exception('EOS type must be one of the following: {}'.format(self.eos_list))
        self.use_axial_eos = use_axial_eos

        # set parameter space dimensions
        nvol = len(thermo_flist)

        # Store the reduced coordinates of qpoints
        self.qred = np.zeros((nqpt,3))

        self.volume = np.empty((nvol,4)) # [dataset, [total cell volume, a1,a2,a3]]

        # Read data from _THERMO files
        for v in range(nvol):
            fname = self.thermo_flist[v]
            temp = np.loadtxt(fname,usecols=(0))
            if v==0:
                self.temperature = temp
                self.ntemp = len(self.temperature)
                self.free_energy = np.zeros((nvol,self.ntemp))
            else:
                if not np.allclose(self.temperature, temp):
                    raise Exception('All temperature arrays should be equal, but there was a discrepancy in file {}'.format(fname))

            F_thermal = np.loadtxt(fname,usecols=(1))

            # Read data from _OUT file
            gs = OutFile(out_flists[v])
            self.volume[v,0] = gs.volume
            self.volume[v,1:] = gs.acell

            if v==0:
                self.distinct_acell = self.reduce_acell(self.volume[v,1:])


            if v == self.equilibrium_index:
                self.equilibrium_volume = self.volume[v,:]
                if self.verbose:
                    print('Equilibrium volume (static): {} ha/bohr^3'.format(self.equilibrium_volume))

            # get total energy
            E = gs.etotal[0]

            self.free_energy[v,:] = E*np.ones((self.ntemp)) + F_thermal/(cst.ha_to_ev*cst.ev_to_j*cst.avogadro) + self.pressure*self.volume[v,0]*np.ones((self.ntemp))

        # Minimize Ftotal

        if self.verbose:
            for t,T in enumerate(self.temperature):
                print('for T = {}K, free energy:'.format(T))
                print('minimal value for dataset {}'.format(np.argmin(self.free_energy[:,t])))
                sort = self.free_energy[:,t].argsort()
                print('sorted order: {}'.format(sort))
                print('delta min:{}'.format(self.free_energy[sort[1],t]-self.free_energy[sort[0],t]))
                
        self.temperature_dependent_acell = self.minimize_free_energy()

    def minimize_free_energy(self):

        if self.symmetry == 'cubic':

            fit = np.zeros((self.ntemp))

            if self.eos_type == 'Birch-Murnaghan':
                myeos = eos.birch_murnaghan_EV
            if self.eos_type == 'Murnaghan':
                myeos = eos.murnaghan_EV

            for t, T in enumerate(self.temperature):
                p0 = [self.equilibrium_volume[0],self.free_energy[1,t],self.bulk_modulus,4.0]
                popt, pcov = curve_fit(myeos, self.volume[:,0],self.free_energy[:,t],p0)
                fit[t] = (4*popt[0])**(1./3)

            fit = np.expand_dims(fit,axis=0)

            return fit


        if self.symmetry == 'hexagonal':

            # Only independent fit for now... take the paraboloid fit from GibbsFreeEnergy (put this in FreeEnergy global class?)
            fit = np.zeros((2,self.ntemp))

            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig,arr = plt.subplots(2,3,sharey=False,figsize=(10,10))
            ax = arr[1,2].twinx()
            mycmap = cm.jet
            color_idx = np.linspace(0, 1, self.ntemp)
            print(self.equilibrium_volume)

            from scipy.optimize import leastsq

#            fit = np.zeros((2,self.ntemp))
#            fit2d = np.zeros((2,self.ntemp))
            fitg = np.zeros((2,self.ntemp))
            fit2dg = np.zeros((2,self.ntemp))
            fit2d_cf = np.zeros((2,self.ntemp))
            fit2d_par = np.zeros((2,self.ntemp))
            fitind_par = np.zeros((2,self.ntemp))
            fit2d_quad = np.zeros((2,self.ntemp))


#            if plot:
#                import matplotlib.pyplot as plt
#                from mpl_toolkits.mplot3d import Axes3D

            # This delta is the difference between real HGH minimum and PAW minimum, at the current pressure
            if self.pressure_gpa==0.0:
                delta =  [0.011000356489930141,0.0844492602578999] #for 0gpa
            if self.pressure_gpa == 0.5:
                delta = [0.004513105799999195,0.0239153519999995] #for 0.5gpa
            if self.pressure_gpa == 1.0:
                delta = [0.0,0.0] #for 1gpa
            if self.pressure_gpa == 1.5:
                delta = [0.0031372478999998066,0.001351464000000746] #for 1.5gpa
            if self.pressure_gpa == 3.0:
                delta = [0.0,0.0] #for 3gpa
            if self.pressure_gpa == 3.5:
                delta = [0.001538456200000482,-0.0026159529999993936] #for 3.5gpa
            if self.pressure_gpa == 5.0:
                delta = [0.016926552700000208,-0.0055531139999995816] #for 5gpa


            for t,T in enumerate(self.temperature):


                # define bounds for optimal parameters
                bounds_axial = [[0.97*self.equilibrium_volume[1],0.97*self.equilibrium_volume[3],1.03*self.free_energy[self.equilibrium_index,t],0.,0.],
                                [1.03*self.equilibrium_volume[1],1.03*self.equilibrium_volume[3],0.97*self.free_energy[self.equilibrium_index,t],100.,30.]]
                bounds_vol = [[0.97*self.equilibrium_volume[0],1.03*self.free_energy[self.equilibrium_index,t],0.,0.],
                                [1.03*self.equilibrium_volume[0],0.97*self.free_energy[self.equilibrium_index,t],100.,30.]]
                va = np.sqrt(3)/2*self.equilibrium_volume[3]
                vc = np.sqrt(3)/2*(self.equilibrium_volume[1]**2)
                bounds_a = [[va*(0.97*self.equilibrium_volume[1])**2,1.03*self.free_energy[self.equilibrium_index,t],0.,0.],
                                [va*(1.03*self.equilibrium_volume[1])**2,0.97*self.free_energy[self.equilibrium_index,t],100.,30.]]
                bounds_c = [[0.97*vc*self.equilibrium_volume[3],1.03*self.free_energy[self.equilibrium_index,t],0.,0.],
                                [1.03*vc*self.equilibrium_volume[3],0.97*self.free_energy[self.equilibrium_index,t],100.,30.]]
                bounds_para = [[0.97*self.equilibrium_volume[1],0.,0.97*self.equilibrium_volume[3],0.,1.03*self.free_energy[self.equilibrium_index,t]],
                                [1.03*self.equilibrium_volume[1],np.inf,1.03*self.equilibrium_volume[3],np.inf,0.97*self.free_energy[self.equilibrium_index,t]]]

                # First, treat a
                # a0,c0, E0, B0, B0' 
                '''or, B0=9.8GPa, B0'=7.6, B0prime has no units!!!'''
                p0 = [self.equilibrium_volume[1],self.equilibrium_volume[3], self.free_energy[self.equilibrium_index,t],10.0*cst.gpa_to_habo3,8.0]
                popt, pcov= curve_fit(eos.murnaghan_EV_axial, self.volume[:,0], self.free_energy[:,t], p0)
                print('\nfor T={}K, a= {} c={}'.format(T,popt[0],popt[1]))

                delta = [0.0031372478999998066,0.001351464000000746]
                # Fit only the a variation
                p0a = [self.equilibrium_volume[0], self.free_energy[self.equilibrium_index,t], 10.0*cst.gpa_to_habo3,8.0]
                popta, pcova= curve_fit(eos.murnaghan_EV, self.volume[:7,0], self.free_energy[:7,t], p0a,bounds=bounds_a)
#                popta, pcova= curve_fit(eos.murnaghan_EV, self.volume[:10,0], self.free_energy[:10,t], p0a)
                aa = np.sqrt(2*popta[0]/(np.sqrt(3)*self.equilibrium_volume[3]))
                print('a data only:, v={}, a={}'.format(popta[0],aa-delta[0]))
                print('uncertainty: {}'.format(2*np.sqrt(pcova[0,0])/np.sqrt(3)*(self.equilibrium_volume[3])))


                # Fit volume, whithout splitting into a0 and c0
                p0v = [self.equilibrium_volume[0], self.free_energy[self.equilibrium_index,t], 10.0*cst.gpa_to_habo3,8.0]
                poptv, pcovv= curve_fit(eos.murnaghan_EV, self.volume[:,0], self.free_energy[:,t], p0v,bounds=bounds_vol)
                print('volume fit = {}'.format(poptv[0]))
                print('K0={},k0p={}'.format(poptv[2]/cst.gpa_to_habo3,poptv[3]))


                # Fit only the c variation

#                varr = np.zeros((7))
#                varr[0] = self.volume[0,0]
#                varr[1:] = self.volume[7:13,0]
#                farr = np.zeros((7))
#                farr[0] = self.free_energy[0,t]
#                farr[1:] = self.free_energy[7:13,t]
                varr = np.zeros((3))
                varr[0] = self.volume[0,0]
                varr[1:] = self.volume[3:5,0]
                farr = np.zeros((3))
                farr[0] = self.free_energy[0,t]
                farr[1:] = self.free_energy[3:5,t]


                

                p0c = [self.equilibrium_volume[0], self.free_energy[self.equilibrium_index,t], 10.0*cst.gpa_to_habo3,8.0]
                poptc, pcovc= curve_fit(eos.murnaghan_EV, varr, farr, p0c,bounds=bounds_c)

                cc = 2*poptc[0]/(np.sqrt(3)*(self.equilibrium_volume[1]**2))
                print('c data only,v = {}, c={}'.format(poptc[0],cc-delta[1]))
                print('uncertainty: {}'.format(2*np.sqrt(pcovc[0,0])/(np.sqrt(3))*(self.equilibrium_volume[1]**2)))
#                print(self.free_energy[:,t])

                print('Minimizing free energy surface')
                #afit = np.polyfit(self.volume[:3,1],self.free_energy[:3,t],2)
                #fit[0,t] = -afit[1]/(2*afit[0])
                #cfit = np.polyfit(self.volume[3:,3],self.free_energy[3:,t],2)
                #fit[1,t] = -cfit[1]/(2*cfit[0])

                
                #fit2, cov2 = leastsq(self.residuals, x0=[afit[0],afit[0],cfit[0],cfit[0],self.free_energy[1,t]], args=(self.volume[:,1],self.volume[:,3],
                #    self.free_energy[:,t]),maxfev=4000)
                #fit2d[:,t] = fit2[0],fit2[2]
 #               print('\nT={}'.format(T))
                #print(fit2)
                #print(cov2)
#                print('independent fit')
#                print(fit[:,t])
#                print('2d fit')
#                print(fit2d[:,t])

                # Fit Gibbs free energy
                #afitg = np.polyfit(self.volume[:3,1],self.gibbs_free_energy[:3,t],2)
                afitg = aa
                #fitg[0,t] = -afitg[1]/(2*afitg[0])
                #cfitg = np.polyfit(self.volume[3:,3],self.gibbs_free_energy[3:,t],2)
                cfitg = cc
                #fitg[1,t] = -cfitg[1]/(2*cfitg[0])

                K0,K0p = 10.0*cst.gpa_to_habo3, 8.0
#                if t==0:
                # ystart from equilibrium values
                fit2g = least_squares(self.residuals, x0=[self.equilibrium_volume[1],self.equilibrium_volume[3],self.free_energy[self.equilibrium_index,t],K0,K0p], bounds=bounds_axial,args=(self.volume[:,1],self.volume[:,3],self.free_energy[:,t]))
                fitcf,cfopt = curve_fit(eos.murnaghan_EV_axial2D,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t], p0=[afitg,cfitg,self.free_energy[self.equilibrium_index,t],K0,K0p],bounds=bounds_axial)
                # start from independent solution
#                    fit2g = least_squares(self.residuals, x0=[afitg,cfitg,self.free_energy[self.equilibrium_index,t],K0,K0p], bounds=bounds_axial,args=(self.volume[:,1],self.volume[:,3],
#                        self.free_energy[:,t]))
#                    fitcf,cfopt = curve_fit(eos.murnaghan_EV_axial2D,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t], p0=[afitg,cfitg,self.free_energy[self.equilibrium_index,t],K0,K0p],bounds=bounds_axial)

                fitpar,covpar = curve_fit(self.paraboloid,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t],
                        p0=[self.equilibrium_volume[1],self.equilibrium_volume[1],self.equilibrium_volume[3],self.equilibrium_volume[3],self.free_energy[self.equilibrium_index,t]],
                        bounds=bounds_para)

#                fitquad,covquad = curve_fit(self.quadratic2D,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t],p0=[1,1,1,1,1,self.free_energy[self.equilibrium_index,t]])
#                fit2d_quad[0,t] = (2*fitquad[3]*fitquad[4] - fitquad[2]*fitquad[3])/(fitquad[2]**2 - 4*fitquad[0]*fitquad[4])
#                fit2d_quad[1,t] = (-4*fitquad[0]*fitquad[4]*fitquad[3] + 2*fitquad[0]*fitquad[2]*fitquad[3])/(fitquad[2]*(fitquad[2]**2-4*fitquad[0]*fitquad[4]))-fitquad[1]/fitquad[2]


#                else:
#                    fit2g = least_squares(self.residuals, x0=[fit2dg[0,t-1],fit2dg[1,t-1],self.free_energy[self.equilibrium_index,t],K0,K0p], bounds=bounds_axial,args=(self.volume[:,1],self.volume[:,3],self.free_energy[:,t]))
#                    fitcf,cfopt = curve_fit(eos.murnaghan_EV_axial2D,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t], p0=[fitcf[0],fitcf[1],self.free_energy[self.equilibrium_index,t],K0,K0p],bounds=bounds_axial)
#                    fitpar,covpar = curve_fit(self.paraboloid,[self.volume[:,1],self.volume[:,3]], self.free_energy[:,t],
#                            p0=[fitpar[0],fitpar[1],fitpar[2],fitpar[3],self.free_energy[self.equilibrium_index,t]],
#                            bounds=bounds_para)

                # quadratic fit for a and c
                myarrv = np.zeros((5))
                myarrv[0] = self.volume[0,3]
                myarrv[1:] = self.volume[5:9,3]
                myarrf = np.zeros((5))
                myarrf[0] = self.free_energy[0,t]
                myarrf[1:] = self.free_energy[5:9,t]

                if t==0:
                    fitaq, fitacov= curve_fit(self.quadratic, self.volume[:5,1],self.free_energy[:5,t])
                    fitcq, fitccov= curve_fit(self.quadratic, myarrv,myarrf)
                else:
                    fitaq, fitacov= curve_fit(self.quadratic, self.volume[:5,1],self.free_energy[:5,t],p0=fitaq)
                    fitcq, fitccov= curve_fit(self.quadratic,myarrv,myarrf,p0=fitcq)

                fitind_par[:,t] = -fitaq[1]/(2*fitaq[0]), -fitcq[1]/(2*fitcq[0])
                
                # print different fit results and uncertainty
                print('with least_squares')
                print('a={}, c={},K0={},K0p={}'.format(fit2g.x[0],fit2g.x[1],fit2g.x[3]*cst.habo3_to_gpa,fit2g.x[4]))
                print('cost function: {}'.format(fit2g.cost))
                fit2dg[:,t] = fit2g.x[:2]
#                print('Gibbs')
            #    print(fitg[:,t]-delta)
            #    print(fit2dg[:,t]-delta)
                #fitg[:,t] = fitg[:,t]-delta
                fit2dg[:,t] = fit2dg[:,t]-delta
                print('with curve_fit')
#                print(cfopt)
                print('a={}, c={},K0={},K0p={}'.format(fitcf[0],fitcf[1],fitcf[3]*cst.habo3_to_gpa,fitcf[4]))
                fit2d_cf[:,t] = fitcf[0],fitcf[1]

                print('Uncertainties:')
                for i in range(len(fitcf)):
                    print('{}'.format(np.sqrt(cfopt[i, i])))

                print('with paraboloid')
                print('a={}, c={}'.format(fitpar[0],fitpar[2]))
                print('Uncertainties: {} {}'.format(np.sqrt(covpar[0,0]),np.sqrt(covpar[2,2])))
                fit2d_par[:,t] = fitpar[0],fitpar[2]
                print(covpar)
#                print(self.gibbs_free_energy[:,t])
                
                print('independent fit: a={},c={}'.format(fitind_par[0,t],fitind_par[1,t]))
                print(fitacov)
                print(fitccov)

#                print('full 2D quadratic: a={}, c={}'.format(fit2d_quad[0,t],fit2d_quad[1,t]))
#                print(covquad)
############### stopped copying here

#                if plot:
#                    fig = plt.figure()
#                    arr = fig.add_subplot(111,projection='3d')
#                    arr.plot(self.volume[:3,1],self.volume[:3,3],self.gibbs_free_energy[:3,t],marker='o',color='k',linestyle='None') #at T=0
#                    arr.plot(self.volume[3:,1],self.volume[3:,3],self.gibbs_free_energy[3:,t],marker='o',color='b',linestyle='None') #at T=0
#
#                    xmesh = np.linspace(0.99*self.volume[0,1],1.01*self.volume[2,1],200)
#                    ymesh = np.linspace(0.99*self.volume[3,3],1.01*self.volume[5,3],200)
#                    xmesh,ymesh = np.meshgrid(xmesh,ymesh)
#                    zmesh = self.paraboloid(xmesh,ymesh,p0=fit2g)
#                    zlim = arr.get_zlim3d()
#                    arr.plot_wireframe(xmesh,ymesh,zmesh)
#                    xx = np.ones((10))
#                    arr.plot(fit2dg[0,t]*xx,fit2dg[1,t]*xx,np.linspace(0.99999*zlim[0],1.00001*zlim[1],10),color='magenta',linewidth=2,zorder=3)
#
#                    out='FIG/{}_{}K.png'.format(self.rootname,T)
#                    create_directory(out)
#                    plt.savefig(out)
#                    plt.show()
#                    plt.close()
#
#            self.independent_fit = fit
#            self.fit2d = fit2d 
#            self.fitg = fitg
#            self.fit2dg = fit2dg


#                # 2D fit with lmfit
#                # First, create my 2D mesh:
#                ac_mesh = np.meshgrid(self.volume[:,1],self.volume[:,3])
# #               print(ac_mesh)
#                free_energy2D = self.free_energy[:,t]
##                print(np.shape(free_energy2D))
##                lmfit_model = lmfit.Model(eos.murnaghan_EV_axial2D)
#                lmfit_model = lmfit.Model(eos.birch_murnaghan_EV_axial2D)
#
#                params = lmfit_model.make_params()
#                params['a0'].set(value=self.equilibrium_volume[1], vary=True,min=0.95*self.equilibrium_volume[1],max=1.05*self.equilibrium_volume[1])
#                params['c0'].set(value=self.equilibrium_volume[3], vary=True,min=0.95*self.equilibrium_volume[3],max=1.05*self.equilibrium_volume[3])
#                params['E0'].set(value=free_energy2D[self.equilibrium_index],vary=True,min=1.10*free_energy2D[self.equilibrium_index],max=0.95*free_energy2D[self.equilibrium_index])
#                params['K0'].set(value=8.7, vary=True,min=6.0,max=30.0)
#                params['K0p'].set(value=8.9, vary=True,min=2.0,max=12.0)
#
#                lmfit_result = lmfit_model.fit(free_energy2D,mesh=[self.volume[:,1],self.volume[:,3]],a0=params['a0'],c0=params['c0'],E0=params['E0'],K0=params['K0'],K0p=params['K0p'])
#                print(lmfit_result.fit_report())
#                print(lmfit_result.params.pretty_print())
##                fitlsq, covlsq = least_squares(self.residuals,x0=[aa,cc,popta[1],popta[2],popta[3]], args=(self.volume[:,1],self.volume[:,3],self.free_energy[:,t]))
##                print(fitlsq)


                #### Plotting starts here
                arr[0,0].plot(self.volume[1:7,0],self.free_energy[1:7,t],marker='D',linestyle='None',color=mycmap(color_idx[t]))
#                arr[0,0].plot(self.volume[1:10,0],self.free_energy[1:10,t],marker='D',linestyle='None',color=mycmap(color_idx[t]))
                arr[0,0].plot(self.volume[0,0],self.free_energy[0,t],marker='o',linestyle='None',color=mycmap(color_idx[t]))
                V0=np.sqrt(3)/2*aa**2*self.equilibrium_volume[3] 
                arr[0,0].plot(V0, eos.murnaghan_EV(V0,popta[0],popta[1],popta[2],popta[3]),'kx')
                arr[0,1].plot(self.volume[0,0],self.free_energy[0,t],marker='o',linestyle='None',color=mycmap(color_idx[t]),mec='black')
                arr[0,1].plot(self.volume[7:,0],self.free_energy[7:,t],marker='s',linestyle='None',color=mycmap(color_idx[t]))
                #arr[0,1].plot(self.volume[10:,0],self.free_energy[10:,t],marker='s',linestyle='None',color=mycmap(color_idx[t]))
                V0=np.sqrt(3)/2*cc*self.equilibrium_volume[1]**2
                arr[0,1].plot(V0, eos.murnaghan_EV(V0,poptc[0],poptc[1],poptc[2],poptc[3]),'kx')

                dummyx = np.linspace(0.99*np.amin(self.volume[:,0]),1.01*np.amax(self.volume[:,0]),100)
                dummyy = eos.murnaghan_EV_axial(dummyx,popt[0],popt[1],popt[2],popt[3],popt[4])
                dummyyv = eos.murnaghan_EV(dummyx,poptv[0],poptv[1],poptv[2],poptv[3])
                dummyya = eos.murnaghan_EV(dummyx,popta[0],popta[1],popta[2],popta[3])
                dummyyc = eos.murnaghan_EV(dummyx,poptc[0],poptc[1],poptc[2],poptc[3])


                arr[0,2].plot(self.volume[1:7,0],self.free_energy[1:7,t],marker='D',linestyle='None',color=mycmap(color_idx[t]))
                arr[0,2].plot(self.volume[7:,0],self.free_energy[7:,t],marker='s',linestyle='None',color=mycmap(color_idx[t]))
#                arr[0,2].plot(self.volume[1:10,0],self.free_energy[1:10,t],marker='D',linestyle='None',color=mycmap(color_idx[t]))
#                arr[0,2].plot(self.volume[10:,0],self.free_energy[10:,t],marker='s',linestyle='None',color=mycmap(color_idx[t]))
                arr[0,2].plot(self.volume[0,0],self.free_energy[0,t],marker='o',linestyle='None',color=mycmap(color_idx[t]),markeredgecolor='black')

                arr[0,0].plot(dummyx,dummyya,color=mycmap(color_idx[t]))
                arr[0,1].plot(dummyx,dummyyc,color=mycmap(color_idx[t]))
                arr[0,2].plot(dummyx,dummyy,color=mycmap(color_idx[t]))

                #arr[0,0].plot(dummyx,dummyyv,color='black',linestyle='dashed')
                #arr[0,1].plot(dummyx,dummyyv,color='black',linestyle='dashed')
                arr[0,2].plot(dummyx,dummyyv,color='black',linestyle='dotted')


                #plot acell(T)
                arr[1,0].plot(T, aa-delta[0], 'or',linestyle='None',label='indep eos')
                arr[1,1].plot(T, cc-delta[1], 'ob',linestyle='None')
                if t==0:
                    arr[1,2].plot(T, popt[0], 'or',linestyle='None',label='a')
                else: 
                    arr[1,2].plot(T, popt[0], 'or',linestyle='None')

                if t==0:
                    ax.plot(T,popt[1],'ob',linestyle='None',label='c')
                else:
                    ax.plot(T,popt[1],'ob',linestyle='None')

                arr[1,0].plot(T,fit2dg[0,t],'xr',linestyle='None',label='least_squares eos')
                arr[1,1].plot(T,fit2dg[1,t],'xb',linestyle='None')

                arr[1,0].plot(T,fit2d_cf[0,t],'sr',linestyle='None',label='curve_fit eos')
                arr[1,1].plot(T,fit2d_cf[1,t],'sb',linestyle='None')

                arr[1,0].plot(T,fit2d_par[0,t],'*r',linestyle='None',label='paraboloid')
                arr[1,1].plot(T,fit2d_par[1,t],'*b',linestyle='None')

                arr[1,0].plot(T,fitind_par[0,t],'*g',linestyle='None',label='quadr indep')
                arr[1,1].plot(T,fitind_par[1,t],'*c',linestyle='None')

#                arr[1,0].plot(T,fit2d_quad[0,t],'hm',linestyle='None',label='quadr 2d')
#                arr[1,1].plot(T,fit2d_quad[1,t],'hy',linestyle='None')


                fit[:,t] = aa-delta[0], cc-delta[1]

            arr[0,0].set_title("a variation only")
            arr[0,1].set_title("c variation only")
            arr[0,2].set_title("a,c variation")
            fig.suptitle("Murnaghan EOS")
            arr[0,0].set_ylabel("Free energy (Ha)")
            arr[0,0].set_xlabel("Volume (bohr^3)")
            arr[0,2].set_xlabel("Volume (bohr^3)")
            arr[0,1].set_xlabel("Volume (bohr^3)")
            arr[1,0].set_ylabel("a (bohr)")
            arr[1,1].set_ylabel("c (bohr)")
            arr[1,2].set_ylabel("a (bohr)")
            ax.set_ylabel("c (bohr)",color='blue')
            arr[1,0].set_xlabel("Temperature (K)")
            arr[1,1].set_xlabel("Temperature (K)")
            arr[1,2].set_xlabel("Temperature (K)")
            arr[1,2].legend(numpoints=1,loc=1)
            ax.legend(numpoints=1,loc=2)
#            arr[1,0].legend(numpoints=1)

            plt.savefig("Murnaghan_1p5gpa_from_anaddb.png")
            plt.show()

            return fit

    def residuals(self,params,x,y,z):

        V = np.sqrt(3)/2*x**2*y

        return z - eos.murnaghan_EV_axial2D([x,y],params[0],params[1],params[2],params[3],params[4])

    def paraboloid(self, mesh, a0,A,c0,C,B):

        x,y = mesh
        z = (x-a0)**2/A**2 + (y-c0)**2/C**2 + B

        return z

    def quadratic(self, x, a,b,c):

        return a*x**2+b*x+c

    def quadratic2D(self, mesh, a,b,c,d,e,f):

        x,y = mesh
        return a*x**2 + b*x + c*x*y + d*y + e*y**2 + f

    def residuals_parab(self,params,x,y,z):

        return z - self.paraboloid([x,y],params)

    def write_acell(self):

        outfile = 'OUT/{}_acell_from_eos.dat'.format(self.rootname)
        nc_outfile = 'OUT/{}_acell.nc'.format(self.rootname)

        # Write output in netCDF format
        create_directory(nc_outfile)

        with nc.Dataset(nc_outfile, 'w') as dts:

            dts.createDimension('number_of_temperatures', self.ntemp)
            dts.createDimension('number_of_lattice_parameters', len(self.distinct_acell))

            data = dts.createVariable('temperature','d', ('number_of_temperatures'))
            data[:] = self.temperature[:]
            data.units = 'Kelvin'


            data = dts.createVariable('acell_from_gibbs','d',('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.temperature_dependent_acell[:,:]
            data.units = 'Bohr radius'


        # Write ascii file
        outfile = 'OUT/{}_acell_from_eos.dat'.format(self.rootname)

        create_directory(outfile)

        with open(outfile, 'w') as f:

            f.write('Temperature dependent lattice parameters via Gibbs free energy\n\n')

            if self.symmetry == 'cubic':


                f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t]))

                f.close()


            if self.symmetry == 'hexagonal':

                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t],self.temperature_dependent_acell[1,t]))

                # Independent fit: fitg, 2D fit: fit2dg
#                f.write('\n\nTemperature dependent lattice parameters via Gibbs free energy\n\n')
#                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
#                for t,T in enumerate(self.temperature):
 #                   f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fitg[0,t],self.fitg[1,t]))



                f.close()


class GibbsFreeEnergy(FreeEnergy):

    #Input files
    ddb_flists = None
    out_flists = None

    #Parameters
    wtq = [1.0]
    temperature = None

    check_anaddb = False
    equilibrium_index = None

    def __init__(self,

        rootname,
        units,
        symmetry,

        ddb_flists = None,
        out_flists = None,

        wtq = [1.0],
        temperature = np.arange(0,300,50),
        tmin_slope = 500,

        check_anaddb = False,

        bulk_modulus = None,
        bulk_modulus_units = None,
        pressure = 0.0,
        pressure_units = None,

        equilibrium_index = None,
        manual_correction = False,

        eos_type = 'Murnaghan',
        use_axial_eos = False,

        **kwargs):


        print('Thermal expansion via Gibbs free energy')
        if not ddb_flists:
            raise Exception('Must provide a list of files for ddb_flists')
        if not out_flists:
            raise Exception('Must provide a list of files for out_flists')        

        if len(out_flists) != np.shape(ddb_flists)[0]:
            raise Exception('ddb_flists and out_flists must have the same number of volumes,but ddb_flists has {} and out_flist has {}'.format(np.shape(ddb_flists)[0],len(out_flists)))


        #Set input files
        self.ddb_flists = ddb_flists
        self.out_flists = out_flists
        self.symmetry = symmetry

        if not self.symmetry:
            raise Exception('Symmetry type must be specified')

        super(GibbsFreeEnergy,self).__init__(rootname,units)
        self.check_anaddb = check_anaddb

        self.equilibrium_index = equilibrium_index  #Transfer to Python indexing
        if self.equilibrium_index is None:
            raise Exception('Must provide the equilibrium volume index in volume list (Index starting at 1)')
        
        if self.verbose:
            print('Equilibrium data is the {} volume in list'.format(self.equilibrium_index))
        self.equilibrium_index -= 1

        self.temperature = temperature
        self.ntemp = len(self.temperature) 
        self.tmin_slope = tmin_slope

        if not bulk_modulus:
            raise Exception('Must provide an estimate of the bulk modulus and specify its units, in GPa or Ha/bohr^3')
        if bulk_modulus is not None:
            if bulk_modulus_units == 'GPa':
                self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
            elif bulk_modulus_units == 'HaBo3':
                self.bulk_modulus = bulk_modulus
            else:
                raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')
        self.pressure_units = pressure_units
        self.pressure = pressure

        if self.pressure_units == 'None':
            raise Exception('Please specify pressure units')

        if self.pressure_units == 'GPa':
            self.pressure_gpa = self.pressure
            self.pressure = self.pressure*cst.gpa_to_habo3
        else:
            self.pressure_gpa = self.pressure*cst.habo3_to_gpa

        print('\nComputing at external pressure {} GPa'.format(self.pressure_gpa))

        #Define EOS type
        self.eos_type = eos_type
        if self.eos_type not in self.eos_list:
            raise Exception('EOS type must be one of the following: {}'.format(self.eos_list))
        self.use_axial_eos = use_axial_eos


        # set parameter space dimensions
        nvol, nqpt = np.shape(self.ddb_flists)
        self.free_energy = np.zeros((nvol,self.ntemp))
        self.static_energy = np.zeros_like(self.free_energy)
        self.phonon_free_energy = np.zeros_like(self.free_energy)

        #Store reduced coordinates of qpoints
        self.qred = np.zeros((nqpt,3))

        self.volume = np.empty((nvol,4)) # 1st index = data index, 2nd index : total cell volume, (a1,a2,a3)

        # Check that all qpt lists have the same lenght, and that it is equal to the number of wtq
        for v in range(nvol):
            if len(ddb_flists[v][:]) != len(wtq):
                raise Exception('all ddb lists must have the same number of files, and this number should be equal to the number of qpt weights')

        self.set_weights(wtq)

        # Loop on all volumes
        for v in range(nvol):

            print('\n\nReading data from {}'.format(self.out_flists[v]))

           # Open OUTfile
            gs = OutFile(out_flists[v])
            self.volume[v,0] = gs.volume
            self.volume[v,1:] = gs.acell

            # check how many lattice parameters are unequivalent
            # This will need to be checked if I want a more general formulation, as for tetragonal systems,
            # the angles may change too if the cell is primitive and not cartesian...
            if v == 0:  
                self.distinct_acell = self.reduce_acell(self.volume[v,1:])
                nmode = 3*gs.natom
                self.natom = gs.natom
                self.omega = np.zeros((nvol,nqpt,nmode))

            if v == self.equilibrium_index:
                self.equilibrium_volume = self.volume[v,:]

            # get E
            E = gs.etotal[0]

            # initialize F_0, F_T 
            F_0 = 0.
            F_T = np.zeros((self.ntemp))

            # Add electronic entropy term, maybe later

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
                    print('Qpt is Gamma')
                else:
                    ddb.compute_dynmat()
                        ##### CHECK WITH GABRIEL IF I SHOULD HAVE LOTO SPLITTING AT GAMMA (WHERE DOES HIS CODE TREAT THE ELECTRIC FIELD PERTURBAITON IN THE DDB AT GAMMA???)


                # Store frequencies
                self.omega[v,i,:] = ddb.omega

                # get F0 contribution
                F_0 += self.wtq[i]*self.get_f0(ddb.omega) 
                # get Ftherm contribution
                F_T += self.wtq[i]*self.get_fthermal(ddb.omega,nmode)
                
            # Sum free energy = E + F_0 + F_T + PV
            self.free_energy[v,:] = (E+F_0)*np.ones((self.ntemp)) + F_T + self.pressure*self.volume[v,0]*np.ones((self.ntemp))
            self.static_energy[v,:] = E*np.ones((self.ntemp)) + self.pressure*self.volume[v,0]*np.ones((self.ntemp))
            self.phonon_free_energy[v,:] = F_0*np.ones((self.ntemp)) + F_T

        if self.check_anaddb:
            # Convert results in J/mol-cell, to compare with anaddb output
            self.ha2molc(F_0,F_T,v)


        # For sanity checks, the following sort the datasets by increasing free energy. 
        if self.verbose:
            for t,T in enumerate(self.temperature):
                print('for T = {}K:'.format(T))
                print('minimal free energy value has index {}'.format(np.argmin(self.free_energy[:,t])))
                sort = self.free_energy[:,t].argsort()
                print('sorted order: {}'.format(sort))
                print('delta min:{}'.format(self.free_energy[sort[1],t]-self.free_energy[sort[0],t]))


        # Minimize F, according to crystal symmetry
        # The fitting parameters will be stored in the output file, so that the fitting can be plotted afterwards.
        self.temperature_dependent_acell = self.minimize_free_energy()
        self.alpha= self.get_alpha_from_acell(self.temperature_dependent_acell)
        self.discrete_alpha= self.discrete_alpha_vs_reftemp(self.temperature_dependent_acell, 0.)
        self.discrete_room_temp_alpha= self.discrete_alpha_vs_reftemp(self.temperature_dependent_acell, 293.)
        self.room_temp_alpha= self.get_alpha_vs_reftemp(self.temperature_dependent_acell, 293.)
        self.compute_specific_heat(nqpt, nmode)

        
    def minimize_free_energy(self):

        if self.symmetry == 'cubic':

            # Define EOS type, for volumic fit
            if self.eos_type == 'Murnaghan':
                myeos = eos.murnaghan_EV

            elif self.eos_type == 'Birch-Murnaghan':
                myeos = eos.birch_murnaghan_EV


            fit = np.zeros((self.ntemp))
            self.fit_params = np.zeros((4,self.ntemp))
            self.fit_params_list = "V0,E0,B0,B0p"

            for t, T in enumerate(self.temperature):
                p0 = [self.equilibrium_volume[0],self.free_energy[self.equilibrium_index,t],self.bulk_modulus,4.0]
                popt, pcov = curve_fit(myeos, self.volume[:,0],self.free_energy[:,t],p0)

                fit[t] = (4*popt[0])**(1./3)
                self.fit_params[:,t] = popt

                # Print fitting parameters if required
                if self.verbose:
                    print('\n For T = {}K:'.format(T))
                    print('    a(T) = {} bohr'.format(fit[t]))
                    print('    Bulk modulus: {:>5.0f} GPa'.format(popt[2]*cst.habo3_to_gpa))
                    print('    Covariance matrix: {}'.format(pcov))

            fit = np.expand_dims(fit,axis=0)
            print('########### T=0 K')
            print('delta = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format((fit[0,0]-self.equilibrium_volume[1]).round(4), (fit[0,0]-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))


            return fit

        if self.symmetry == 'hexagonal':

            fit = np.zeros((2,self.ntemp))
            self.volumic_fit_params = np.zeros((4,self.ntemp)) # [ [V0,E0,B0, B0'],T]
            self.c_over_a = np.zeros((self.ntemp))
#            self.fit_params = np.zeros((5,self.ntemp)) # defined in self.fit_params_list

            # Define EOS type, for volumic fit
            if self.eos_type == 'Murnaghan':
                myeos = eos.murnaghan_EV
                if self.use_axial_eos:
                    myeos2D = eos.murnaghan_EV_axial2D
            if self.eos_type == 'Birch-Murnaghan':
                myeos = eos.birch_murnaghan_EV
                if self.use_axial_eos:
                    myeos2D = eos.birch_murnaghan_EV_axial2D


            for t,T in enumerate(self.temperature):

                # define bounds for optimal parameters
                bounds = [[0.95*self.equilibrium_volume[0],1.08*self.free_energy[self.equilibrium_index,t],0.1*self.bulk_modulus,0.],
                                [1.08*self.equilibrium_volume[0],0.95*self.free_energy[self.equilibrium_index,t],5*self.bulk_modulus,50.]]
     
                print('\n\n########## For T = {} K #####################\n'.format(T))

                # Fit volume with EOS
                '''FIX ME could also be done with lmfit...'''
                p0 = [self.equilibrium_volume[0], self.free_energy[self.equilibrium_index,t],self.bulk_modulus,4.0]
                popt, pcov= curve_fit(myeos, self.volume[:,0], self.free_energy[:,t], p0,bounds=bounds)
                self.volumic_fit_params[:,t] = popt
                if self.verbose:
                    print('From volumic EOS fit:')
                    print("  K0={} GPa,K0'={}".format(popt[2]/cst.gpa_to_habo3,popt[3]))

                if t == 0:
                    print('Zero-point volume change: {:>8.4f} Bohr^3'.format(popt[0]-self.equilibrium_volume[0]))


                #Free energy surface 2D fit
                if self.use_axial_eos:

                    print('   Using {} axial EOS'.format(self.eos_type))
                    self.fit_params_list = "a0, c0, E0, B0, B0p"
                    self.fit_params = np.zeros((5,self.ntemp))

                    # 2D fit with lmfit, 2D EOS
                    # First, create 2D mesh:
                    ac_mesh = np.meshgrid(self.volume[:,1],self.volume[:,3])
                    free_energy2D = self.free_energy[:,t]

                    #Create the model and initialize the parameters
                    lmfit_model = lmfit.Model(myeos2D)

                    params = lmfit_model.make_params()
                    params['a0'].set(value=self.equilibrium_volume[1], vary=True,min=0.95*self.equilibrium_volume[1],max=1.08*self.equilibrium_volume[1])
                    params['c0'].set(value=self.equilibrium_volume[3], vary=True,min=0.95*self.equilibrium_volume[3],max=1.08*self.equilibrium_volume[3])
                    params['E0'].set(value=free_energy2D[self.equilibrium_index],vary=True,min=1.10*free_energy2D[self.equilibrium_index],max=0.95*free_energy2D[self.equilibrium_index])
                    params['K0'].set(value=self.bulk_modulus, vary=True,min=0.1*self.bulk_modulus,max=2*self.bulk_modulus)
                    params['K0p'].set(value=4.0, vary=True,min=2.0,max=30.0)

                    lmfit_result = lmfit_model.fit(free_energy2D,mesh=[self.volume[:,1],self.volume[:,3]],a0=params['a0'],c0=params['c0'],E0=params['E0'],K0=params['K0'],K0p=params['K0p'])
                    print(lmfit_result.params.pretty_print())

                    if self.verbose:
                        print(lmfit_result.fit_report())

                    fit[0,t] = lmfit_result.params['a0'].value
                    fit[1,t] = lmfit_result.params['c0'].value

                    self.fit_params[0,t] = lmfit_result.params['a0'].value
                    self.fit_params[1,t] = lmfit_result.params['c0'].value
                    self.fit_params[2,t] = lmfit_result.params['E0'].value
                    self.fit_params[3,t] = lmfit_result.params['K0'].value
                    self.fit_params[4,t] = lmfit_result.params['K0p'].value
      

                ####################################
                else:

#                    print('   Using Paraboloid2D')
#
#                    self.fit_params_list = 'a0, A, c0, C, B=E0'
#                    # 2D fit with lmfit, Paraboloid function
#                    # First, create 2D mesh:
#                    ac_mesh = np.meshgrid(self.volume[:,1],self.volume[:,3])
#                    free_energy2D = self.free_energy[:,t]
#
#                    #Create the model and initialize the parameters
#                    lmfit_model_para = lmfit.Model(eos.paraboloid_2D)
#
#                    params_para = lmfit_model_para.make_params()
#                    params_para['a0'].set(value=self.equilibrium_volume[1], vary=True,min=0.95*self.equilibrium_volume[1],max=1.08*self.equilibrium_volume[1])
#                    params_para['c0'].set(value=self.equilibrium_volume[3], vary=True,min=0.95*self.equilibrium_volume[3],max=1.08*self.equilibrium_volume[3])
#                    params_para['A'].set(value=1.,vary=True,min=0.)
#                    params_para['C'].set(value=1.,vary=True,min=0.)
#                    params_para['B'].set(value=free_energy2D[self.equilibrium_index],vary=True,min=1.10*free_energy2D[self.equilibrium_index],max=0.95*free_energy2D[self.equilibrium_index])
#
#                    lmfit_result_para=lmfit_model_para.fit(free_energy2D,mesh=[self.volume[:,1],self.volume[:,3]],a0=params_para['a0'],A=params_para['A'],c0=params_para['c0'],C=params_para['C'],B=params_para['B'])
#                    print(lmfit_result_para.params.pretty_print())
#
#                    if self.verbose:
#                        print(lmfit_result_para.fit_report())
#
#                    fit[0,t] = lmfit_result_para.params['a0'].value
#                    fit[1,t] = lmfit_result_para.params['c0'].value
#
#                    self.fit_params[0,t] = lmfit_result_para.params['a0'].value
#                    self.fit_params[1,t] = lmfit_result_para.params['A'].value
#                    self.fit_params[2,t] = lmfit_result_para.params['c0'].value
#                    self.fit_params[3,t] = lmfit_result_para.params['C'].value
#                    self.fit_params[4,t] = lmfit_result_para.params['B'].value
#
                    ## Test using a more general quadratic surface (i.e. rotated paraboloid)
                    print('   Using Ellipsoid2D')

                    self.fit_params_list = 'A, B, C, D, E, F'
                    self.fit_params = np.zeros((6,self.ntemp))

                    # 2D fit with lmfit, Paraboloid function
                    # First, create 2D mesh:
                    ac_mesh = np.meshgrid(self.volume[:,1],self.volume[:,3])
                    free_energy2D = self.free_energy[:,t]

                    #Create the model and initialize the parameters
                    lmfit_model_ell = lmfit.Model(eos.ellipsoid_2D)

                    params_ell = lmfit_model_ell.make_params()
#                    params_ell['a0'].set(value=self.equilibrium_volume[1], vary=True,min=0.95*self.equilibrium_volume[1],max=1.08*self.equilibrium_volume[1])
#                    params_ell['c0'].set(value=self.equilibrium_volume[3], vary=True,min=0.95*self.equilibrium_volume[3],max=1.08*self.equilibrium_volume[3])
                    params_ell['A'].set(value=1.,vary=True)
                    params_ell['B'].set(value=1.,vary=True)
                    params_ell['C'].set(value=1.,vary=True)
                    params_ell['D'].set(value=1.,vary=True)
                    params_ell['E'].set(value=1.,vary=True)
                    params_ell['F'].set(value=free_energy2D[self.equilibrium_index],vary=True)

#                    params_ell['F'].set(value=free_energy2D[self.equilibrium_index],vary=True,min=1.20*free_energy2D[self.equilibrium_index],max=0.80*free_energy2D[self.equilibrium_index])

                    lmfit_result_ell=lmfit_model_ell.fit(free_energy2D,mesh=[self.volume[:,1],self.volume[:,3]],A=params_ell['A'],B=params_ell['B'],C=params_ell['C'],D=params_ell['D'],E=params_ell['E'],
                            F=params_ell['F'])
                    print(lmfit_result_ell.params.pretty_print())

                    if self.verbose:
                        print(lmfit_result_ell.fit_report())

                    if t==0:
                        print('From ellipsoid')
                        A = lmfit_result_ell.params['A'].value
                        B = lmfit_result_ell.params['B'].value
                        C = lmfit_result_ell.params['C'].value
                        D = lmfit_result_ell.params['D'].value
                        E = lmfit_result_ell.params['E'].value


                        a0 = (B*E-2*C*D)/(4*A*C-B**2)
                        c0 = -1./B*(2*A*a0+D)
                        print('a0={:>7.4f}, delta a = {:>7.4f}'.format(a0, (a0-self.equilibrium_volume[1]).round(4)))
                        print('c0={:>7.4f}, delta c = {:>7.4f}'.format(c0, (c0-self.equilibrium_volume[3]).round(4)))
                        vol0 = np.sqrt(3)/2*a0**2*c0
                        print('Volume change = {} Bohr^3'.format(vol0 - self.equilibrium_volume[0]))


                    fit[0,t] = a0
                    fit[1,t] = c0

                    self.fit_params[0,t] = lmfit_result_ell.params['A'].value
                    self.fit_params[1,t] = lmfit_result_ell.params['B'].value
                    self.fit_params[2,t] = lmfit_result_ell.params['C'].value
                    self.fit_params[3,t] = lmfit_result_ell.params['D'].value
                    self.fit_params[4,t] = lmfit_result_ell.params['E'].value
                    self.fit_params[5,t] = lmfit_result_ell.params['F'].value

                
            print('########### T=0 K')
            print('delta a = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format((fit[0,0]-self.equilibrium_volume[1]).round(4), (fit[0,0]-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))
            print('delta c = {:>7.4f} bohr, delta c/c0 stat = {:>6.4f}%'.format((fit[1,0]-self.equilibrium_volume[3]).round(4), (fit[1,0]-self.equilibrium_volume[3]).round(4)/self.equilibrium_volume[3].round(4)*100))

            self.c_over_a = fit[1,:]/fit[0,:]
            c_over_a_stat = self.equilibrium_volume[3]/self.equilibrium_volume[1]
            print('delta(c/a) = {:>7.5f}, {:>7.5f}% vs static'.format(self.c_over_a[0]-c_over_a_stat, (self.c_over_a[0]-c_over_a_stat)/(c_over_a_stat)*100))
            print('stat: c={}, a={}, c/a={}'.format(self.equilibrium_volume[3], self.equilibrium_volume[1], self.equilibrium_volume[3]/self.equilibrium_volume[1]))
            print('ZPAE: c={}, a={}, c/a={}'.format(fit[1,0], fit[0,0], fit[1,0]/fit[0,0]))
            return fit

#    def residuals(self,params,x,y,z):
#
#        V = np.sqrt(3)/2*x**2*y
#        return z - eos.murnaghan_EV_axial2D([x,y],params[0],params[1],params[2],params[3],params[4])

    def compute_specific_heat(self, nqpt, nmode):
        cv = np.zeros((nqpt,nmode,self.ntemp))

        for i,n in itt.product(range(nqpt),range(nmode)):
            cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)
            
        self.cv = np.einsum('q,qvt->t',self.wtq,cv)

    def paraboloid(self, mesh, a0,A,c0,C,B):

        x,y = mesh
        z = (x-a0)**2/A**2 + (y-c0)**2/C**2 + B

        return z

    def quadratic(self, x, a,b,c):

        return a*x**2+b*x+c

#    def residuals_parab(self,params,x,y,z):
#
#        return z - self.paraboloid([x,y],params)

    def discrete_alpha_vs_reftemp(self,acell,ref_temp):

        # get the thermal expansion coefficient vs room temperature lattice parameter
        find_t, = np.where(self.temperature==ref_temp)
        if len(find_t) == 0:
            temp_index = np.argmax(np.where(self.temperature<ref_temp))
        else: 
            temp_index = find_t[0]

        nacell = acell.shape[0]
        alpha = np.zeros((nacell,self.ntemp))

        dx = self.temperature[temp_index+1] - self.temperature[temp_index] 
        ref_acell = acell[:,temp_index] + (acell[:,temp_index+1]-acell[:,temp_index])/dx*(ref_temp - self.temperature[temp_index])

        for t in range(self.ntemp):
            dt = self.temperature[t] - ref_temp

            if dt == 0.0:
                continue
            else:
                alpha[:,t] =  (acell[:,t] - ref_acell)/(ref_acell*dt)

        return alpha

    def get_alpha_from_acell(self,acell):

        # get the thermal expansion coefficient from the lattice parameter,
        # using central finite difference derivative

        nacell = acell.shape[0]
        alpha = np.zeros((nacell,self.ntemp))
        a0 = acell[:,0] 

        if self.temperature[0] != 0.:
            raise Exception('Thermal expansion coefficient calculation requires that the first temperature is 0.')

        dt = self.temperature[1]

        for t in range(self.ntemp):

            if t == 0:
                continue

            elif t == self.ntemp-1:
                # Backwards finite difference for the last one
                alpha[:,t] =  (acell[:,t] - acell[:,t-1])/(a0*dt)

            else:
                # Central finite difference
                # Should there be a warning if the step is not constant??? 
                # This should not happen from the definition of the temperature array.
                alpha[:,t] = (acell[:,t+1] - acell[:,t-1])/(2*a0*dt)

        return alpha

    def get_alpha_vs_reftemp(self,acell, ref_temp):

        # get the thermal expansion coefficient from the lattice parameter,
        # using a non-zero temperature reference
        # and central finite difference derivative

        nacell = acell.shape[0]
        alpha = np.zeros((nacell,self.ntemp))

        find_t, = np.where(self.temperature==ref_temp)
        if len(find_t) == 0:
            temp_index = np.argmax(np.where(self.temperature<ref_temp))
        else: 
            temp_index = find_t[0]

        dt = self.temperature[temp_index+1] - self.temperature[temp_index] 
        ref_acell = acell[:,temp_index] + (acell[:,temp_index+1]-acell[:,temp_index])/dt*(ref_temp - self.temperature[temp_index])
        #print(acell[:,0],ref_acell)
        #print(acell)

        for t in range(self.ntemp):

            if t == 0:
                # Forward finite difference of the first one
                alpha[:,t] =  (acell[:,t+1] - acell[:,t])/(ref_acell*dt)

            elif t == self.ntemp-1:
                # Backwards finite difference for the last one
                alpha[:,t] =  (acell[:,t] - acell[:,t-1])/(ref_acell*dt)

            else:
                # Central finite difference
                # Should there be a warning if the step is not constant??? 
                # This should not happen from the definition of the temperature array.
                alpha[:,t] = (acell[:,t+1] - acell[:,t-1])/(2*ref_acell*dt)

        return alpha

    def write_nc(self):

        nc_outfile = 'OUT/{}_TE.nc'.format(self.rootname)

        #  Write output in netCDF format
        create_directory(nc_outfile)

        with nc.Dataset(nc_outfile, 'w') as dts:

            #Define the type of calculation:
            dts.description = 'FreeEnergy'

            dts.createDimension('number_of_temperatures', self.ntemp)
            dts.createDimension('number_of_lattice_parameters', len(self.distinct_acell))
            dts.createDimension('number_of_volumes', np.shape(self.free_energy)[0])
            dts.createDimension('four', 4)
            dts.createDimension('number_of_fit_parameters', np.shape(self.fit_params)[0])

            data = dts.createVariable('temperature','d', ('number_of_temperatures'))
            data[:] = self.temperature[:]
            data.units = 'Kelvin'


            data = dts.createVariable('acell_from_gibbs','d',('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.temperature_dependent_acell[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('c_over_a', 'd', ('number_of_temperatures'))
            if self.symmetry == 'hexagonal':
                data[:] = self.c_over_a[:]
            data.units = 'Unitless'

            data = dts.createVariable('free_energy', 'd', ('number_of_volumes', 'number_of_temperatures'))
            data[:,:] = self.free_energy[:,:]
            data.units = 'hartree'

            data = dts.createVariable('static_energy', 'd', ('number_of_volumes', 'number_of_temperatures'))
            data[:,:] = self.static_energy[:,:]
            data.units = 'hartree'

            data = dts.createVariable('phonon_free_energy', 'd', ('number_of_volumes', 'number_of_temperatures'))
            data[:,:] = self.phonon_free_energy[:,:]
            data.units = 'hartree'

            data = dts.createVariable('volume','d', ('number_of_volumes','four'))
            data[:,:] = self.volume[:,:]
            data.units = 'bohr^3'

            data = dts.createVariable('equilibrium_volume','d', ('four'))
            data[:] = self.equilibrium_volume[:]
            data.units = 'bohr^3'

            data = dts.createVariable('fit_parameters', 'd', ('number_of_fit_parameters','number_of_temperatures'))
            data[:,:] = self.fit_params[:,:]
            data.units = self.fit_params_list
            if self.symmetry == 'hexagonal':
                if self.use_axial_eos:
                    data.description = self.eos_type
                else:
                    data.description = 'Ellipsoid2D'
            elif self.symmetry == 'cubic':
                data.description = self.eos_type


            data = dts.createVariable('volumic_fit_parameters', 'd', ('four','number_of_temperatures'))
            if self.symmetry == 'hexagonal':
                data[:,:] = self.volumic_fit_params[:,:]
                data.units = "V0,E0,B0,B0p"
                data.description = self.eos_type

            data = dts.createVariable('alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('discrete_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.discrete_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('discrete_room_temp_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.discrete_room_temp_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('room_temp_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.room_temp_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('specific_heat','d',('number_of_temperatures'))
            data[:] = self.cv
            data.units = 'Ha/K'


    def write_acell(self):

        # Write results from Free energy minimisation in ascii format
        outfile = 'OUT/{}_acell_from_eos.dat'.format(self.rootname)
        create_directory(outfile)

        with open(outfile, 'w') as f:

            if self.symmetry == 'cubic':
                eos_str = self.eos_type
            elif self.symmetry == 'hexagonal':
                if self.use_axial_eos:
                    eos_str = self.eos_type
                else:
                    eos_str = 'Paraboloid2D'
                    
            f.write('Temperature dependent lattice parameters via Gibbs free energy, using {} EOS\n\n'.format(eos_str))

            if self.symmetry == 'cubic':


                f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t]))

                f.close()


            if self.symmetry == 'hexagonal':

                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t],self.temperature_dependent_acell[1,t]))

                # Independent fit: fitg, 2D fit: fit2dg
#                f.write('\n\nTemperature dependent lattice parameters via Gibbs free energy\n\n')
#                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
#                for t,T in enumerate(self.temperature):
 #                   f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fitg[0,t],self.fitg[1,t]))



                f.close()


class Gruneisen(FreeEnergy):

    #Input files
    ddb_flists = None
    out_flists = None
    elastic_fname = None

    #Parameters
    wtq = [1.0]
    temperature = None
    pressure = 0.0
    pressure_gpa= 0.0 

    check_anaddb = False
    manual_correction = False
    verbose = False
    bulk_modulus = None

    eos_type = 'Murnaghan'

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
        pressure_gpa = 0.0,
        pressure_units = None,
        bulk_modulus_units = None,
        manual_correction = False,
        verbose = False,

        eos_type = 'Murnaghan',

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
        self.manual_correction = manual_correction
        self.verbose = verbose

        self.temperature = temperature
        self.ntemp = len(self.temperature) 
        
        self.pressure_units = pressure_units
        self.pressure = pressure

        if self.pressure_units == 'GPa':
            self.pressure_gpa = self.pressure
            self.pressure = self.pressure*cst.gpa_to_habo3
        else:
            self.pressure_gpa == self.pressure*cst.habo3_to_gpa

        print('External pressure is {} GPa'.format(self.pressure_gpa))

        if bulk_modulus:
            if bulk_modulus_units == 'GPa':
                self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
            elif bulk_modulus_units == 'HaBo3':
                self.bulk_modulus = bulk_modulus
            else:
                raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')
################# This could be in a function??
        #Define EOS type
        self.eos_type = eos_type
        if self.eos_type not in self.eos_list:
            raise Exception('EOS type must be one of the following: {}'.format(self.eos_list))

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

            print('\n\nReading data from {}'.format(out_flists[v]))
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
                self.eigvect = np.zeros((nvol, nqpt, nmode, nmode), dtype=complex)
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
                    print('Qpt is Gamma')
                else:
                    ddb.compute_dynmat()
                        ##### CHECK WITH GABRIEL IF I SHOULD HAVE LOTO SPLITTING AT GAMMA (WHERE DOES HIS CODE TREAT THE ELECTRIC FIELD PERTURBAITON IN THE DDB AT GAMMA???)
                        ### I should maybe implement the proper ASR corrections, like in anaddb and phonopy...

                # Store frequencies for Gruneisen parameters
                self.omega[v,i,:] = ddb.omega

                # get F0 contribution
                F_0 += self.wtq[i]*self.get_f0(ddb.omega) 
                # get Ftherm contribution
                F_T += self.wtq[i]*self.get_fthermal(ddb.omega,nmode)

        # Add a check for the right volumes ordering?
        # cubic : aminus equil aplus
        #hexagonal : aminus equil aplus cminus equil cplus
            # FIX ME :Do I really need this???
                
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
        if self.verbose:
            for t,T in enumerate(self.temperature):
                print('For T={} K'.format(T))
                print(self.gibbs_free_energy[:,t])
                print('Minimal Gibbs free energy has index {}'.format(np.argmin(self.gibbs_free_energy[:,t])))
                print('sorted order: {}'.format(self.gibbs_free_energy[:,t].argsort()))   



        # Minimize F
        #Here, I have F[nvol,T] and also the detailed acell for each volume
        #But I do not have a very detailed free energy surface. I should interpolate it on a finer mesh, give a model function? 
        # For Helmholtz, what is usually done is to fit the discrete F(V,T) = F(a,T), F(b,T), F(c,T)... (each separately) with a parabola, one I have the fitting parameters I can
            # easily get the parabola's minimum.

        # To check the results, add the fitting parameters in the output file. So the fitting can be plotted afterwards.

        # I will have to think about what to do when there is also pressure... do I just use a paraboloid for fitting and minimizing?
        # That would be the main idea. If there is 1 independent acell, it is a parabola (x^2), if there are 2 it is a paraboloid (x^2 + y^2), if there are 3 it would be a paraboloic "volume" (x^2 +
        # y^2 + z^2)
        
        # Read elastic compliance from file
        if self.elastic_fname:
            elastic = ElasticFile(self.elastic_fname)
            self.compliance = elastic.compliance_relaxed
            self.compliance_rigid = elastic.compliance_clamped

            bmod = self.get_bulkmodulus_from_elastic(elastic.stiffness_relaxed)
            bmod2 = self.get_bulkmodulus_from_elastic(elastic.stiffness_clamped)
            self.bulkmodulus_from_elastic = bmod
            ##FIX ME: what to do i we want to use th inputted (experimental) bulk modulus?)
            if self.symmetry == 'cubic':
                if self.bulk_modulus is None:
                    print('Using bulk modulus from elastic constants.')
                    self.bulk_modulus = self.bulkmodulus_from_elastic*cst.gpa_to_habo3
                    self.bulk_modulus_rigid = bmod2*cst.gpa_to_habo3
                else:
                    print('Using bulk modulus from input file.')

            if self.verbose:
                print('Bulk modulus from elastic constants = {:>7.3f} GPa'.format(bmod))
                print('Bulk modulus from elastic constants (clamped) = {:>7.3f} GPa'.format(bmod2))

                print('Elastic constants:')
                print('c11 = {}, c33 = {}, c12 = {}, c13 = {} GPa'.format(elastic.stiffness_relaxed[0,0],elastic.stiffness_relaxed[2,2],elastic.stiffness_relaxed[0,1],elastic.stiffness_relaxed[0,2]))
                print('Clamped:')
                print('c11 = {}, c33 = {}, c12 = {}, c13 = {} GPa'.format(elastic.stiffness_clamped[0,0],elastic.stiffness_clamped[2,2],elastic.stiffness_clamped[0,1],elastic.stiffness_clamped[0,2]))

                print('Compliance constants:')
                print('s11 = {}, s33 = {}, s12 = {}, s13 = {} GPa^-1'.format(elastic.compliance_relaxed[0,0],elastic.compliance_relaxed[2,2],elastic.compliance_relaxed[0,1],elastic.compliance_relaxed[0,2]))
                print('Clamped:')
                print('s11 = {}, s33 = {}, s12 = {}, s13 = {} GPa^-1'.format(elastic.compliance_clamped[0,0],elastic.compliance_clamped[2,2],elastic.compliance_clamped[0,1],elastic.compliance_clamped[0,2]))



        # Get Gruneisen parameters and alpha, according to crystal symmetry

        ### Add a check for homogenious acell increase (for central finite difference)
        self.equilibrium_volume = self.volume[1,:]
        self.minimize_free_energy_from_eos()


        self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
        self.gruneisen_from_dynmat = self.get_gruneisen_from_dynmat(nqpt,nmode,nvol)

        self.acell_via_gruneisen = self.get_acell(nqpt,nmode)
        self.acell_via_gruneisen_from_dynmat = self.get_acell_from_dynmat(nqpt,nmode)

        # FIX ME: add a variable for the reference temperature, and make this computation optional
        self.discrete_room_temp_alpha = self.discrete_alpha_vs_reftemp(self.acell_via_gruneisen, 293.)
        self.discrete_alpha = self.discrete_alpha_vs_reftemp(self.acell_via_gruneisen, 0.)
        self.room_temp_alpha= self.get_alpha_vs_reftemp(self.acell_via_gruneisen, 293.)
        

    def minimize_free_energy_from_eos(self):
        # Minimize_free_energy, with an EOS, to obtain the fitting parameters E0, V0, B0, B0'
        # When done in the Gruneisen class, the purpose is mainly to extract B0(T)
        # Volumic fit only as fitting with (a_0, c_0) is not stable
        # Not available for cubic symmetry, as there are not enough volumes present.

        if self.symmetry == 'cubic':

            if self.verbose:
                print('Free energy will not be minimized, as there are not enough volumes.')
                
              # Keep the code for now, maybe I could add it back it I implement the finite
              # difference with more than 3 volumes

#             # Define EOS type, for volumic fit
#             if self.eos_type == 'Murnaghan':
#                 myeos = eos.murnaghan_EV
# 
#             elif self.eos_type == 'Birch-Murnaghan':
#                 myeos = eos.birch_murnaghan_EV
# 
# 
#             self.eos_fit_params = np.zeros((4,self.ntemp))
#             self.eos_fit_params_list = "V0,E0,B0,B0p"
# 
#             for t, T in enumerate(self.temperature):
#                 p0 = [self.equilibrium_volume[0],self.free_energy[1,t],self.bulk_modulus,4.0]
#                 popt, pcov = curve_fit(myeos, self.volume[:,0],self.free_energy[:,t],p0)
# 
#                 self.eos_fit_params[:,t] = popt
# 
#                 # Print fitting parameters if required
#                 if self.verbose:
#                     print('\n For T = {}K:'.format(T))
#                     print('    Bulk modulus: {:>5.0f} GPa'.format(popt[2]*cst.habo3_to_gpa))
#                     print('    Covariance matrix: {}'.format(pcov))

            return 

        if self.symmetry == 'hexagonal':

            self.eos_fit_params = np.zeros((4,self.ntemp)) # [ [V0,E0,B0, B0'],T]
            self.eos_fit_params_list = "V0,E0,B0,B0p"

            # Define EOS type, for volumic fit
            if self.eos_type == 'Murnaghan':
                myeos = eos.murnaghan_EV
            if self.eos_type == 'Birch-Murnaghan':
                myeos = eos.birch_murnaghan_EV

            for t, T in enumerate(self.temperature):


                p0 = [self.equilibrium_volume[0],self.free_energy[1,t],self.bulk_modulus,4.0]
                # define bounds for optimal parameters
                bounds = [[0.95*self.equilibrium_volume[0],1.08*self.free_energy[1,t],0.1*self.bulk_modulus,0.],
                                [1.10*self.equilibrium_volume[0],0.95*self.free_energy[1,t],5*self.bulk_modulus,50.]]

                popt, pcov = curve_fit(myeos, self.volume[:,0], self.free_energy[:,t], p0, bounds=bounds)

                self.eos_fit_params[:,t] = popt

                # Print fitting parameters if required
                if self.verbose:
                    print('\n For volumic EOS fit at T = {}K:'.format(T))
                    print('    Bulk modulus: {:>5.0f} GPa, bulk modulud derivative: {}'.format(popt[2]*cst.habo3_to_gpa, popt[3]))
                    print('    Covariance matrix: {}'.format(pcov))

            return 


    # Clean up which functions are really necessary in this class. 
    # If they may be used by any class, move them to the global class
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

        if self.symmetry == 'cubic':
            
            gru = np.zeros((nqpt,nmode))
            self.gruvol = np.zeros((nqpt,nmode))

            for q,v in itt.product(range(nqpt),range(nmode)):

                if q==0 and v<3:
                # put Gruneisen at zero for acoustic modes at Gamma
                    gru[q,v] = 0
                else:
                    # This is the LINEAR gruneisen parameters
                    gru[q,v] = -self.equilibrium_volume[1]/self.omega[1,q,v]*(self.omega[2,q,v]-self.omega[0,q,v])/(self.volume[2,1]-self.volume[0,1])

                    # This is the VOLUMIC one (that is, gru(linear)/3)
                    self.gruvol[q,v] = -self.equilibrium_volume[0]/self.omega[1,q,v]*np.polyfit(self.volume[:,0],self.omega[:,q,v],1)[0]

            # correct divergence at q-->0
            # this would extrapolate Gruneisen at q=0 from neighboring qpts
            #x = [np.linalg.norm(self.qred[1,:]),np.linalg.norm(self.qred[2,:])]
            #for v in range(3):
            #    y = [gru[1,v],gru[2,v]]
            #    gru[0,v] = np.polyfit(x,y,1)[1]
            
            gru = np.expand_dims(gru, axis=0)

            return gru


        if self.symmetry == 'hexagonal':
            
            gru = np.zeros((2,nqpt,nmode)) # Gru_a, Gru_c
            self.gru_vol = np.zeros((nqpt,nmode))
            # FIX ME: what ios gru2?!?
            gru2 = np.zeros((2,nqpt,nmode)) #withfinite difference, on the frequencies

            # FIX ME: these were custom for BiTeI, no need for it in the master version
            # or, add it in a function, that can be called if necessary
            nlarge = np.zeros((2)) #remove after testing large Gruneisens
            large_a = []
            large_c = []
            nlarge_5 = np.zeros((2))
            sign_a = [0,0]
            sign_a5 = [0,0]
            sign_c = [0,0]
            sign_c5 = [0,0]
            
            verylarge_a = []
            verylarge_c = []

            omega_la = []
            omega_lc = []
            delta_la = []
            delta_lc = []

            dela = 0.
            delc = 0.
            delaabs = 0.
            delcabs = 0.

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


                    #This is to check the average and absoluta avergae value of delta omega
                    dela += self.omega[2,q,v]-self.omega[0,q,v]
                    delc += self.omega[5,q,v]-self.omega[3,q,v]
                    delaabs += np.abs(self.omega[2,q,v]-self.omega[0,q,v])
                    delcabs += np.abs(self.omega[5,q,v]-self.omega[3,q,v])
                    # This is the VOLUMIC one (that is, gru(linear)/3)
                    self.gru_vol[q,v] = (gru[0,q,v] + gru[1,q,v])/3.

                    # Test to evaluate the impact of the very large Gruneisens on the final sum
                    if np.abs(gru[0,q,v])>5:
                        nlarge_5[0] +=1
                        omega_la.append(self.omega[1,q,v])
                        delta_la.append(self.omega[2,q,v] - self.omega[0,q,v])
                        #gru[0,q,v] = 0.
                        if np.sign(gru[0,q,v])>0:
                            sign_a5[0] +=1
                        else:
                            sign_a5[1] +=1

                        if np.abs(gru[0,q,v])>10:
                            verylarge_a.append(1)
                            if np.sign(gru[0,q,v])>0:
                                sign_a[0] +=1
                            else:
                                sign_a[1] +=1
                            #gru[0,q,v] = 0.
                            nlarge[0] += 1
                        else:
                            verylarge_a.append(0)
                        large_a.append([q,v]) 

                    if np.abs(gru[1,q,v])>5:
                        nlarge_5[1] +=1
                        omega_lc.append(self.omega[1,q,v])
                        delta_lc.append(self.omega[5,q,v]-self.omega[3,q,v])
                        #gru[1,q,v] = 0.
                        if np.sign(gru[1,q,v])>0:
                            sign_c5[0]+=1
                        else:
                            sign_c5[1]+=1

                        if np.abs(gru[1,q,v])>10:
                            verylarge_c.append(1)
                            if np.sign(gru[1,q,v])>0:
                                sign_c[0]+=1
                            else:
                                sign_c[1]+=1

                            #gru[1,q,v] = 0.
                            nlarge[1] += 1
                        else:
                            verylarge_c.append(0)
                        large_c.append([q,v])


            if self.verbose:
                print('For |gru|>5:')
                print('Number of large Gruneisen parameters put to 0 : {} for gamma_a, {} for gamma_c'.format(nlarge_5[0],nlarge_5[1]))
                print('Gamma a : {}+, {}-, Gamma c : {}+, {}-'.format(sign_a5[0],sign_a5[1],sign_c5[0],sign_c5[1]))
                print('For |gru|>10:')
                print('Number of large Gruneisen parameters put to 0 : {} for gamma_a, {} for gamma_c'.format(nlarge[0],nlarge[1]))
                print('Gamma a : {}+, {}-, Gamma c : {}+, {}-'.format(sign_a[0],sign_a[1],sign_c[0],sign_c[1]))

                print('They are for (q,v):')
                print('Gamma_a: (q,v), verylarge, omega0, delta omega')
                if len(large_a) != 0:
                    if len(large_a)==1:
                        print('{}, {}, {}, {}'.format(large_a,verylarge_a,omega_la,delta_la))
                    else:
                        for p,pp in enumerate(large_a):
                            print('{}, {}, {}, {}'.format(pp,verylarge_a[p],omega_la[p],delta_la[p]))

                print('Gamma_c: (q,v), verylarge, omega0, delta omega')
                if len(large_c) != 0:
                    if len(large_c)==1:
                        print('{}, {}, {}, {}'.format(large_c,verylarge_c,omega_lc,delta_lc))
                    else:
                        for p,pp in enumerate(large_c):
                            print('{}, {}, {}, {}'.format(pp,verylarge_c[p],omega_lc[p],delta_lc[p]))

                print('delta omega_a average (abs): {} ( {})'.format(dela/585.,delaabs/585.))
                print('delta omega_ac average (abs): {} ( {})'.format(delc/585.,delcabs/585.))

                gg = open('{}_largegruneisens.dat'.format(self.rootname),'w')
                gg.write('Some information about large Gruneisen parameters\n\n')
                gg.write('Mode Gruneisens with |gamma|>5: there  are {} for gamma_a ({}+, {}-) and {} for gamma_c ({}+,{}-)\n'.format(nlarge_5[0],sign_a5[0],sign_a5[1],nlarge_5[1],sign_c5[0],sign_c5[1]))
                gg.write('Mode Gruneisens with |gamma|>10: there  are {} for gamma_a ({}+, {}-) and {} for gamma_c ({}+,{}-)\n\n'.format(nlarge[0],sign_a[0],sign_a[1],nlarge[1],sign_c[0],sign_c[1]))

                
                gg.write('Gamma_a: <delta omega> = {:>.4e} Ha, <|delta omega|> = {:>.4e} Ha\n'.format(dela/585.,delaabs/585.))
                gg.write('{:8}  {:9}  {:12}  {:12}\n'.format('(q,v)','verylarge','omega0','delta omega'))
                if len(large_a) != 0:
                    if len(large_a) == 1:
                        gg.write('{:^8} {:^8}  {:>.4e}    {:>.4e}\n'.format(str(large_a),str(verylarge_a),omega_la,delta_la)) 
                    else:
                        for p,pp in enumerate(large_a):
                            gg.write('{:^8} {:^8}  {:>.4e}    {:>.4e}\n'.format(str(pp),str(verylarge_a[p]),omega_la[p],delta_la[p]))
                
                gg.write('\n\nGamma_c: <delta omega> = {:>.8e} Ha, <|delta omega|> = {:>.8e} Ha\n'.format(delc/585.,delcabs/585.))
                gg.write('{:8}  {:9}  {:12}  {:12}\n'.format('(q,v)','verylarge','omega0','delta omega'))
                if len(large_c) != 0:
                    if len(large_c) == 1:
                        gg.write('{:^8} {:^8}  {:>.4e}    {:>.4e}\n'.format(str(large_c),str(verylarge_c),omega_lc,delta_lc)) 
                    else:
                        for p,pp in enumerate(large_c):
                            gg.write('{:^8} {:^8}  {:>.4e}    {:>.4e}\n'.format(str(pp),str(verylarge_c[p]),omega_lc[p],delta_lc[p]))

                gg.close()
     


            self.gru2 = gru2
#            print(gru_tot[0,3:])

            return gru 

#
            
    def get_acell(self, nqpt, nmode):

        # Evaluate acell(T) from Gruneisen parameters
        if self.symmetry == 'cubic':
            
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

            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0, :, :])/(9*self.equilibrium_volume[0]*self.bulk_modulus)
            alphavol=np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruvol)/(self.equilibrium_volume[0]*self.bulk_modulus)
            #alpha2 = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0, :, :])/(self.equilibrium_volume[0]*self.bulk_modulus)


            # Then, get a(T)
            integral = 1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[0, :, :])
            a = self.equilibrium_volume[1]*(integral+1)

            # Renormalize the a(T=0) with the zero-point energy contribution
            integral_plushalf = 1./(9*self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt_plushalf,self.gruneisen[0, :, :])
            new_acell0 = self.equilibrium_volume[1]*(1+1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qv,qv->',self.wtq,self.omega[1,:,:],self.gruneisen[0, :, :])*0.5)
            print('################ static_acell0 : {:>7.4f} bohr, new_acell0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4), new_acell0.round(4)))
            print('delta = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_acell0-self.equilibrium_volume[1], (new_acell0-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))
            new_V0 = new_acell0**3/4.
            print('stat V0: {}, newV0: {}'.format(self.equilibrium_volume[0], new_V0))
            integral_newacell0 = 1./(9*self.bulk_modulus*new_V0)*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen[0, :, :])

            a_plushalf = new_acell0*(integral+1)
            #a_plushalf = new_acell0*(integral_newacell0+1)
            
            test_acell = self.equilibrium_volume[1]*(integral_plushalf + 1)
            print('testacell={} bohr'.format(test_acell[0]))


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
            self.alpha = np.expand_dims(alpha, axis=0)

            self.cv = np.einsum('q,qvt->t',self.wtq,cv)
            return a

        if self.symmetry == 'hexagonal':
            
            
            plot = False
            self.c_over_a = np.zeros((self.ntemp))
            if self.verbose:
                ggg = open('{}_integrals.dat'.format(self.rootname),'w')   

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

            self.alpha_a = alpha_a
            self.alpha_c = alpha_c
            alpha_a2 = ( (self.compliance_rigid[0,0]+self.compliance_rigid[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]
            alpha_c2 = ( 2*self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[0,:,:]) +
                self.compliance_rigid[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen[1,:,:]))/self.equilibrium_volume[0]

            alpha_af = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
                self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]
            alpha_cf = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
                self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]


            self.cv = np.einsum('q,qvt->t',self.wtq,cv)

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

            #daa_slope = np.polyfit(self.temperature[16:],daa[16:],1)
            #print('Delta a/a intersect: {:>8.5e}, new a0 = {} bohr'.format(daa_slope[1],-daa_slope[1]*self.equilibrium_volume[1]+self.equilibrium_volume[1]))
            #dcc_slope = np.polyfit(self.temperature[16:],dcc[16:],1)
            #print('Delta c/c intersect: {:>8.5e}, new c0 = {} bohr'.format(dcc_slope[1],-dcc_slope[1]*self.equilibrium_volume[3]+self.equilibrium_volume[3]))

            a2 = (self.compliance_rigid[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a2 = self.equilibrium_volume[1]*(a2/self.equilibrium_volume[0] + 1)

            c2 = 2*self.compliance_rigid[0,2]*integral_a + self.compliance[2,2]*integral_c
            c2 = self.equilibrium_volume[3]*(c2/self.equilibrium_volume[0] + 1)

            # Test the da/a at T=0, using the 1/2 factor
            da0 = ((self.compliance[0,0]+self.compliance[0,1])*integral_a0 + self.compliance[0,2]*integral_c0)/self.equilibrium_volume[0]
            dc0 = (2*self.compliance[0,2]*integral_a0 + self.compliance[2,2]*integral_c0)/self.equilibrium_volume[0]


            new_a0 = self.equilibrium_volume[1]*(1 + da0)
            new_c0 = self.equilibrium_volume[3]*(1 + dc0)
            aplushalf = new_a0*(aterm/self.equilibrium_volume[0] + 1)
            cplushalf = new_c0*(cterm/self.equilibrium_volume[0] + 1)


            acell = np.array([a,c])
            self.acell_plushalf = np.array([aplushalf,cplushalf])
            self.c_over_a = cplushalf/aplushalf


            print('############### T=0')
            #new_acell0 = self.equilibrium_volume[1]*(1+1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qv,qv->',self.wtq,self.omega[1,:,:],self.gruneisen)*0.5)
            print('################ static_acell :a0 = {:>7.4f} bohr, new_a0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4), new_a0.round(4)))
            print('delta a = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_a0-self.equilibrium_volume[1], (new_a0-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))

            print('################ static_acell :c0 = {:>7.4f} bohr, new_c0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[3].round(4), new_c0.round(4)))
            print('delta c = {:>7.4f} bohr, delta c/c0 stat = {:>6.4f}%'.format(new_c0-self.equilibrium_volume[3], (new_c0-self.equilibrium_volume[3]).round(4)/self.equilibrium_volume[3].round(4)*100))

            c_over_a_stat = self.equilibrium_volume[3]/self.equilibrium_volume[1]
            print('delta(c/a) = {:>7.5f}, {:>7.5f}% vs static'.format(self.c_over_a[0]-c_over_a_stat, (self.c_over_a[0]-c_over_a_stat)/(c_over_a_stat)*100))
            print('stat: c={}, a={}, c/a={}'.format(self.equilibrium_volume[3], self.equilibrium_volume[1], self.equilibrium_volume[3]/self.equilibrium_volume[1]))
            print('ZPAE: c={}, a={}, c/a={}'.format(cplushalf[0], aplushalf[0], self.c_over_a[0]))

#            self.acell2 = np.array([a2,c2])
            if self.verbose:
                'write details of compliance vs intergrals in <rootname>_integrals.dat file'''
                ggg.write('Terms entering the thermal expansion of a\n\n')
                ggg.write('{:15}  {:18}  {:18}  {:18}  {:18}\n'.format('Temperature (K)','compliance_a','integral_a','compliance_c','integral_c'))
                for t, T in enumerate(self.temperature):
                    ggg.write('{:>15.0f}  {:>.12e}  {:>.12e}  {:>.12e}  {:>.12e}\n'.format(T,self.compliance[0,0]+self.compliance[0,1],integral_a[t],self.compliance[0,2],integral_c[t]))
                ggg.write('\n\nTerms entering the thermal expansion of c\n\n')
                ggg.write('{:15}  {:18}  {:18}  {:18}  {:18}\n'.format('Temperature (K)','compliance_a','integral_a','compliance_c','integral_c'))
                for t, T in enumerate(self.temperature):
                    ggg.write('{:>15.0f}  {:>.12e}  {:>.12e}  {:>.12e}  {:>.12e}\n'.format(T,2*self.compliance[0,2],integral_a[t],self.compliance[2,2],integral_c[t]))

                ggg.close()

            print('From Gruneisen parameters, using phonon frequencies')
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

                #arr[0,2].plot(self.temperature, a*cst.bohr_to_ang,'r',label='n')
                #arr[0,2].plot(self.temperature, aplushalf*cst.bohr_to_ang,'k:',label='n + 1/2')
                #arr[0,2].plot(self.temperature, self.fitg[0,:]*cst.bohr_to_ang, 'o',color='m',label='Gibbs')
#                arr[0,1].plot(self.temperature, a2*cst.bohr_to_ang,'g',label='rigid')


#                twin1 = arr[1].twinx()
                arr[1,1].plot(self.temperature,dcc*1E3,'b',label='relaxed')
                #arr[1,2].plot(self.temperature,c*cst.bohr_to_ang,'b',label='n')
                #arr[1,2].plot(self.temperature,cplushalf*cst.bohr_to_ang,'k:',label='n + 1/2')
                #arr[1,2].plot(self.temperature, self.fitg[1,:]*cst.bohr_to_ang, 'o',color='m',label='Gibbs')
               

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
                # Reeber 1999 data
                reeber_alphaA = EXPfile('experimental_data/GaN_alphaAT_Reeber1999.nc')
                reeber_alphaA.read_nc()
                reeber_alphaC = EXPfile('experimental_data/GaN_alphaCT_Reeber1999.nc')
                reeber_alphaC.read_nc()
                reeber_a = EXPfile('experimental_data/GaN_aT_Reeber1999.nc')
                reeber_a.read_nc()
                reeber_c = EXPfile('experimental_data/GaN_cT_Reeber1999.nc')
                reeber_c.read_nc()

                arr[0,0].plot(reeber_alphaA.xaxis, reeber_alphaA.yaxis,'xr',label='Reeber1999')
                arr[1,0].plot(reeber_alphaC.xaxis, reeber_alphaC.yaxis,'xb',label='Reeber1999')
                arr[0,2].plot(reeber_a.xaxis, reeber_a.yaxis,'xr',label='Reeber1999')
                arr[1,2].plot(reeber_c.xaxis, reeber_c.yaxis,'xb',label='Reeber1999')

#                arr[0,1].plot(self.temperature,self.independent_fit[0,:]*cst.bohr_to_ang,'go',markersize=7,label=r'ind\_fit')
#                arr[1,1].plot(self.temperature,self.independent_fit[1,:]*cst.bohr_to_ang,'go',markersize=7,label=r'ind\_fit')
#                arr[0,1].plot(self.temperature,self.fit2d[0,:]*cst.bohr_to_ang,'mo',markersize=5,label='fit2d')
#                arr[1,1].plot(self.temperature,self.fit2d[1,:]*cst.bohr_to_ang,'mo',markersize=5,label='fit2d')

                roder_a = EXPfile('experimental_data/GaN_aT_Roder2005.nc')
                roder_a.read_nc()
                roder_a1 = EXPfile('experimental_data/GaN_aT_Roder2005_ref1.nc')
                roder_a1.read_nc()
                roder_a4 = EXPfile('experimental_data/GaN_aT_Roder2005_ref4.nc')
                roder_a4.read_nc()
                roder_a6 = EXPfile('experimental_data/GaN_aT_Roder2005_ref6.nc')
                roder_a6.read_nc()
                roder_c = EXPfile('experimental_data/GaN_cT_Roder2005.nc')
                roder_c.read_nc()
                roder_c1 = EXPfile('experimental_data/GaN_cT_Roder2005_ref1.nc')
                roder_c1.read_nc()
                roder_c4 = EXPfile('experimental_data/GaN_cT_Roder2005_ref4.nc')
                roder_c4.read_nc()
                roder_c6 = EXPfile('experimental_data/GaN_cT_Roder2005_ref6.nc')
                roder_c6.read_nc()

                arr[0,2].plot(roder_a.xaxis, roder_a.yaxis,'vk',label='Roder2005')
                arr[1,2].plot(roder_c.xaxis, roder_c.yaxis,'vk',label='Roder2005')
                arr[0,2].plot(roder_a1.xaxis, roder_a1.yaxis,'Pk',label='Roder2005-ref1')
                arr[1,2].plot(roder_c1.xaxis, roder_c1.yaxis,'Pk',label='Roder2005-ref1')
                arr[0,2].plot(roder_a4.xaxis, roder_a4.yaxis,'sk',label='Roder2005-ref4')
                arr[1,2].plot(roder_c4.xaxis, roder_c4.yaxis,'sk',label='Roder2005-ref4')
                arr[0,2].plot(roder_a6.xaxis, roder_a6.yaxis,'*k',label='Roder2005-ref6')
                arr[1,2].plot(roder_c6.xaxis, roder_c6.yaxis,'*k',label='Roder2005-ref6')

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
                deltaa = (a[0]*cst.bohr_to_ang-reeber_a.yaxis[0])*np.ones(self.ntemp)
                deltac = (c[0]*cst.bohr_to_ang-reeber_c.yaxis[0])*np.ones(self.ntemp)
                deltaa2 = (aplushalf[0]*cst.bohr_to_ang-reeber_a.yaxis[0])*np.ones(self.ntemp)
                deltac2 = (cplushalf[0]*cst.bohr_to_ang-reeber_c.yaxis[0])*np.ones(self.ntemp)

                arr[0,2].plot(self.temperature,a*cst.bohr_to_ang-deltaa,'r',linestyle='solid',label='n (shifted)')
                arr[0,2].plot(self.temperature,aplushalf*cst.bohr_to_ang-deltaa2,'k',linestyle='dotted',label='n + 1/2 (shifted)')
                arr[1,2].plot(self.temperature,c*cst.bohr_to_ang-deltac,'b',linestyle='solid',label='n (shifted)')
                arr[1,2].plot(self.temperature,cplushalf*cst.bohr_to_ang-deltac2,'k',linestyle='dotted',label='n + 1/2 (shifted)')


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

            self.alpha = np.array([alpha_a, alpha_c])

            return acell

    def get_acell_from_dynmat(self, nqpt, nmode):

        # Evaluate acell(T) from Gruneisen parameters, using the dynamical matrix variation
        if self.symmetry == 'cubic':
            
            # First, get alpha(T)

            # Get Bose-Einstein factor and specific heat Cv
            self.bose = np.zeros((nqpt,nmode, self.ntemp))
            cv = np.zeros((nqpt,nmode,self.ntemp))

            for i,n in itt.product(range(nqpt),range(nmode)):
                self.bose[i,n,:] = self.get_bose(self.omega[1,i,n],self.temperature)
                cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)

            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            alpha = np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[0, :, :])/(9*self.equilibrium_volume[0]*self.bulk_modulus)


            # Then, get a(T)
            integral = 1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen_from_dynmat[0, :, :])
            a = self.equilibrium_volume[1]*(integral+1)

            # Renormalize the a(T=0) with the zero-point energy contribution
            new_acell0 = self.equilibrium_volume[1]*(1+1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qv,qv->',self.wtq,self.omega[1,:,:],self.gruneisen_from_dynmat[0, :, :])*0.5)
            print('################ static_acell0 : {:>7.4f} bohr, new_acell0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4), new_acell0.round(4)))
            print('delta = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_acell0-self.equilibrium_volume[1], (new_acell0-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))
            new_V0 = new_acell0**3/4.
            print('stat V0: {}, newV0: {}'.format(self.equilibrium_volume[0], new_V0))
            integral_newacell0 = 1./(9*self.bulk_modulus*new_V0)*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen_from_dynmat[0, :, :])

            a_plushalf = new_acell0*(integral+1)
            
            #Test with rigid compliance tensor
            new_acell0_rigid = self.equilibrium_volume[1]*(1+1./(9*self.bulk_modulus_rigid*self.equilibrium_volume[0])*np.einsum('q,qv,qv->',self.wtq,self.omega[1,:,:],self.gruneisen_from_dynmat[0, :, :])*0.5)
            print('################ From Rigid compliance tensor, static_acell0 : {:>7.4f} bohr, new_acell0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4),
                new_acell0_rigid.round(4)))
            print('delta = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_acell0-self.equilibrium_volume[1], (new_acell0-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))

            #test_acell = self.equilibrium_volume[1]*(integral_plushalf + 1)
            #print('testacell={} bohr'.format(test_acell[0]))


            #integralvol = 1./(self.bulk_modulus*self.volume[1,0])*np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruvol)
            #vol = self.equilibrium_volume[0]*(integralvol+1)
            #avol = (vol*4)**(1./3)

            for t,T in enumerate(self.temperature):
                print('T={}K, a={} bohr, delta a = {} bohr'.format(T,a[t],a[t]-a[0]))

            a = np.expand_dims(a,axis=0)
            self.alpha_from_dynmat = np.expand_dims(alpha, axis=0)
            self.acell_plushalf_from_dynmat = np.expand_dims(a_plushalf,axis=0)
            #self.cv = np.einsum('q,qvt->t',self.wtq,cv)
            return a

        if self.symmetry == 'hexagonal':
            
            

            if self.verbose:
                ggg = open('{}_integrals.dat'.format(self.rootname),'w')   

            # First, get alpha(T)

            # Get Bose-Einstein factor and specific heat Cv
            self.bose = np.zeros((nqpt,nmode, self.ntemp))
            cv = np.zeros((nqpt,nmode,self.ntemp))

            for i,n in itt.product(range(nqpt),range(nmode)):
                self.bose[i,n,:] = self.get_bose(self.omega[1,i,n],self.temperature)
                cv[i,n,:] = self.get_specific_heat(self.omega[1,i,n],self.temperature)

            hwt = np.einsum('qv,qvt->qvt',self.omega[1,:,:],self.bose)
            # fix this properly later!!! 
            #cv[0,:3,:] = 0

            # Get alpha_a,c with compliance 
            alpha_a = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[0,:,:]) +
                self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[1,:,:]))/self.equilibrium_volume[0]
            alpha_c = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[0,:,:]) +
                self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[1,:,:]))/self.equilibrium_volume[0]

            self.alpha_a_from_dynmat = alpha_a
            self.alpha_c_from_dynmat = alpha_c
            alpha_a2 = ( (self.compliance_rigid[0,0]+self.compliance_rigid[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[0,:,:]) +
                self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[1,:,:]))/self.equilibrium_volume[0]
            alpha_c2 = ( 2*self.compliance_rigid[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[0,:,:]) +
                self.compliance_rigid[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gruneisen_from_dynmat[1,:,:]))/self.equilibrium_volume[0]

            #alpha_af = ( (self.compliance[0,0]+self.compliance[0,1])*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
            #    self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]
            #alpha_cf = ( 2*self.compliance[0,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[0,:,:]) +
            #    self.compliance[2,2]*np.einsum('q,qvt,qv->t',self.wtq,cv,self.gru2[1,:,:]))/self.equilibrium_volume[0]


            #self.cv = np.einsum('q,qvt->t',self.wtq,cv)

            # Then, get a(T) and c(T)
            integral_a = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen_from_dynmat[0,:,:])
            integral_c = np.einsum('q,qvt,qv->t',self.wtq,hwt,self.gruneisen_from_dynmat[1,:,:])
            integral_a0 = 0.5*np.einsum('q,qv,qv',self.wtq,self.omega[1,:,:],self.gruneisen_from_dynmat[0,:,:])
            integral_c0 = 0.5*np.einsum('q,qv,qv',self.wtq,self.omega[1,:,:],self.gruneisen_from_dynmat[1,:,:])


            aterm = (self.compliance[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a = self.equilibrium_volume[1]*(aterm/self.equilibrium_volume[0] + 1)
            daa = aterm/self.equilibrium_volume[0]

            cterm = 2*self.compliance[0,2]*integral_a + self.compliance[2,2]*integral_c
            c = self.equilibrium_volume[3]*(cterm/self.equilibrium_volume[0] + 1)
            dcc = cterm/self.equilibrium_volume[0]

            #daa_slope = np.polyfit(self.temperature[16:],daa[16:],1)
            #print('Delta a/a intersect: {:>8.5e}, new a0 = {} bohr'.format(daa_slope[1],-daa_slope[1]*self.equilibrium_volume[1]+self.equilibrium_volume[1]))
            #dcc_slope = np.polyfit(self.temperature[16:],dcc[16:],1)
            #print('Delta c/c intersect: {:>8.5e}, new c0 = {} bohr'.format(dcc_slope[1],-dcc_slope[1]*self.equilibrium_volume[3]+self.equilibrium_volume[3]))

            a2 = (self.compliance_rigid[0,0]+self.compliance[0,1])*integral_a + self.compliance[0,2]*integral_c
            a2 = self.equilibrium_volume[1]*(a2/self.equilibrium_volume[0] + 1)

            c2 = 2*self.compliance_rigid[0,2]*integral_a + self.compliance[2,2]*integral_c
            c2 = self.equilibrium_volume[3]*(c2/self.equilibrium_volume[0] + 1)

            # Test the da/a at T=0, using the 1/2 factor
            da0 = ((self.compliance[0,0]+self.compliance[0,1])*integral_a0 + self.compliance[0,2]*integral_c0)/self.equilibrium_volume[0]
            dc0 = (2*self.compliance[0,2]*integral_a0 + self.compliance[2,2]*integral_c0)/self.equilibrium_volume[0]
            da0_rigid = ((self.compliance_rigid[0,0]+self.compliance_rigid[0,1])*integral_a0 + self.compliance_rigid[0,2]*integral_c0)/self.equilibrium_volume[0]
            dc0_rigid = (2*self.compliance_rigid[0,2]*integral_a0 + self.compliance_rigid[2,2]*integral_c0)/self.equilibrium_volume[0]


            new_a0 = self.equilibrium_volume[1]*(1 + da0)
            new_c0 = self.equilibrium_volume[3]*(1 + dc0)
            aplushalf = new_a0*(aterm/self.equilibrium_volume[0] + 1)
            cplushalf = new_c0*(cterm/self.equilibrium_volume[0] + 1)


            new_a0_rigid = self.equilibrium_volume[1]*(1 + da0_rigid)
            new_c0_rigid = self.equilibrium_volume[3]*(1 + dc0_rigid)

            acell = np.array([a,c])
            self.acell_plushalf_from_dynmat = np.array([aplushalf,cplushalf])


            print('############### T=0')
            #new_acell0 = self.equilibrium_volume[1]*(1+1./(9*self.bulk_modulus*self.equilibrium_volume[0])*np.einsum('q,qv,qv->',self.wtq,self.omega[1,:,:],self.gruneisen)*0.5)
            print('################ static_acell :a0 = {:>7.4f} bohr, new_a0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4), new_a0.round(4)))
            print('delta a = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_a0-self.equilibrium_volume[1], (new_a0-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))

            print('################ static_acell :c0 = {:>7.4f} bohr, new_c0: {:>7.4f} (bohr)'.format(self.equilibrium_volume[3].round(4), new_c0.round(4)))
            print('delta c = {:>7.4f} bohr, delta c/c0 stat = {:>6.4f}%'.format(new_c0-self.equilibrium_volume[3], (new_c0-self.equilibrium_volume[3]).round(4)/self.equilibrium_volume[3].round(4)*100))
            # From rigid compliance tensor
            print('################ static_acell :a0 = {:>7.4f} bohr, new_a0_rigid: {:>7.4f} (bohr)'.format(self.equilibrium_volume[1].round(4), new_a0_rigid.round(4)))
            print('delta a = {:>7.4f} bohr, delta a/a0 stat = {:>6.4f}%'.format(new_a0_rigid-self.equilibrium_volume[1], (new_a0_rigid-self.equilibrium_volume[1]).round(4)/self.equilibrium_volume[1].round(4)*100))

            print('################ static_acell :c0 = {:>7.4f} bohr, new_c0_rigid: {:>7.4f} (bohr)'.format(self.equilibrium_volume[3].round(4), new_c0_rigid.round(4)))
            print('delta c = {:>7.4f} bohr, delta c/c0 stat = {:>6.4f}%'.format(new_c0_rigid-self.equilibrium_volume[3], (new_c0_rigid-self.equilibrium_volume[3]).round(4)/self.equilibrium_volume[3].round(4)*100))


#            self.acell2 = np.array([a2,c2])
            if self.verbose:
                'write details of compliance vs intergrals in <rootname>_integrals.dat file'''
                ggg.write('Terms entering the thermal expansion of a\n\n')
                ggg.write('{:15}  {:18}  {:18}  {:18}  {:18}\n'.format('Temperature (K)','compliance_a','integral_a','compliance_c','integral_c'))
                for t, T in enumerate(self.temperature):
                    ggg.write('{:>15.0f}  {:>.12e}  {:>.12e}  {:>.12e}  {:>.12e}\n'.format(T,self.compliance[0,0]+self.compliance[0,1],integral_a[t],self.compliance[0,2],integral_c[t]))
                ggg.write('\n\nTerms entering the thermal expansion of c\n\n')
                ggg.write('{:15}  {:18}  {:18}  {:18}  {:18}\n'.format('Temperature (K)','compliance_a','integral_a','compliance_c','integral_c'))
                for t, T in enumerate(self.temperature):
                    ggg.write('{:>15.0f}  {:>.12e}  {:>.12e}  {:>.12e}  {:>.12e}\n'.format(T,2*self.compliance[0,2],integral_a[t],self.compliance[2,2],integral_c[t]))

                ggg.close()

            print('From Gruneisen parameters, using dynamical matrix')
            print('da/a at T=0 = {}, da = {} bohr, a0 = {} bohr'.format(da0,da0*self.equilibrium_volume[1],da0*self.equilibrium_volume[1] + self.equilibrium_volume[1]))
            print('dc/c at T=0 = {}, dc = {} bohr, c0 = {} bohr'.format(dc0,dc0*self.equilibrium_volume[3],dc0*self.equilibrium_volume[3]+ self.equilibrium_volume[3]))

            for t,T in enumerate(self.temperature):
                print('T={}K, a={}, c={}'.format(T,a[t],c[t]))
                print('plushalf, a={}, c={}'.format(aplushalf[t],cplushalf[t]))



            self.alpha_from_dynmat = np.array([alpha_a, alpha_c])

            return acell


    def discrete_alpha_vs_reftemp(self,acell,ref_temp):

        # Get the thermal expansion coefficient vs room temperature lattice parameter
        find_t, = np.where(self.temperature==ref_temp)
        if len(find_t)==0:
            temp_index = np.argmax(np.where(self.temperature<ref_temp))
        else: 
            temp_index = find_t[0]

        nacell = acell.shape[0]
        alpha = np.zeros((nacell,self.ntemp))

        dx = self.temperature[temp_index+1] - self.temperature[temp_index] 

        ref_acell = acell[:,temp_index] + (acell[:,temp_index+1]-acell[:,temp_index])/dx*(ref_temp - self.temperature[temp_index])

        for t in range(self.ntemp):
            dt = self.temperature[t] - ref_temp

            if dt == 0.0:
                continue
            else:
                alpha[:,t] =  (acell[:,t] - ref_acell)/(ref_acell*dt)

        return alpha

    def get_alpha_vs_reftemp(self,acell, ref_temp):

        # get the thermal expansion coefficient from the lattice parameter,
        # using a non-zero temperature reference
        # and central finite difference derivative

        nacell = acell.shape[0]
        alpha = np.zeros((nacell,self.ntemp))

        find_t, = np.where(self.temperature==ref_temp)
        if len(find_t) == 0:
            temp_index = np.argmax(np.where(self.temperature<ref_temp))
        else: 
            temp_index = find_t[0]

        dt = self.temperature[temp_index+1] - self.temperature[temp_index] 
        ref_acell = acell[:,temp_index] + (acell[:,temp_index+1]-acell[:,temp_index])/dt*(ref_temp - self.temperature[temp_index])


        for t in range(self.ntemp):

            if t == 0:
                # Forward finite difference of the first one
                alpha[:,t] =  (acell[:,t+1] - acell[:,t])/(ref_acell*dt)

            elif t == self.ntemp-1:
                # Backwards finite difference for the last one
                alpha[:,t] =  (acell[:,t] - acell[:,t-1])/(ref_acell*dt)

            else:
                # Central finite difference
                # Should there be a warning if the step is not constant??? 
                # This should not happen from the definition of the temperature array.
                alpha[:,t] = (acell[:,t+1] - acell[:,t-1])/(2*ref_acell*dt)

        return alpha

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


    def get_gruneisen_from_dynmat(self,nqpt,nmode,nvol):

        # This computes the gruneisen parameters from the change in the dynamical matrix
        # like phonopy (and anaddb?)
            
        # for now, I reopen all files, but later on change the loop ordering (anyway, it should not do everything linearly, but rather use functions depending of input parameters

        if self.symmetry == 'cubic':

            gru = np.zeros((nqpt,nmode))
            gru_rot = np.zeros_like(gru)
            dplus =  2
            d0 = 1
            dminus = 0 # put this in a function
            # Get the linear cell dimension change, not volumic
            # gamma^a = gamma^V/3
            dV = self.volume[dplus,1] - self.volume[dminus,1]
            V0 = self.volume[d0,1]

            for i in range(nqpt):
                dD = np.zeros((nmode,nmode),dtype=complex)
                for vol in range(nvol):

                    # open the ddb file
                    ddb = DdbFile(self.ddb_flists[vol][i])

                    # Check if qpt is Gamma
                    is_gamma = ddb.is_gamma

                    dynmat = ddb.get_mass_scaled_dynmat_cart()

                    if vol==dplus:
                        dD = dD + dynmat
                    if vol==dminus:
                        dD = dD - dynmat
                    if vol==d0:
                        eigval, eigvect = np.linalg.eigh(dynmat)
                        omega = np.sqrt(np.abs(eigval)) * np.sign(eigval)

                        if is_gamma:
                            eigval[0] = 0.0
                            eigval[1] = 0.0
                            eigval[2] = 0.0

                        for ieig,eig in enumerate(eigval):
                            if eig < 0.0:
                                warnings.warn('Negative eigenvalue changed to 0')
                                eigval[ieig] = 0.0

                # end loop on volumes
           
                dD_at_q = []

                for v in range(nmode):

                    vect = eigvect[:,v]
                    dD_at_q.append(np.vdot(np.transpose(vect), np.dot(dD,vect)).real)   

                dD_at_q = np.array(dD_at_q)

                rot_eigvect, rot_dD_at_q = self.rotate_eigenvectors(eigval.real, eigvect, dD)

                for v in range(nmode):

                    if eigval[v].real < tol12:
                        gru[i,v] = 0
                        gru_rot[i, v] = 0
                    else:
                        gru[i,v] = -V0*dD_at_q[v]/(2*np.abs(eigval[v].real)*dV)
                        gru_rot[i,v] = -V0*rot_dD_at_q[v]/(2*np.abs(eigval[v].real)*dV)


            gru = np.expand_dims(gru, axis=0)
            self.gru_rot = np.expand_dims(gru_rot, axis=0)

        if self.symmetry == 'hexagonal':

            gru = np.zeros((2, nqpt, nmode))
            gru_rot = np.zeros_like(gru)
            dplus =  [2, 5]
            d0 = [1, 4]
            dminus = [0, 3] # put this in a function...
        
            dV = [self.volume[dplus[0],1] - self.volume[dminus[0],1], self.volume[dplus[1],3] - self.volume[dminus[1],3]]
            V0 = [self.volume[d0[0], 1], self.volume[d0[1], 3]]

            for a in range(2):

                for i in range(nqpt):
                    dD = np.zeros((nmode,nmode),dtype=complex)
                    for vol in range(3):

                        ivol = 3*a + vol
                        # open the ddb file
                        ddb = DdbFile(self.ddb_flists[ivol][i])

                        # Check if qpt is Gamma
                        is_gamma = ddb.is_gamma

                        dynmat = ddb.get_mass_scaled_dynmat_cart()

                        if ivol==dplus[a]:
                            dD = dD + dynmat
                        if ivol==dminus[a]:
                            dD = dD - dynmat
                        if ivol==d0[a]:
                            eigval, eigvect = np.linalg.eigh(dynmat)
                            omega = np.sqrt(np.abs(eigval)) * np.sign(eigval)

                            if is_gamma:
                                eigval[0] = 0.0
                                eigval[1] = 0.0
                                eigval[2] = 0.0

                            for ieig,eig in enumerate(eigval):
                                if eig < 0.0:
                                    warnings.warn('Negative eigenvalue changed to 0')
                                    eigval[ieig] = 0.0

                    # end loop on volumes
                    dD_at_q = []

                    rot_eigvect, rot_dD_at_q = self.rotate_eigenvectors(eigval.real, eigvect, dD)

                    for v in range(nmode):

                        vect = eigvect[:,v]
                        dD_at_q.append(np.vdot(np.transpose(vect), np.dot(dD,vect)).real)   

                    dD_at_q = np.array(dD_at_q)

                    for v in range(nmode):

                        if eigval[v].real < tol12:
                            gru[a,i,v] = 0
                            gru_rot[a, i, v] = 0
                        else:
                            if a == 0:
                                gru[a,i,v] = -V0[a]*dD_at_q[v]/(4*np.abs(eigval[v].real)*dV[a])
                                gru_rot[a,i,v] = -V0[a]*rot_dD_at_q[v]/(4*np.abs(eigval[v].real)*dV[a])
                            elif a == 1:
                                gru[a,i,v] = -V0[a]*dD_at_q[v]/(2*np.abs(eigval[v].real)*dV[a])
                                gru_rot[a, i,v] = -V0[a]*rot_dD_at_q[v]/(2*np.abs(eigval[v].real)*dV[a])

            self.gru_rot = gru_rot

        return gru

    def rotate_eigenvectors(self, eigval, eigvect, dD):
        rot_eigvect = np.zeros_like(eigvect)
        eigval_dD = np.zeros_like(eigval)
        for deg in self.degenerate_sets(eigval):
            dD_part = np.dot(np.transpose(np.conjugate(eigvect[:, deg])), np.dot(dD, eigvect[:, deg]))
            eigval_dD[deg], eigvect_dD = np.linalg.eigh(dD_part)
            rot_eigvect[:, deg] = np.dot(eigvect[:, deg], eigvect_dD) # why not vdot??
        return rot_eigvect, eigval_dD

    def degenerate_sets(self, freq, cutoff=1E-12):
        indices = []
        done = []
        for i in range(len(freq)):
            if i in done:
                continue
            else:
                f_set = [i]
                done.append(i)
            for j in range(i+1, len(freq)):
                if (np.abs(freq[f_set] - freq[j]) < cutoff).any():
                        f_set.append(j)
                        done.append(j)
            indices.append(f_set[:])

        #print('indices:',indices)
        return indices


    def write_nc(self):

        nc_outfile = 'OUT/{}_TE.nc'.format(self.rootname)

        #  First, write output in netCDF format
        create_directory(nc_outfile)

        with nc.Dataset(nc_outfile, 'w') as dts:

             #Define the type of calculation:
            dts.description = 'Gruneisen'

            dts.createDimension('number_of_temperatures', self.ntemp)
            dts.createDimension('number_of_lattice_parameters', len(self.distinct_acell))
            dts.createDimension('one',1)
            dts.createDimension('four', 4)
            dts.createDimension('number_of_qpoints', self.gruneisen.shape[1])
            dts.createDimension('number_of_modes', self.gruneisen.shape[2])
            dts.createDimension('number_of_volumes', np.shape(self.volume)[0])

            data = dts.createVariable('temperature','d', ('number_of_temperatures'))
            data[:] = self.temperature[:]
            data.units = 'Kelvin'

            data = dts.createVariable('acell_from_gruneisen','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_via_gruneisen[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('acell_from_gruneisen_from_dynmat','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_via_gruneisen_from_dynmat[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('acell_from_gruneisen_plushalf','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_plushalf[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('acell_from_gruneisen_plushalf_from_dynmat','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.acell_plushalf_from_dynmat[:,:]
            data.units = 'Bohr radius'

            data = dts.createVariable('c_over_a', 'd', ('number_of_temperatures'))
            if self.symmetry == 'hexagonal':
                data[:] = self.c_over_a[:]
            data.units = 'Unitless'

            data = dts.createVariable('volume','d', ('number_of_volumes','four'))
            data[:,:] = self.volume[:,:]
            data.units = 'bohr^3'

            data = dts.createVariable('equilibrium_volume','d', ('four'))
            data[:] = self.equilibrium_volume[:]
            data.units = 'bohr^3'


#            data = dts.createVariable('acell_from_helmholtz','d',('number_of_lattice_parameters','number_of_temperatures'))
#            data[:,:] = self.temperature_dependent_acell[:,:]
#            data.units = 'Bohr radius'
#
#            data = dts.createVariable('acell_from_gibbs','d',('number_of_lattice_parameters','number_of_temperatures'))
#            data[:,:] = self.fitg[:,:]
#            data.units = 'Bohr radius'

            data = dts.createVariable('alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('alpha_from_dynmat','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.alpha_from_dynmat[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('discrete_room_temp_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.discrete_room_temp_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('discrete_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.discrete_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('room_temp_alpha','d', ('number_of_lattice_parameters','number_of_temperatures'))
            data[:,:] = self.room_temp_alpha[:,:]
            data.units = 'K^-1'

            data = dts.createVariable('bulk_modulus_habo3','d',('one'))
            data[:] = self.bulkmodulus_from_elastic*cst.gpa_to_habo3

            data = dts.createVariable('specific_heat','d',('number_of_temperatures'))
            data[:] = self.cv
            data.units = 'Ha/K'

            data = dts.createVariable('eos_fit_parameters', 'd', ('four','number_of_temperatures'))
            if self.symmetry == 'hexagonal':
                data[:,:] = self.eos_fit_params[:,:]
                data.units = "V0,E0,B0,B0p"
                data.description = self.eos_type

            data = dts.createVariable('gruneisen_parameters', 'd', ('number_of_lattice_parameters', 'number_of_qpoints', 'number_of_modes'))
            data[:, :, :] = self.gruneisen[:, :, :]

            data = dts.createVariable('gruneisen_parameters_from_dynmat', 'd', ('number_of_lattice_parameters', 'number_of_qpoints', 'number_of_modes'))
            data[:, :, :] = self.gruneisen_from_dynmat[:, :, :]

            data = dts.createVariable('gruneisen_parameters_from_dynmat_rot', 'd', ('number_of_lattice_parameters', 'number_of_qpoints', 'number_of_modes'))
            data[:, :, :] = self.gru_rot[:, :, :]

            data = dts.createVariable('omega_equilibrium', 'd', ('number_of_qpoints', 'number_of_modes'))
            data[:, :] = self.omega[1, :, :]
            data.units = 'Hartree'



    def write_acell(self):
        # Then, write them in ascii file

        outfile = 'OUT/{}_acell_from_gruneisen.dat'.format(self.rootname)
        create_directory(outfile)

        with open(outfile, 'w') as f:

            f.write('Temperature dependent lattice parameters via Gruneisen parameters\n\n')

            if self.symmetry == 'cubic':

                f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.acell_via_gruneisen[0,t]))

                f.write('\n\nWith ZPR-latt using Gruneisens\n\n')
                f.write('{:12}      {:<12}\n'.format('Temperature','a (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.acell_plushalf[0,t]))


                f.close()


            if self.symmetry == 'hexagonal':

                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.acell_via_gruneisen[0,t],self.acell_via_gruneisen[1,t]))

                f.write('\n\nFrom Gruneisen with n+1/2\n\n')
                f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
                for t,T in enumerate(self.temperature):
                    f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.acell_plushalf[0,t],self.acell_plushalf[1,t]))


                f.close()

#         # Write also results from Free energy minimisation
#         # This makes no sense, as it cannot work for 3 input points only. It is more properly done using Gibbs class.
#         outfile = 'OUT/{}_acell_from_freeenergy.dat'.format(self.rootname)
# 
#         create_directory(outfile)
# 
#         with open(outfile, 'w') as f:
# 
#             f.write('Temperature dependent lattice parameters via Helmholtz free energy\n\n')
# 
#             if self.symmetry == 'cubic':
# 
# 
#                 f.write('{:12}    {:12}\n'.format('Temperature','a (bohr)'))
#                 for t,T in enumerate(self.temperature):
#                     f.write('{:>8.1f} K    {:>12.8f}\n'.format(T,self.temperature_dependent_acell[0,t]))
# 
#                 f.close()
# 
# 
#             if self.symmetry == 'hexagonal':
# 
#                 f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
#                 for t,T in enumerate(self.temperature):
#                     f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fit2d[0,t],self.fit2d[1,t]))
# 
#                 # Independent fit: fitg, 2D fit: fit2dg
#                 f.write('\n\nTemperature dependent lattice parameters via Gibbs free energy\n\n')
#                 f.write('{:12}      {:<12}    {:<12}\n'.format('Temperature','a (bohr)','c (bohr)'))
#                 for t,T in enumerate(self.temperature):
#                     f.write('{:>8.1f} K    {:>12.8f}    {:>12.8f}\n'.format(T,self.fitg[0,t],self.fitg[1,t]))
# 
# 
# 
#                 f.close()


class GruneisenPath(FreeEnergy):

    #Input files
    ddb_flists = None
    out_flists = None
    elastic_fname = None

    #Parameters
    wtq = [1.0]
    pressure = 0.0
    pressure_gpa= 0.0 

    verbose = False
    bulk_modulus = None


    def __init__(self,

        rootname,
        units,
        symmetry,

        ddb_flists = None,
        out_flists = None,
        elastic_fname = None,

        #wtq = [1.0],

        bulk_modulus = None,
        pressure = 0.0,
        pressure_gpa = 0.0,
        pressure_units = None,
        bulk_modulus_units = None,
        #manual_correction = False,
        verbose = False,

        #eos_type = 'Murnaghan',

        **kwargs):


        print('Computing Gruneisen parameters on qpath')
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

        super(GruneisenPath,self).__init__(rootname,units)
        #self.check_anaddb = check_anaddb
        #self.manual_correction = manual_correction
        self.verbose = verbose

        #self.temperature = temperature
        #self.ntemp = len(self.temperature) 
        
        self.pressure_units = pressure_units
        self.pressure = pressure

        if self.pressure_units == 'GPa':
            self.pressure_gpa = self.pressure
            self.pressure = self.pressure*cst.gpa_to_habo3
        else:
            self.pressure_gpa == self.pressure*cst.habo3_to_gpa

        print('External pressure is {} GPa'.format(self.pressure_gpa))

        if bulk_modulus:
            if bulk_modulus_units == 'GPa':
                self.bulk_modulus = bulk_modulus*cst.gpa_to_habo3
            elif bulk_modulus_units == 'HaBo3':
                self.bulk_modulus = bulk_modulus
            else:
                raise Exception('Bulk modulus units must be GPa or Ha/bohr^3')
################# This could be in a function??
        ##Define EOS type
        #self.eos_type = eos_type
        #if self.eos_type not in self.eos_list:
        #    raise Exception('EOS type must be one of the following: {}'.format(self.eos_list))

        # set parameter space dimensions
        '''why the hell does the shape sometimes work and sometimes not???'''
#        nvol, nqpt = np.shape(self.ddb_flists)
        nvol, nqpt = len(self.ddb_flists), len(self.ddb_flists[0])
        #self.free_energy = np.zeros((nvol,self.ntemp))
        #self.gibbs_free_energy = np.zeros((nvol,self.ntemp))

        self.qred = np.zeros((nqpt,3))

        self.volume = np.empty((nvol,4)) # 1st index = data index, 2nd index : total cell volume, (a1,a2,a3)

        # Check that all qpt lists have the same lenght, and that it is equal to the number of wtq
        for v in range(nvol):
            if len(ddb_flists[v][:]) != len(ddb_flists[0][:]):
                raise Exception('all ddb lists must have the same number of files.\n List index {} has {} entries while list index 0 has {}.'.format(v,len(ddb_flists[v][:]),len(ddb_flists[0][:])))

        #self.set_weights(wtq)

        # Loop on all volumes
        for v in range(nvol):

            print('\n\nReading data from {}'.format(out_flists[v]))
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
                self.eigvect = np.zeros((nvol, nqpt, nmode, nmode), dtype=complex)

            # for each qpt:
            # Redundent if I do it in the function...
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
                        ### I should maybe implement the proper ASR corrections, like in anaddb and phonopy...

                # Store frequencies for Gruneisen parameters
                self.omega[v,i,:] = ddb.omega

        ### End of loop on volumes, all data has been read ###

        # Read elastic compliance from file
        if self.elastic_fname:
            elastic = ElasticFile(self.elastic_fname)
            self.compliance = elastic.compliance_relaxed
            self.compliance_rigid = elastic.compliance_clamped

            bmod = self.get_bulkmodulus_from_elastic(elastic.stiffness_relaxed)
            bmod2 = self.get_bulkmodulus_from_elastic(elastic.stiffness_clamped)
            self.bulkmodulus_from_elastic = bmod
            ##FIX ME: what to do i we want to use th inputted (experimental) bulk modulus?)
            if self.symmetry == 'cubic':
                if self.bulk_modulus is None:
                    print('Using bulk modulus from elastic constants.')
                    self.bulk_modulus = self.bulkmodulus_from_elastic*cst.gpa_to_habo3
                    self.bulk_modulus_rigid = bmod2*cst.gpa_to_habo3
                else:
                    print('Using bulk modulus from input file.')

            if self.verbose:
                print('Bulk modulus from elastic constants = {:>7.3f} GPa'.format(bmod))
                print('Bulk modulus from elastic constants (clamped) = {:>7.3f} GPa'.format(bmod2))

                print('Elastic constants:')
                print('c11 = {}, c33 = {}, c12 = {}, c13 = {} GPa'.format(elastic.stiffness_relaxed[0,0],elastic.stiffness_relaxed[2,2],elastic.stiffness_relaxed[0,1],elastic.stiffness_relaxed[0,2]))
                print('Clamped:')
                print('c11 = {}, c33 = {}, c12 = {}, c13 = {} GPa'.format(elastic.stiffness_clamped[0,0],elastic.stiffness_clamped[2,2],elastic.stiffness_clamped[0,1],elastic.stiffness_clamped[0,2]))

                print('Compliance constants:')
                print('s11 = {}, s33 = {}, s12 = {}, s13 = {} GPa^-1'.format(elastic.compliance_relaxed[0,0],elastic.compliance_relaxed[2,2],elastic.compliance_relaxed[0,1],elastic.compliance_relaxed[0,2]))
                print('Clamped:')
                print('s11 = {}, s33 = {}, s12 = {}, s13 = {} GPa^-1'.format(elastic.compliance_clamped[0,0],elastic.compliance_clamped[2,2],elastic.compliance_clamped[0,1],elastic.compliance_clamped[0,2]))



        # Get Gruneisen parameters, according to crystal symmetry

        ### Add a check for homogenious acell increase (for central finite difference)
        self.equilibrium_volume = self.volume[1,:]

        #self.gruneisen = self.get_gruneisen(nqpt,nmode,nvol)
        self.gruneisen_from_dynmat = self.get_gruneisen_from_dynmat(nqpt,nmode,nvol)


    def get_gruneisen_from_dynmat(self,nqpt,nmode,nvol):

        # for now, I reopen all files, but later on change the loop ordering (anyway, it should not do everything linearly, but rather use functions depending of input parameters
        if self.symmetry == 'cubic':

            gru = np.zeros((nqpt,nmode))
            gru_rot = np.zeros_like(gru)
            dplus =  2
            d0 = 1
            dminus = 0 # put this in a function
            # Get the linear cell dimension change, not volumic
            # gamma^a = gamma^V/3
            dV = self.volume[dplus,1] - self.volume[dminus,1]
            V0 = self.volume[d0,1]

            previous_eigvect = None
            previous_eigvect_rot = None
            band_order = np.arange(nmode)
            band_order2 = np.arange(nmode)
            band_order_rot = np.arange(nmode)
            band_order_rot2 = np.arange(nmode)
            omega_equilibrium = np.empty((nqpt, nmode))
            omega_equilibrium_rot = np.empty((nqpt, nmode))

            for i in range(nqpt):
            #for i in range(5):
                #print('for qpt {}'.format(i+1))
                dD = np.zeros((nmode,nmode),dtype=complex)
                for vol in range(nvol):

                    # open the ddb file
                    ddb = DdbFile(self.ddb_flists[vol][i])

                    # Check if qpt is Gamma
                    is_gamma = ddb.is_gamma

                    dynmat = ddb.get_mass_scaled_dynmat_cart()

                    if vol==dplus:
                        dD = dD + dynmat
                    if vol==dminus:
                        dD = dD - dynmat
                    if vol==d0:
                        eigval, eigvect = np.linalg.eigh(dynmat)

                        if is_gamma:
                            eigval[0] = 0.0
                            eigval[1] = 0.0
                            eigval[2] = 0.0

                        for ieig,eig in enumerate(eigval):
                            if eig < 0.0:
                                warnings.warn('Negative eigenvalue changed to 0')
                                eigval[ieig] = 0.0

                        omega_equilibrium[i, :] = np.sqrt(np.abs(eigval)) * np.sign(eigval)
                        omega_equilibrium_rot[i, :] = np.sqrt(np.abs(eigval)) * np.sign(eigval)

                # end loop on volumes
                dD_at_q = []

                # Look up how to follow the modes, by computing the projections
                # this should be done before apprnding dD_at_q?
                # for a given qpt, compare eigvect to prev_eigvect
                # Reorder the elements of dD_at_q (and omega?) according to the highest projection coefficient
                # What to do if the path is discontinuous? add a 'isclose' to compare the qpoints?
                for v in range(nmode):
                    vect = eigvect[:,v]
                    dD_at_q.append(np.vdot(np.transpose(vect), np.dot(dD,vect)).real)   

                dD_at_q = np.array(dD_at_q)

                # Estimate band connexion from projections
                #print('no rotation')
                if previous_eigvect is not None:
                    band_order, band_order2 = self.estimate_band_connection(previous_eigvect, eigvect, band_order, band_order2)
                    #print(band_order)
                #print('with rotation')
                rot_eigvect, rot_dD_at_q = self.rotate_eigenvectors(eigval.real, eigvect, dD)
                if previous_eigvect_rot is not None:
                    band_order_rot, band_order_rot2 = self.estimate_band_connection(previous_eigvect_rot, rot_eigvect, band_order_rot, band_order_rot2)
                    #print(band_order_rot)
                previous_eigvect = eigvect
                previous_eigvect_rot = rot_eigvect
                eigval = eigval[band_order]
                eigval_rot = eigval[band_order_rot]
                dD_at_q = dD_at_q[band_order]
                rot_dD_at_q = rot_dD_at_q[band_order_rot]
                omega_equilibrium[i, :] = omega_equilibrium[i, band_order]
                omega_equilibrium_rot[i, :] = omega_equilibrium_rot[i, band_order_rot]

                # test the rotate_eigvect...
                # define omega from eigvals
                for v in range(nmode):

                    if eigval[v].real < tol12:
                        gru[i,v] = 0
                    else:
                        gru[i,v] = -V0*dD_at_q[v]/(2*np.abs(eigval[v].real)*dV)

                    if eigval_rot[v].real < tol12:
                        gru_rot[i,v] = 0
                    else:
                        gru_rot[i,v] = -V0*rot_dD_at_q[v]/(2*np.abs(eigval_rot[v].real)*dV)

            gru = np.expand_dims(gru, axis=0)
            self.gru_rot = np.expand_dims(gru_rot, axis=0)
            self.omega_equilibrium = omega_equilibrium
            self.omega_equilibrium_rot = omega_equilibrium_rot


        if self.symmetry == 'hexagonal':

            gru = np.zeros((2, nqpt, nmode))
            dplus =  [2, 5]
            d0 = [1, 4]
            dminus = [0, 3] # put this in a function...
        
            dV = [self.volume[dplus[0],1] - self.volume[dminus[0],1], self.volume[dplus[1],3] - self.volume[dminus[1],3]]
            V0 = [self.volume[d0[0], 1], self.volume[d0[1], 3]]

            for a in range(2):

                for i in range(nqpt):
                    dD = np.zeros((nmode,nmode),dtype=complex)
                    for vol in range(3):

                        ivol = 3*a + vol
                        # open the ddb file
                        ddb = DdbFile(self.ddb_flists[ivol][i])

                        # Check if qpt is Gamma
                        is_gamma = ddb.is_gamma

                        dynmat = ddb.get_mass_scaled_dynmat_cart()

                        if ivol==dplus[a]:
                            dD = dD + dynmat
                        if ivol==dminus[a]:
                            dD = dD - dynmat
                        if ivol==d0[a]:
                            eigval, eigvect = np.linalg.eigh(dynmat)
                            omega = np.sqrt(np.abs(eigval)) * np.sign(eigval)

                            if is_gamma:
                                eigval[0] = 0.0
                                eigval[1] = 0.0
                                eigval[2] = 0.0

                            for ieig,eig in enumerate(eigval):
                                if eig < 0.0:
                                    warnings.warn('Negative eigenvalue changed to 0')
                                    eigval[ieig] = 0.0

                    # end loop on volumes
                    dD_at_q = []

                    for v in range(nmode):

                        vect = eigvect[:,v]
                        dD_at_q.append(np.vdot(np.transpose(vect), np.dot(dD,vect)).real)   

                    dD_at_q = np.array(dD_at_q)

                    for v in range(nmode):

                        if eigval[v].real < tol12:
                            gru[a,i,v] = 0
                        else:
                            if a == 0:
                                gru[a,i,v] = -V0[a]*dD_at_q[v]/(4*np.abs(eigval[v].real)*dV[a])
                            elif a == 1:
                                gru[a,i,v] = -V0[a]*dD_at_q[v]/(2*np.abs(eigval[v].real)*dV[a])
        
        return gru

    def estimate_band_connection(self, previous_eigvect, current_eigvect, previous_order, previous_order2):

        dim = len(previous_order)
        band_order = []
        connexion_order = []
        overlap_matrix = np.abs(np.dot(np.transpose(np.conjugate(previous_eigvect)), current_eigvect))

        for v in range(dim):
            #Overlap of previous_eigvect[:, v] with current eigenvectors
            overlaps = np.abs(np.dot(np.transpose(np.conjugate(previous_eigvect[:, v])), current_eigvect))
            #print(overlaps)
            # check also if the projections are more 'clear' (less mixed) if I add the 
            # rotate_eigenvectors function (write it as is, check what it does)
            # check also how to add the LoTo contribution to the frequencies 
            for j in reversed(range(dim)):
                '''why reversed?!?'''
                if j in band_order:
                    overlaps[j] = 0

            band_order.append(np.argmax(overlaps))

        for olaps in overlap_matrix:
            maxval = 0
            for j in reversed(range(dim)):
                val = olaps[j]
                if j in connexion_order:
                    continue
                if val > maxval:
                    maxval = val
                    maxindex = j
            connexion_order.append(maxindex)


        current_order = [band_order[x] for x in previous_order]
        current_order2 = [connexion_order[x] for x in previous_order2]
        if current_order != current_order2:
            print('current_order', current_order)
            print('phonopy algo', current_order2)

        return current_order, current_order2

    def rotate_eigenvectors(self, eigval, eigvect, dD):
        rot_eigvect = np.zeros_like(eigvect)
        eigval_dD = np.zeros_like(eigval)
        for deg in self.degenerate_sets(eigval):
            dD_part = np.dot(np.transpose(np.conjugate(eigvect[:, deg])), np.dot(dD, eigvect[:, deg]))
            eigval_dD[deg], eigvect_dD = np.linalg.eigh(dD_part)
            rot_eigvect[:, deg] = np.dot(eigvect[:, deg], eigvect_dD) # why not vdot??
        return rot_eigvect, eigval_dD

    def degenerate_sets(self, freq, cutoff=1E-12):
        indices = []
        done = []
        for i in range(len(freq)):
            if i in done:
                continue
            else:
                f_set = [i]
                done.append(i)
            for j in range(i+1, len(freq)):
                if (np.abs(freq[f_set] - freq[j]) < cutoff).any():
                        f_set.append(j)
                        done.append(j)
            indices.append(f_set[:])

        #print('indices:',indices)
        return indices

    def write_nc(self):

        nc_outfile = 'OUT/{}_GRUNPATH.nc'.format(self.rootname)

        #  First, write output in netCDF format
        create_directory(nc_outfile)

        with nc.Dataset(nc_outfile, 'w') as dts:

             #Define the type of calculation:
            dts.description = 'Gruneisen Path'

            dts.createDimension('number_of_lattice_parameters', len(self.distinct_acell))
            dts.createDimension('one', 1)
            dts.createDimension('three', 3)
            dts.createDimension('four', 4)
            dts.createDimension('number_of_qpoints', self.gruneisen_from_dynmat.shape[1])
            dts.createDimension('number_of_modes', self.gruneisen_from_dynmat.shape[2])
            dts.createDimension('number_of_volumes', np.shape(self.volume)[0])

            data = dts.createVariable('volume','d', ('number_of_volumes','four'))
            data[:,:] = self.volume[:,:]
            data.units = 'bohr^3'

            data = dts.createVariable('equilibrium_volume','d', ('four'))
            data[:] = self.equilibrium_volume[:]
            data.units = 'bohr^3'

            data = dts.createVariable('bulk_modulus_habo3','d',('one'))
            data[:] = self.bulkmodulus_from_elastic*cst.gpa_to_habo3

            data = dts.createVariable('gruneisen_parameters_from_dynmat', 'd', ('number_of_lattice_parameters', 'number_of_qpoints', 'number_of_modes'))
            data[:, :, :] = self.gruneisen_from_dynmat[:, :, :]

            data = dts.createVariable('omega_equilibrium', 'd', ('number_of_qpoints', 'number_of_modes'))
            data[:, :] = self.omega_equilibrium[ :, :]
            data.units = 'Hartree'

            data = dts.createVariable('reduced_coordinates_of_qpoints', 'd', ('number_of_qpoints', 'three'))
            data[:,:] = self.qred[:, :]

            data = dts.createVariable('gruneisen_parameters_from_dynmat_rot', 'd', ('number_of_lattice_parameters', 'number_of_qpoints', 'number_of_modes'))
            data[:, :, :] = self.gru_rot[:, :, :]

            data = dts.createVariable('omega_equilibrium_rot', 'd', ('number_of_qpoints', 'number_of_modes'))
            data[:, :] = self.omega_equilibrium_rot[ :, :]
            data.units = 'Hartree'

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
        if self.dedp:
            if not self.gap_fname:
                raise Exception('Must provide a netCDF file containing gap energies. Please use netcdf_gap.py') 

        
        self.nfile = len(self.etotal_flist)

        self.etotal = np.empty((self.nfile))
        self.gap_energy = np.empty((self.nfile))
        self.volume = np.empty((self.nfile))

        if self.dedp:
            gap = GapFile(self.gap_fname)
            self.gap_energy = gap.gap_energy
            if len(self.gap_energy) != self.nfile:
                raise Exception('{} contains {} gap values, while there are {} files in etotal_flist. '.format(self.gap_fname, len(self.gap_energy),self.nfile))

        for n, fname in enumerate(self.etotal_flist):

            gs = GsrFile(fname)
            self.etotal[n] = gs.etotal
            self.volume[n] = gs.volume


        self.bulk_modulus, self.bulk_modulus_derivative, self.equilibrium_volume, self.equilibrium_energy =  self.get_bulk_modulus()
        self.effective_pressure = self.get_effective_pressure()

        if self.dedp:
    
            self.pressure, self.dedp_fit, self.dedp = self.get_dedp()


    def get_bulk_modulus(self):

        
        if self.initial_params:
            p0 = self.initial_params
            popt,pcov = curve_fit(eos.murnaghan_EV, self.volume, self.etotal, p0)  

        else:
            p0 = [self.volume[-1],self.etotal[-1],130.,4.0]  # Guess for initial parameters
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

        print('Bulk modulus from static EOS : {} GPa'.format(popt[2]/cst.gpa_to_habo3))
        
        return popt[2], popt[3], popt[0], popt[1]


    def get_effective_pressure(self):

        #Extract effective pressure at a given volume from Murnaghan EOS
        pdata = eos.murnaghan_PV(self.volume,self.equilibrium_volume, self.bulk_modulus, self.bulk_modulus_derivative)

        return pdata


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

            data = dts.createVariable('effective_pressure_from_EOS','d',('number_of_points'))
            data[:] = self.effective_pressure*cst.habo3_to_gpa
            data.units = 'GPa'

            data = dts.createVariable('volume','d',('number_of_points'))
            data[:] = self.volume
            data.units = 'bohr^3'

            data = dts.createVariable('gap_energy','d', ('number_of_points'))
            data[:] = self.gap_energy
            data.units = 'hartree'

            data = dts.createVariable('dE_dP','d',('one'))
            if self.dedp:
                data[:] = self.dedp*cst.ha_to_ev*1000/cst.habo3_to_gpa
                data.units = 'meV/GPa'

            data = dts.createVariable('dedp_fit','d',('two'))
            if self.dedp:
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
        thermo_flist = None,
        elastic_fname = None,
        etotal_flist = None,
        gap_fname = None,
        rootname = 'te2.out',

        #Parameters
        wtq = [1.0],
        temperature = None,

        #Options
        gibbs = False, # Default value is Helmoltz free energy, at P=0 (or, at constant P)
        gibbs_anaddb = False,
        check_anaddb = False,
        units = 'eV',
        symmetry = None,
        bulk_modulus = None,
        bulk_modulus_units = None,
        pressure = 0.0,
        pressure_units = None,
        eos_type = 'Murnaghan',
        use_axial_eos = False,
        tmin_slope = 500,

        expansion = True,
        gruneisen = False,
        gruneisen_path = False,
        bulkmodulus = False,
        dedp = False,
        initial_params = None,
        static_plot = False,
        manual_correction = False,
        equilibrium_index = None,
        verbose = False,

        **kwargs):

    # Choose appropriate type of free energy 
    # FIX ME this should just be called Static, or something like that...
    if bulkmodulus:
        
        static_calc = Static(
                    rootname = rootname,
                    symmetry = symmetry,
                    
                    etotal_flist = etotal_flist,
                    gap_fname = gap_fname,
            
                    dedp = dedp,
                    initial_params = initial_params,

                    static_plot = static_plot,
                    eos_type = eos_type,

                    **kwargs)

        ## write static output files
        static_calc.write_output()
        static_calc.write_netcdf()

    # Compute Gruneisen parameters on a qpath
    if gruneisen_path:
         calc = GruneisenPath(
                out_flists = out_flists, 
                ddb_flists = ddb_flists,
    
                rootname = rootname,
                symmetry = symmetry,
    
                #wtq = wtq,
                #temperature = temperature,
                units = units,
                #check_anaddb = check_anaddb,
                elastic_fname = elastic_fname,
                pressure = pressure,
                pressure_units = pressure_units,
    
                bulk_modulus = bulk_modulus,
                bulk_modulus_units = bulk_modulus_units,

                #manual_correction = manual_correction,
                verbose = verbose,

                #eos_type = eos_type,

                **kwargs)

         calc.write_nc()

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
    
                    equilibrium_index = equilibrium_index,
                    verbose = verbose,
                    pressure = pressure,
                    pressure_units = pressure_units,
                    eos_type = eos_type,
                    use_axial_eos = use_axial_eos,
                    manual_correction = manual_correction, # should be removed...
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

                    manual_correction = manual_correction,
                    verbose = verbose,

                    eos_type = eos_type,

                    **kwargs)


        else:

            if gibbs_anaddb:

                calc = Gibbs_from_anaddb(
                    out_flists = out_flists, 
                    thermo_flist = thermo_flist,
        
                    rootname = rootname,
                    symmetry = symmetry,
        
                    wtq = wtq,
                    temperature = temperature,
                    units = units,
## FIX ME : IF THERE WAS DATA FOR BULK MODULUS, USE IT!! OR, SIMPLY COMPUTE IT FROM DDB VOLUME DATA...                    
                    bulk_modulus = bulk_modulus,
                    bulk_modulus_units = bulk_modulus_units,
    
                    equilibrium_index = equilibrium_index,
                    verbose = verbose,
                    pressure = pressure,
                    pressure_units = pressure_units,
                    eos_type = eos_type,
                    use_axial_eos = use_axial_eos,
        
                        **kwargs)
            else:
                raise Exception("For thermal expansion, you should choose between minimization of the Helmoltz/Gibbs free energy, or Grüneisen parameters. What are you trying to compute?")


        # Write output files
        calc.write_nc()
        calc.write_acell()

    return
    


###########################


