from __future__ import print_function
__author__ = 'brousseauv'

import netCDF4 as nc
from epcfile import EpcFile
from constants import ha_to_ev, bohr_to_ang, habo3_to_gpa

"""
Classes for *TE.nc output file, produced with ThemalExpansion module
"""


class FreeEnergyFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the .nc file and read it
        fname = fname if fname else self.fname

        super(FreeEnergyFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as dts:

            self.temperature = dts.variables['temperature'][:]
            self.temp_units = dts.variables['temperature'].getncattr('units')

            self.acell = dts.variables['acell_from_gibbs'][:, :] * bohr_to_ang
            self.free_energy = dts.variables['free_energy'][:, :]  # * ha_to_ev  # in eV
            self.static_energy = dts.variables['static_energy'][:, :]  # * ha_to_ev  # in eV
            self.phonon_free_energy = dts.variables['phonon_free_energy'][:, :]  # * ha_to_ev  # in eV
            self.volume = dts.variables['volume'][:, :]  # For now, leave in Bohr^3
            self.equilibrium_volume = dts.variables['equilibrium_volume'][:]
            self.equilibrium_volume[0] = self.equilibrium_volume[0] * bohr_to_ang**3
            self.equilibrium_volume[1:] = self.equilibrium_volume[1:] * bohr_to_ang
            self.fit_parameters = dts.variables['fit_parameters'][:, :]
            self.fit_parameters_list = dts.variables['fit_parameters'].getncattr('units')
            self.fit_function = dts.variables['fit_parameters'].getncattr('description')
            self.eos_fit = dts.variables['volumic_fit_parameters'][:, :]  # For now, in Ha and Ha/bohr^3
            self.bulk_modulus_from_eos = self.eos_fit[2, :] * habo3_to_gpa
            self.specific_heat = dts.variables['specific_heat'][:] * ha_to_ev  # Should be in eV/K

            # For cubics, eos fit is FE fit
            if self.eos_fit.mask.all():
                self.eos_fit = self.fit_parameters  # E0, V0, B0, B0'

            self.alpha = dts.variables['alpha'][:, :]
            self.room_temp_alpha = dts.variables['room_temp_alpha'][:, :]

    @property
    def ntemp(self):
        return len(self.temperature)

    @property
    def nacell(self):
        return self.acell.shape[0]


class GruneisenFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the .nc file and read it
        fname = fname if fname else self.fname

        super(GruneisenFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as dts:

            self.temperature = dts.variables['temperature'][:]
            self.temp_units = dts.variables['temperature'].getncattr('units')

#            self.acell = dts.variables['acell_from_gruneisen'][:, :] * bohr_to_ang
#            self.acell_zpm = dts.variables['acell_from_gruneisen_plushalf'][:, :] * bohr_to_ang
            self.acell = dts.variables['acell_from_gruneisen_plushalf'][:, :] * bohr_to_ang

#            self.free_energy = dts.variables['free_energy'][:] * ha_to_ev  # in eV
#            self.fit_parameters = dts.variables['fit_parameters'][:, :]
#            self.fit_parameters_list = dts.variables['fit_parameters'].getncattr('units')
#            self.fit_function = dts.variables['fit_parameters'].getncattr('description')
#            self.eos_fit = dts.variables['volumic_fit_parameters'][:, :]  # For now, in Ha and Ha/bohr^3

#            # For cubics, eos fit is FE fit
#            if self.eos_fit.mask.all():
#                self.eos_fit = self.fit_parameters  # E0, V0, B0, B0'

            self.alpha = dts.variables['alpha'][:, :]
            self.room_temp_alpha = dts.variables['room_temp_alpha'][:, :]
            self.specific_heat = dts.variables['specific_heat'][:] * ha_to_ev  # in eV/K
            self.bulk_modulus_from_elastic = dts.variables['bulk_modulus_habo3'][:] * habo3_to_gpa
            self.volume = dts.variables['volume']  # For now, leave in Bohr^3
            self.equilibrium_volume = dts.variables['equilibrium_volume'][:]
            self.equilibrium_volume[0] = self.equilibrium_volume[0] * bohr_to_ang**3
            self.equilibrium_volume[1:] = self.equilibrium_volume[1:] * bohr_to_ang
            self.gruneisen = dts.variables['gruneisen_parameters'][:, :, :]
            self.gruneisen_from_dynmat = dts.variables['gruneisen_parameters_from_dynmat'][:, :, :]
            self.gruneisen_from_dynmat_rot = dts.variables['gruneisen_parameters_from_dynmat_rot'][:, :, :]
            self.omega = dts.variables['omega_equilibrium'][:, :] * ha_to_ev * 1E3  # in meV

    @property
    def ntemp(self):
        return len(self.temperature)

    @property
    def nacell(self):
        return self.acell.shape[0]


class GruneisenPathFile(EpcFile):

    def read_nc(self, fname=None):

        # Open the .nc file and read it
        fname = fname if fname else self.fname

        super(GruneisenPathFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as dts:

            self.bulk_modulus_from_elastic = dts.variables['bulk_modulus_habo3'][:] * habo3_to_gpa
            self.volume = dts.variables['volume']  # For now, leave in Bohr^3
            self.equilibrium_volume = dts.variables['equilibrium_volume'][:]
            self.equilibrium_volume[0] = self.equilibrium_volume[0] * bohr_to_ang**3
            self.equilibrium_volume[1:] = self.equilibrium_volume[1:] * bohr_to_ang
            self.gruneisen_from_dynmat = dts.variables['gruneisen_parameters_from_dynmat'][:, :, :]
            # self.gruneisen_from_dynmat_rot = dts.variables['gruneisen_parameters_from_dynmat_rot'][:, :, :]
            self.omega = dts.variables['omega_equilibrium'][:, :] * ha_to_ev * 1E3  # in meV
            # self.omega_rot = dts.variables['omega_equilibrium_rot'][:, :] * ha_to_ev * 1E3  # in meV
            self.qred = dts.variables['reduced_coordinates_of_qpoints'][:, :]

    @property
    def nacell(self):
        return self.gruneisen_from_dynmat.shape[0]

    @property
    def nqpt(self):
        return self.gruneisen_from_dynmat.shape[1]

    @property
    def nmode(self):
        return self.gruneisen_from_dynmat.shape[2]


def set_file_class(fname):

    with nc.Dataset(fname, 'r') as dts:
        calc_type = dts.getncattr('description')

    print('Reading {}...   is {}File'.format(fname, calc_type))
    if calc_type == 'FreeEnergy':
        data = FreeEnergyFile(fname)
        data.read_nc()

    elif calc_type == 'Gruneisen':
        data = GruneisenFile(fname)
        data.read_nc()

    return data
