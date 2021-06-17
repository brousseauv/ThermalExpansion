from ElectronPhononCoupling import EpcFile
import netCDF4 as nc
from numpy import amin, amax

class EIGfile(EpcFile):

    def __init__(self, *args, **kwargs):

        super(EIGfile, self).__init__(*args, **kwargs)
        self.eig0 = None
        self.kpoints = None

    # Open the EIG.nc file and read it
    def read_nc(self, fname=None):

        fname = fname if fname else self.fname
        super(EIGfile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as ncdata:
            self.eig0 = ncdata.variables['Eigenvalues'][:, :, :]
            self.kpoints = ncdata.variables['Kptns'][:, :]

    @property
    def nsppol(self):
        return self.eig0.shape[0] if self.eig0 is not None else None

    @property
    def nkpt(self):
        return self.eig0.shape[1] if self.eig0 is not None else None

    @property
    def max_band(self):
        return self.eig0.shape[2] if self.eig0 is not None else None

    def get_gap_energy(self, valence):

        return amin(self.eig0[0, :, valence]) - amax(self.eig0[0, :, valence-1])
