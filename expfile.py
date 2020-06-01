import netCDF4 as nc

__author__ = "Veronique Brousseau-Couture"

from ElectronPhononCoupling import EpcFile 

class EXPfile(EpcFile):

    def __init__(self, *args, **kwargs):

        super(EXPfile, self).__init__(*args,**kwargs)

        self.xaxis = None
        self.yaxis = None
        self.ndata = None

    def read_nc(self, fname=None):

        fname = fname if fname else self.fname
        super(EXPfile, self).read_nc(fname)

        with nc.Dataset(fname,'r') as ncdata:

            self.xaxis = ncdata.variables['ax1'][:]
            self.yaxis = ncdata.variables['ax2'][:]
            self.ndata = len(self.xaxis)
            self.xaxis_units = ncdata.variables['ax1'].getncattr('units')
            self.yaxis_units = ncdata.variables['ax2'].getncattr('units')


