__author__ = "brousseauv"

""" Base class and subclasses for ThermalExpansion output *_TE.nc
    plotting, with possible comparison with experiment.
"""

from .plotter import Plotter
from .tefile import set_file_class
import eos as eos
from expfile import EXPfile
from constants import bohr_to_ang
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np

# import eos as eos
# import lmfit as lmfit

rc('text', usetex=True)
rc('font', family='sans-serif', weight='bold')


class TEplot(Plotter):

    """
    Main class for plotting outputs from *_TE.nc file.

    Parameters:
    -----------

    ax: a matplotlib.axes.AxesSubplot instance

    """

    def __init__(self, ax=None, **kwargs):
        self._init_figure(ax=ax, **kwargs)

    def set_labels(self, label_size):

        self.ax.set_xlabel(self.xlabel, fontsize=label_size)
        self.ax.set_ylabel(self.ylabel, fontsize=label_size)

    def int_formatter(self, x, pos):
        return '%i' % x

    def float_formatter(self, x, pos):
        return '%5.2f' % x

    def float4_formatter(self, x, pos):
        return '%5.4f' % x

    def exp_formatter(self, x, pos):
        return '%5.2e' % x

    def plot_exp_data(self, exp_data, marker_list=None, color_list=None, **kwargs):

        for i, ncfile in enumerate(exp_data):
            self.check_units(ncfile)

            if marker_list:
                if color_list:
                    self.ax.plot(ncfile.xaxis, ncfile.yaxis, marker=marker_list[i], color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.xaxis, ncfile.yaxis, marker=marker_list[i], **kwargs)
            else:
                if color_list:
                    self.ax.plot(ncfile.xaxis, ncfile.yaxis, marker='x', color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.xaxis, ncfile.yaxis, marker='x', **kwargs)


class AcellPlot(TEplot):
    """
    Plots temperature dependent lattice parameters
    """

    def __init__(self, ax=None, field='acell', **kwargs):
        self.field = field
        super(AcellPlot, self).__init__(ax=ax)

    def plot_te(self, te_data, marker_list=None, color_list=None, **kwargs):

        if self.field == 'acell' or self.field == 'acell_a':
            a = 0
        elif self.field == 'acell_c':
            a = -1

        for i, ncfile in enumerate(te_data):
            if marker_list:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.acell[a, :], marker=marker_list[i],
                                 color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.acell[a, :], marker=marker_list[i], **kwargs)
            else:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.acell[a, :], color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.acell[a, :], **kwargs)

    def check_units(self, ncfile):

        if ncfile.yaxis_units.find('ang') != -1 or ncfile.yaxis_units.find('Ang') != -1:
            pass
        elif ncfile.yaxis_units.find('bohr') != -1 or ncfile.yaxis_units.find('Bohr') != -1:
            ncfile.yaxis = ncfile.yaxis * bohr_to_ang
        else:
            raise Exception('Units from EXPfile should contain either "ang/Ang" or "bohr/Bohr"')

    @property
    def xlabel(self):
        return r'Temperature (K)'

    @property
    def ylabel(self):
        return r'Lattice parameter (\AA)'

    def format_ticks(self):

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.int_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.float_formatter))
        plt.setp(self.ax.get_xticklabels(), fontsize=16, weight='bold')
        plt.setp(self.ax.get_yticklabels(), fontsize=16, weight='bold')


class AlphaPlot(TEplot):
    """
    Plots temperature dependent lattice parameters
    """

    def __init__(self, ax=None, field='alpha', **kwargs):
        self.field = field
        super(AlphaPlot, self).__init__(ax=ax)

    def plot_te(self, te_data, marker_list=None, color_list=None, **kwargs):

        if self.field == 'alpha' or self.field == 'alpha_a':
            a = 0
        elif self.field == 'alpha_c':
            a = -1

        for i, ncfile in enumerate(te_data):
            if marker_list:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.alpha[a, :]*1E6, marker=marker_list[i],
                                 color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.alpha[a, :]*1E6, marker=marker_list[i], **kwargs)
            else:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.alpha[a, :]*1E6, color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.alpha[a, :]*1E6, **kwargs)

    def check_units(self, ncfile):
        if ncfile.yaxis_units.find('K') != -1:
            ncfile.yaxis = ncfile.yaxis*1E6
        else:
            raise Exception('Units from EXPfile should be "K^-1" and contain "K"')

    @property
    def xlabel(self):
        return r'Temperature (K)'

    @property
    def ylabel(self):
        if self.field == 'alpha_a':
            return r'$\alpha_a$ ($10^{-6}K^{-1}$)'
        elif self.field == 'alpha_c':
            return r'$\alpha_c$ ($10^{-6}K^{-1}$)'
        else:
            return r'$\alpha$ ($10^{-6}K^{-1}$)'

    def format_ticks(self):

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.int_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.float_formatter))
        plt.setp(self.ax.get_xticklabels(), fontsize=16, weight='bold')
        plt.setp(self.ax.get_yticklabels(), fontsize=16, weight='bold')


class RoomAlphaPlot(AlphaPlot):

    def __init__(self, ax=None, field='room_alpha', **kwargs):
        self.field = field
        super(RoomAlphaPlot, self).__init__(ax=ax)

    def plot_te(self, te_data, marker_list=None, color_list=None, **kwargs):

        if self.field == 'room_alpha' or self.field == 'room_alpha_a':
            a = 0
        elif self.field == 'room_alpha_c':
            a = -1

        for i, ncfile in enumerate(te_data):
            if marker_list:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.room_temp_alpha[a, :]*1E6, marker=marker_list[i],
                                 color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.room_temp_alpha[a, :]*1E6, marker=marker_list[i], **kwargs)
            else:
                if color_list:
                    self.ax.plot(ncfile.temperature, ncfile.room_temp_alpha[a, :]*1E6, color=color_list[i], **kwargs)
                else:
                    self.ax.plot(ncfile.temperature, ncfile.room_temp_alpha[a, :]*1E6, **kwargs)

    @property
    def ylabel(self):
        if self.field == 'room_alpha_a':
            return r'$\alpha_a$ ($10^{-6}K^{-1}$), ref. 293K'
        elif self.field == 'room_alpha_c':
            return r'$\alpha_c$ ($10^{-6}K^{-1}$), ref. 293K'
        else:
            return r'$\alpha$ ($10^{-6}K^{-1}$), ref. 293K'


class FreeEnergy1DPlot(TEplot):
    """
    Plots temperature dependent free energy curve
    """

    def __init__(self, ax=None, field='free_energy_1d', **kwargs):
        self.field = field
        super(FreeEnergy1DPlot, self).__init__(ax=ax)

    def plot_te(self, te_data, temp=None, withstatic=True, cmap='jet',  **kwargs):

        if len(te_data) > 1:
            raise Exception("""FreeEnergy1D plots handles only one file at a time,
                             and you provided {}.""".format(len(te_data)))

        ncfile = te_data[0]

        if temp is not None:
            tmin = self.find_temp_index(temp[0], ncfile.temperature)
            tmax = self.find_temp_index(temp[-1], ncfile.temperature)

            tarr = ncfile.temperature[tmin:tmax+1]
        else:
            tmin = ncfile.temperature[0]
            tmax = ncfile.temperature[-1]
            tarr = ncfile.temperature

        fmin = np.zeros((len(tarr)+1))
        amin = np.zeros((len(tarr)+1))

        # Plot static data
        equilibrium_index = np.where(ncfile.volume[:, 1] == ncfile.equilibrium_volume[1]/bohr_to_ang)
        if withstatic:
            self.ax.plot(ncfile.volume[:, 1], ncfile.static_energy[:, 0], marker='x', color='k', linestyle='None')
            self.ax.plot(ncfile.equilibrium_volume[1]/bohr_to_ang, ncfile.static_energy[equilibrium_index, 0],
                         marker='o', color='k')

        amin[0] = ncfile.equilibrium_volume[1]/bohr_to_ang
        amin[1:] = ncfile.acell[0, tmin:tmax+1]/bohr_to_ang

        x = np.linspace(0.998*np.amin(ncfile.volume[:, 1]), 1.002*np.amax(ncfile.volume[:, 1]), 100)

        if ncfile.fit_function == 'Murnaghan':
            fit = eos.murnaghan_EV(x**3/4, ncfile.static_fit_parameters[0],
                                   ncfile.static_fit_parameters[1], ncfile.static_fit_parameters[2],
                                   ncfile.static_fit_parameters[3])
            fmin[0] = eos.murnaghan_EV(amin[0]**3/4, ncfile.static_fit_parameters[0],
                                       ncfile.static_fit_parameters[1], ncfile.static_fit_parameters[2],
                                       ncfile.static_fit_parameters[3])

        elif ncfile.fit_function == 'Murnaghan-Birch':
            fit = eos.birch_murnaghan_EV(x**3/4, ncfile.static_fit_parameters[0],
                                         ncfile.static_fit_parameters[1], ncfile.static_fit_parameters[2],
                                         ncfile.static_fit_parameters[3])
            fmin[0] = eos.birch_murnaghan_EV(amin[0]**3/4, ncfile.static_fit_parameters[0],
                                             ncfile.static_fit_parameters[1], ncfile.static_fit_parameters[2],
                                             ncfile.static_fit_parameters[3])

        if withstatic:
            self.ax.plot(x, fit, 'k')

        # Plot Temperature dependent free energy
        mycmap = plt.get_cmap(cmap)
        colors = mycmap(np.linspace(0, 1, len(tarr)))

        for t in range(len(tarr)):
            self.ax.plot(ncfile.volume[:, 1], ncfile.free_energy[:, t], marker='x', color=colors[t], linestyle='None')

            if ncfile.fit_function == 'Murnaghan':
                fit = eos.murnaghan_EV(x**3/4, ncfile.fit_parameters[0, tmin+t],
                                       ncfile.fit_parameters[1, tmin+t], ncfile.fit_parameters[2, tmin+t],
                                       ncfile.fit_parameters[3, tmin+t])
                fmin[t+1] = eos.murnaghan_EV(amin[t+1]**3/4, ncfile.fit_parameters[0, tmin+t],
                                             ncfile.fit_parameters[1, tmin+t], ncfile.fit_parameters[2, tmin+t],
                                             ncfile.fit_parameters[3, tmin+t])

            elif ncfile.fit_function == 'Murnaghan-Birch':
                fit = eos.birch_murnaghan_EV(x**3/4, ncfile.fit_parameters[0, tmin+t],
                                             ncfile.fit_parameters[1, tmin+t], ncfile.fit_parameters[2, tmin+t],
                                             ncfile.fit_parameters[3, tmin+t])
                fmin[t+1, tmin+t] = eos.birch_murnaghan_EV(amin[t+1, tmin+t]**3/4, ncfile.fit_parameters[0, tmin+t],
                                                           ncfile.fit_parameters[1, tmin+t],
                                                           ncfile.fit_parameters[2, tmin+t],
                                                           ncfile.fit_parameters[3, tmin+t])

            self.ax.plot(x, fit, color=colors[t])
            self.ax.plot(amin[t+1], fmin[t+1], marker='o', color=colors[t])
        self.ax.plot(amin[1:], fmin[1:], marker='None', color='darkslategray', linestyle='dashed')


        self.plot_legend(self.ax, tarr, colors)

    def find_temp_index(self, t, arr):
        # Find index of required temperature in the array
        lst = list(arr)
        if t in lst:
            return lst.index(t)
        else:
            raise Exception('Temperature {}K was not found.'.format(t))

    def plot_legend(self, ax, temp, col):

        handles = []

        handles.append(Line2D([0], [0], color='k', linewidth=3.0, label='Static'))

        for t in range(len(temp)):
            handles.append(Line2D([0], [0], color=col[t], linewidth=3.0, label='{:4.0f}K'.format(temp[t])))

        ax.legend(handles=handles, loc=8, bbox_to_anchor=(0.5, -0.30), ncol=6, fontsize=16)

    @property
    def xlabel(self):
        return r'a (Bohr)'

    @property
    def ylabel(self):
        return r'Free energy (Ha)'

    def format_ticks(self):

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.float_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.float4_formatter))
        plt.setp(self.ax.get_xticklabels(), fontsize=16, weight='bold')
        plt.setp(self.ax.get_yticklabels(), fontsize=16, weight='bold')


class FreeEnergy2DPlot(TEplot):
    """
    Plots temperature dependent free energy surface
    """

    def __init__(self, ax=None, field='free_energy_2d', **kwargs):
        self.field = field
        super(FreeEnergy2DPlot, self).__init__(ax=ax)

    def plot_te(self, te_data, temp=0, cmap='plasma', raw=False, **kwargs):

        if len(te_data) > 1:
            raise Exception("""FreeEnergy2D plots handles only one file at a time,
                             and you provided {}.""".format(len(te_data)))

        ncfile = te_data[0]
        index = self.find_temp_index(temp, ncfile.temperature)
        if self.field == 'free_energy_2d':
            print('Plotting 2D free energy for T={}K'.format(temp))
            fe = ncfile.free_energy[:, index]
            title = 'Free energy at {}K'.format(temp)
            write_out = True
        elif self.field == 'static_energy_2d':
            print('Plotting 2D static free energy for T={}K'.format(temp))
            fe = ncfile.static_energy[:, index]
            title = 'Static free energy'
            write_out = True
        elif self.field == 'phonon_free_energy_2d':
            print('Plotting 2D phonon free energy for T={}K'.format(temp))
            fe = ncfile.phonon_free_energy[:, index]
            title = 'Phonon free energy at {}K'.format(temp)
            write_out = False

        a0, c0 = ncfile.equilibrium_volume[1]/bohr_to_ang, ncfile.equilibrium_volume[3]/bohr_to_ang
        equilibrium_index = np.where(ncfile.volume[:, 0] == ncfile.equilibrium_volume[0])
        if len(equilibrium_index) > 1:
            raise Exception('Equilibrium volume was found twice, check your data!')
        else:
            equilibrium_index = equilibrium_index[0]
        grid, fe2d = self.reshape_fe(ncfile.volume[:, 1], ncfile.volume[:, 3], fe)

        # This is for raw data only
        if raw is True:
            pc = self.ax.pcolormesh(grid[0], grid[1], fe2d.transpose(), cmap=cmap)
        # This is for interpolated data
        else:
            self.ax.contour(grid[0], grid[1], fe2d.transpose(), levels=50, cmap='autumn')
            from scipy.interpolate import griddata
            npoint = 1000
            arrx = np.linspace(np.amin(ncfile.volume[:, 1]), np.amax(ncfile.volume[:, 1]), npoint)
            arry = np.linspace(np.amin(ncfile.volume[:, 3]), np.amax(ncfile.volume[:, 3]), npoint)

            gridx, gridy = np.meshgrid(arrx, arry)
            vlist = [ncfile.volume[:, 1], ncfile.volume[:, 3]]
            interp = griddata(list(zip(*vlist)), fe, (gridx, gridy), method='cubic')
            self.ax.contourf(gridx, gridy, interp, 50, cmap=cmap)
            # mask where I have nans
            interp = np.ma.masked_invalid(interp)
            best_fit_index = np.unravel_index(interp.argmin(), interp.shape)
            best_fit = [arrx[best_fit_index[1]], arry[best_fit_index[0]]]
            if write_out:
                print('Minimal FE : a0 = {:>7.4f}, c0 = {:>7.4f}', best_fit[0], best_fit[1])
                if self.field == 'free_energy_2d':
                    print('delta a = {:>7.4f}, delta c = {:>7.4f}'.format(best_fit[0]-a0, best_fit[1]-c0))

                self.ax.plot(best_fit[0], best_fit[1], marker='x', color='white', markersize=4)

#        corner = self.find_mask_corners(interp.transpose(), npoint)
#        arrx2 = arrx[corner:npoint-corner]
#        arry2 = arry[corner:npoint-corner]
#        meshx2, meshy2 = np.meshgrid(arrx2, arry2)
#        interp2 = griddata(list(zip(*vlist)), fe, (meshx2, meshy2), method='cubic')
#        xdata = np.vstack((meshx2.ravel(), meshy2.ravel()))
#        ydata = interp2.ravel()

#        self.ax.plot(arrx[corner], arry[corner], marker='o', color='r', markersize=4)
#        self.ax.plot(arrx[100-corner], arry[100-corner], marker='o', color='r', markersize=4)
#        self.ax.plot(arrx[corner], arry[100-corner], marker='o', color='r', markersize=4)
#        self.ax.plot(arrx[100-corner], arry[corner], marker='o', color='r', markersize=4)

        self.ax.vlines(a0, np.amin(ncfile.volume[:, 3]), np.amax(ncfile.volume[:, 3]), 'k', linestyle='dotted')
        self.ax.hlines(c0, np.amin(ncfile.volume[:, 1]), np.amax(ncfile.volume[:, 1]), 'k', linestyle='dotted')

        self.ax.set_title(title, fontsize=16)

#        plt.colorbar(pc, ax=self.ax)

    def find_temp_index(self, t, arr):
        # Find index of required temperature in the array
        lst = list(arr)
        if t in lst:
            return lst.index(t)
        else:
            raise Exception('Temperature {}K was not found.'.format(t))

    def reshape_fe(self, a, c, data):

        a0 = np.unique(a)
        c0 = np.unique(c)
        arr = np.zeros((len(a0), len(c0)))
        grid = np.meshgrid(a0, c0)

        for v in range(len(data)):
            i = np.where(a0 == a[v])
            j = np.where(c0 == c[v])
            arr[i, j] = data[v]

        arr = np.ma.masked_where(arr == 0, arr)

#        print('reshaped data')
#        for i in range(len(a0)):
#            for j in range(len(c0)):
#                print('{} {} {}'.format(a0[i], c0[j], arr[i,j]))

        return grid, arr

    def find_mask_corners(self, data, n):
        # Find the largest square of data that is not masked
        i = 10
        found = False
        while found is False:
            x = np.ma.array([data[i, i], data[i, n-i], data[n-i, i], data[n-i, n-i]])

            if np.ma.is_masked(x):
                i += 2
            else:
                found is True
                break

        return i

    @property
    def xlabel(self):
        return r'a (Bohr)'

    @property
    def ylabel(self):
        return r'c (Bohr)'

    def format_ticks(self):

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.float_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.float_formatter))
        plt.setp(self.ax.get_xticklabels(), fontsize=16, weight='bold')
        plt.setp(self.ax.get_yticklabels(), fontsize=16, weight='bold')


def generate_plot(
        ax=None,
        field=None,
        te_data=None,
        exp_data=None,
        marker_list=None,
        color_list=None,
        exp_marker_list=None,
        exp_color_list=None,
        label_size=16,

        **kwargs):

    if field.find('acell') != -1:

        myax = AcellPlot(ax=ax, field=field, **kwargs)

    if field.find('alpha') != -1:
        myax = AlphaPlot(ax=ax, field=field, **kwargs)

    if field.find('energy_2d') != -1:
        myax = FreeEnergy2DPlot(ax=ax, field=field, **kwargs)

    if field.find('energy_1d') != -1:
        myax = FreeEnergy1DPlot(ax=ax, field=field, **kwargs)

    myax.plot_te(te_data, marker_list=marker_list, color_list=color_list, linewidth=1.5, **kwargs)
    if exp_data:
        myax.plot_exp_data(exp_data, marker_list=exp_marker_list, color_list=exp_color_list,
                           linestyle='None', markerfacecolor='None', markeredgewidth=1.5)

    myax.set_labels(label_size)
    myax.format_ticks()

    return myax


def read_te_data(flist):
    """
    Read and store TE data in a list of TeFile instances
    """
    te_data = []
    for f in flist:
        data = set_file_class(f)
        te_data.append(data)

    return te_data


def read_exp_data(flist):
    '''
    Read and store experimental data from a list of EXPfile instances
    '''

    exp_data = []
    for f in flist:
        data = EXPfile(f)
        data.read_nc()
        exp_data.append(data)

    return exp_data


def set_legend_handles(markers, colors, labels, markers_exp=None, colors_exp=None, labels_exp=None):
    legend_handles = []

    for i, (marker, color, label) in enumerate(zip(markers, colors, labels)):
        legend_handles.append(Line2D([0], [0], marker=marker, linestyle='solid',
                              linewidth=1.5, color=color, label=label))

    for i, (marker, color, label) in enumerate(zip(markers_exp, colors_exp, labels_exp)):

        legend_handles.append(Line2D([0], [0], marker=marker, linestyle='None', color=color,
                              mew=1.5, markerfacecolor='None', label=label))

    return legend_handles
