__author__ = "brousseauv"

""" Base class and subclasses for ThermalExpansion output *_TE.nc
    plotting, with possible comparison with experiment.
"""

from .plotter import Plotter
from .tefile import set_file_class
from expfile import EXPfile
from constants import bohr_to_ang
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

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
