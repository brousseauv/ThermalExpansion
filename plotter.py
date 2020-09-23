import matplotlib as mpl
import matplotlib.pyplot as plt


class Plotter(object):
    """
    Creates a matplotlib.figure.Figure object
    """

    def _init_figure(self, ax=None, **kwargs):
        """ Initialize internal Figure object """

        # Initialize figure and ax
        if ax:
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0., 0., 1., 1.])
            self.set_size(*kwargs.get('figsize', (6, 6)))

    def show(self):
        """ Display figure"""
        plt.show()

    def get_fig(self):
        """ Get figure """
        return self.fig

    def get_ax(self):
        """ Get ax """
        return self.ax

    @staticmethod
    def get_fname(ftype, rootname, where='.'):
        """ Return file name for a figure """
        from os.path import join as pjoin
        return pjoin('{}'.format(where), '{}.{}'.format(rootname, ftype))

    @staticmethod
    def check_dir(fname):
        """ Check if required directory exists """
        from os.path import dirname, exists
        dirnm = dirname(fname)
        if not exists(dirnm):
            from subprocess import call
            call(['mkdir', '-p', dirnm])

    def savefig(self, fname, **kwargs):
        return self.fig.savefig(fname, **kwargs)

    def set_size(self, w=8, h=6):
        """ Set figure size """
        self.fig.set_size_inches(w, h)
