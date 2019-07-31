#! usr/bin/env python

from numpy import sqrt
# Murnaghan equation of state

def murnaghan_EV(V, V0,E0,K0,K0p):

    eos = E0 + K0*V/K0p*( ((V0/V)**K0p)/(K0p-1) + 1) - K0*V0/(K0p-1)

    return eos 

def murnaghan_PV(V, V0,K0,K0p):

    P = K0/K0p*( (V/V0)**(-K0p) -1 )

    return P

def murnaghan_EV_axial(V, a0,c0,E0,K0,K0p):

    b = sqrt(3)/2.
    eos = E0 + K0*(V)/K0p*( (((b*a0**2*c0)/(V))**K0p)/(K0p-1) + 1) - K0*(b*a0**2*c0)/(K0p-1)

    return eos 

def birch_murnaghan_EV(V,V0,E0,K0,K0p):

    eos = E0 + 9*V0*K0/16*( (( (V0/V)**(2./3) - 1)**3)*K0p + (( (V0/V)**(2./3) -1)**2)*(6 - 4*(V0/V)**(2./3)))

    return eos

def birch_murnaghan_EV_axial(V,a0,c0,E0,K0,K0p):

    b = sqrt(3)/2.
    eos = E0 + 9*(b*a0**2*c0)*K0/16*( (( (b*a0**2*c0/V)**(2./3) - 1)**3)*K0p + (( (b*a0**2*c0/V)**(2./3) -1)**2)*(6 - 4*(b*a0**2*c0/V)**(2./3)))

    return eos
