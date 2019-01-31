#! usr/bin/env python

# Murnaghan equation of state

def murnaghan_EV(V, V0,E0,K0,K0p):

    eos = E0 + K0*V/K0p*( ((V0/V)**K0p)/(K0p-1) + 1) - K0*V0/(K0p-1)

    return eos 

def murnaghan_PV(V, V0,K0,K0p):

    P = K0/K0p*( (V/V0)**(-K0p) -1 )

    return P

