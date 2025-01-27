# -*- coding: utf-8 -*-

"""
python package for the analysis of absorption images
developed by members of the Lithium Project
"""

import numpy as np
import scipy.constants as const


def intensity(power, waist):
    """
    Function:
        This function calculates the intensity of a laser beam for a given power and waist.

    Arguments:
        power -- {scalar} laser power [W]
        waist -- {scalar} laser waist/beam width [m]

    Returns:
        {scalar} intensity [W m^-2]
    """

    return 2 * power / (const.pi * waist**2)


def U_dip(wavelength, power, waist):
    """
    Function:
        This function calculates the dipole potential of a laser beam for a given wavelength power and waist.
        In its current implementation it's only valid for the D2 transition in lithium 6.
        It is realized by specifying the resonant frequency of the transition and the line width of the transition.
        It must be noted, that the formula used is an approximation to the actual analytical expression.
        it is only valid for atoms that have very close excited states like is the case for lithium 6 (D2, D3 transitions)

    Arguments:
        wavelength -- {scalar} laser wavelength [nm]
        power      -- {scalar} laser power [W]
        waist      -- {scalar} laser waist/beam width [m]

    Returns:
        {scalar} dipole potential [J]
    """

    # D2 transition angular frequency [Hz]
    F_RES = 2 * const.pi * const.c / 670.977338e-9

    # D2 transition line width [Hz]
    GAMMA = 2 * const.pi * 5.8724e6

    f_laser = 2 * const.pi * const.c / (wavelength * 1e-9)
    D2_detuning = np.abs(f_laser - F_RES)

    return 3 * const.pi * const.c**2 / (2 * F_RES**3) * GAMMA / D2_detuning * intensity(power, waist)


def trap_freq(wavelength, power, waist):
    """
    Function:
        This function calculates the trap frequency of a dipole trap.
        In its current implementation it's only valid for the D2 transition in lithium 6.
        It is realized by specifying the resonant frequency of the transition, the line width of the transition and the mass of atomic lithium 6.

    Arguments:
        wavelength -- {scalar} laser wavelength [nm]
        power      -- {scalar} laser power [W]
        waist      -- {scalar} laser waist/beam width [m]

    Returns:
        {scalar} trap frequency [Hz]
    """

    # mass of lithium [kg]
    M_LI = 9.9883414e-27

    return np.sqrt(4 * U_dip(wavelength, power, waist) / (M_LI * waist**2)) / (2 * const.pi)