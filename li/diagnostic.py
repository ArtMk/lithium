# -*- coding: utf-8 -*-

"""
python package for the analysis of absorption images
developed by members of the Lithium Project
"""

import os
import numpy as np
import scipy.constants as const


# constants

# trapping frequency
f_res = 2 * const.pi * const.c / 670.977338e-9 # D2 transition angular frequency [Hz]
Gamma = 2 * const.pi * 5.8724e6                # D2 transition line width [Hz]
m_Li = 9.9883414e-27                           # mass of lithium [kg]

# Breit-Rabi
vHFS = 228205260                        # HFS for B = 0 in [MHz]
I = 1                                   # corespin
g_I = -0.0004476540                     # g-factor for the core
g_e = -const.value("electron g factor") # g-factor for electrons
mu_B = const.value("Bohr magneton")     # Bohr magneton
h = const.h                             # Planck constant

# scattering length
A = 0.915                        # scaling constant for a_2D and a_3D
a_0 = const.value('Bohr radius') # Bohr radius


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


def U_dip(wavelength, power, waist, from_trap_freq, trap_freq = 0):
    """
    Function:
        This function calculates the dipole potential of a laser beam for given beam characteristics or a known trap frequency.
        In its current implementation it's only valid for the D2 transition in lithium 6.
        It is realized by specifying the resonant frequency of the transition and the line width of the transition.
        It must be noted, that the formula used is an approximation to the actual analytical expression.
        it is only valid for atoms that have very close excited states like is the case for lithium 6 (D1, D2 transitions)

    Arguments:
        wavelength     -- {scalar} laser wavelength [m]
        power          -- {scalar} laser power [W]
        waist          -- {scalar} laser waist/beam width [m]
        from_trap_freq -- {bool} calculate dipole potential from trap frequency
        trap_freq      -- {scalar} trap frequency [Hz]

    Returns:
        {scalar} dipole potential [K]
        or
        {scalar} dipole potential [J]
    """

    f_laser = 2 * const.pi * const.c / wavelength
    D2_detuning = np.abs(f_laser - f_res)

    if from_trap_freq:
        return m_Li * waist**2 * const.pi**2 * trap_freq**2 / const.k

    else:
        return 3 * const.pi * const.c**2 / (2 * f_res**3) * Gamma / D2_detuning * intensity(power, waist)


def trap_frequency(wavelength, power, waist, from_trap_freq = False):
    """
    Function:
        This function calculates the trap frequency of a dipole trap.
        In its current implementation it's only valid for the D2 transition in lithium 6.
        It is realized by specifying the resonant frequency of the transition, the line width of the transition and the mass of atomic lithium 6.

    Arguments:
        wavelength     -- {scalar} laser wavelength [m]
        power          -- {scalar} laser power [W]
        waist          -- {scalar} laser waist/beam width [m]
        from_trap_freq -- {bool} leave to False

    Returns:
        {scalar} trap frequency [Hz]
    """

    return np.sqrt(4 * U_dip(wavelength, power, waist, from_trap_freq) / (m_Li * waist**2)) / (2 * const.pi)


def breit_rabi(B, state): # B in Gaus, return in MHz
    """
    Function:
        This function calculates the Breit Rabi splitting of hyperfine states.

    Arguments:
        B     -- {scalar} magnetic field [G]
        state -- {integer} hyperfine state

    Returns:
        {scalar} some frequency, idk [Hz]

    """

    # hyperfine momentum quantum number
    F = get_F(state)

    # hyperfine magnetic quantum number
    m_F = get_m_F(state)

    a = (g_e - g_I) * mu_B * B * 1e-4 / (h * vHFS)

    # root = 0

    if F != abs(m_F) or F < I + 1/2:
        root = np.sqrt(1 + 4 * m_F / (2 * I + 1) * a + a**2)

    else:
        root = 1 + m_F / abs(m_F) * a

    return -vHFS / (2 * (2 * I + 1)) + g_I * m_F * mu_B * B * 1e-4 / h + vHFS * (F - 1) * root


def get_F(state):
    """
    Function:
        This function assigns the hyperfine momentum quantum number F to the numbered states.

    Arguments:
        state -- {scalar} numbered hyperfine state

    Returns:
        {scalar} hyperfine momentum quantum number F
    """

    if state <= 2:
        return 1/2

    if state > 2:
        return 3/2


def get_m_F(state):
    """
    Function:
        This function assigns the hyperfine magnetic quantum number m_F to the numbered states.

    Arguments:
        state -- {scalar} numbered hyperfine state

    Returns:
        {scalar} hyperfine magnetic quantum number m_F
    """
    if state <= 2:
        return -state+3/2

    if state > 2:
        return state-9/2


### ~~~~~~~~~~~~~~~~ SCATTERING LENGTH ~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~ SCATTERING LENGTH ~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~ SCATTERING LENGTH ~~~~~~~~~~~~~~~~ ###


def scattering_length_2D(a_3D, freq_z, m = m_Li):
    """
    Function:
        This function calculates the 2D scattering length for a given 3D scattering length and vertical confinement.

    Arguments:
        a_3D -- {scalar} 3D scattering length [m]

    Returns:
        {scalar} 2D scattering length [m]
    """

    l_z = harmonic_osc_length(freq_z, m)

    return l_z * np.sqrt(const.pi / A) * np.exp(- np.sqrt(const.pi / 2) * l_z / (a_3D * a_0))


def harmonic_osc_length(freq_z, m = m_Li):
    """
    Function:
        This function calculates the harmonic oscillator length of the accordion.

    Arguments:
        freq_z -- {scalar} trap frqeuency [Hz]
        m      -- {scalar} particle mass [kg]

    Returns:
        {scalar} harmonic oscillator length [m]
    """

    return np.sqrt(const.hbar / (m * 2 * const.pi * freq_z))


def k_F(n):
    """
    Function:
        This function calculates the Fermi momentum for a given 2D cloud density.

    Arguments:
        n -- {scalar} 2D cloud density [m^-2]

    Returns:
        {scalar} Fermi momentum [m^-1]
    """

    return np.sqrt(4 * const.pi * n)


def E_F(n):
    """
    Function:
        This function calculates the Fermi energy for a given 2D cloud density.

    Arguments:
        n -- {scalar} 2D cloud density [m^-2]

    Returns:
        {scalar} Fermi energy [J]
    """

    return const.hbar**2 * k_F(n)**2 / (2 * m_Li)


def T_F(n):
    """
    Function:
        This function calculates the Fermi temperature for a given 2D cloud density.

    Arguments:
        n -- {scalar} 2D cloud density [m^-2]

    Returns:
        {scalar} Fermi temperature [K]
    """

    return E_F(n) / const.k


def B_to_a2D(B_field, freq_z, m = m_Li, species = "1_2"):

    FB = np.loadtxt("scattering_length_Li6_a_vs_B_12_13_23.txt", skiprows = 1)

    a_3D = []

    if species == "1_2":
        index = 1
    elif species == "1_3":
        index = 2
    elif species == "2_3":
        index = 3

    for B in B_field:
        a_3D.append(FB[:, index][np.isclose(FB[:, 0], B)][0])

    a_3D = np.array(a_3D)

    return scattering_length_2D(a_3D, freq_z, m = m)


def ln_kF_a2D(B_field, n, freq_z, m = m_Li, species = "1_2"):

    a_2D = B_to_a2D(B_field, freq_z, m = m, species = species)

    return np.log(k_F(n) * a_2D)

