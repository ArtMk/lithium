# -*- coding: utf-8 -*-

"""
python package for the analysis of absorption images
developed by members of the Lithium Project
"""

import seaborn
import matplotlib.pyplot as plt
from li.diagnostic import breit_rabi


def breit_rabi_visualize(B, states):
    """
    Function:
        This function plots the Breit-Rabi splitting for a given magnetic field range and a collection of states.

    Arguments:
        B      -- {array-like} magnetic field range
        states -- {array-like} states to be displayed

    Returns:
        nothing, it just makes a plot
    """

    colors = ['gray' for i in range(6)]
    colors_selected = ['steelblue', 'lightsteelblue', 'lightcoral', 'indianred', 'firebrick', 'darkred']

    plt.figure(figsize = (10,7))

    for state in reversed(range(1, 7)):
        lw = None
        ls = "--"

        if state in states:
            colors[state - 1] = colors_selected[state - 1]
            lw = 2.5
            ls = "-"

        plt.plot(B, breit_rabi(B, state), label=f'$|{state}\\rangle$', color = colors[state - 1], lw = lw, ls = ls)

    plt.title('Hyperfine Splitting of the Ground State', fontsize = 18, pad = 13)
    plt.xlabel('Magnetic Field Strength [G]', fontsize = 15)
    plt.ylabel('$\\Delta\\,\\nu$ [MHz]', fontsize = 15)

    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)

    plt.legend(loc='center right', fontsize = 15)

    plt.show()


def spectrum(images, index, columns, values, title, vmin = 0, vmax = 1, cmap = "viridis"):
    """
    Function:
        This function visualizes the response as a function of all loop variables in a heatmap.

    Arguments:
        images  -- {pandas dataframe, containing respsonse from T4 peaks
        index   -- {string} loop variable on y-axis
        columns -- {string} loop variable on x-axis
        values  -- {string} heatmap values (usually response)
        title   -- {string} title of the heatmap
        vmin    -- {scalar} lower bound of colormap
        vmax    -- {scalar} upper bound of colormap
        cmap    -- {string} colormap name

    Returns:
        {matplotlib axis} heatmap of response
    """

    # turn dataframe into heatmap shape
    heat = images.pivot(index = index, columns = columns, values = values)

    ax = plt.axes()

    # plot heatmap
    seaborn.heatmap(heat, ax = ax, vmin = vmin, vmax = vmax, cmap = cmap).invert_yaxis()

    ax.set_title(f"{title}", pad = 13)

    return