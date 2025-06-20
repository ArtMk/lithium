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


def plot_images(images, folder_out, prefix='test', loop_var_name='Vortex_hold ', images_per_row=5, cmap='gray'):
    """
    Load all images in `folder` whose filenames start with `prefix` and display them in rows.
    Titles above each image will be loop_var_name and then the numbers in the suffix after `prefix` (filename without extension).
    Images are ordered based on the integer parsed from the suffix.


    Args:
        folder (str): Path to the folder containing images.
        prefix (str): Filename prefix to filter images.
        images_per_row (int): Number of images per row before breaking to the next row.
        loop_var_name (str): only for the title
    """
    # Gather image file paths
    files = [f for f in os.listdir(folder) if f.startswith(prefix)]
    image_paths = [os.path.join(folder, f) for f in files]

    # Load images and extract suffix titles and order keys
    image_infos = []  # list of tuples: (Image, suffix, order)
    for path in image_paths:
        try:
            img = Image.open(path)
            name, _ = os.path.splitext(os.path.basename(path))
            suffix = name[len(prefix):]
            # Extract digits for ordering; non-digit chars are ignored
            digits = ''.join(filter(str.isdigit, suffix))
            order = int(digits) if digits else float('inf')
            image_infos.append((img, suffix, order))
        except IOError:
            print(f"Warning: {path} is not a valid image file and will be skipped.")

    if not image_infos:
        print("No images found with the given prefix.")
        return

    # Sort images by the extracted numeric order key
    image_infos.sort(key=lambda x: x[2])

    total = len(image_infos)
    rows = (total + images_per_row - 1) // images_per_row

    # Create subplots
    fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 2, rows * 2))
    axes = axes.flatten() if total > 1 else [axes]

    # Plot images
    for ax, (img, title, _) in zip(axes, image_infos):
        ax.imshow(img, cmap=cmap)
        title
        ax.set_title(loop_var_name + title)
        ax.axis('off')

    # Turn off any unused axes
    for ax in axes[len(image_infos):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(folder_out + '/Sequence_' + prefix, dpi=100)
    plt.show()