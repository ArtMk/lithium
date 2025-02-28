# -*- coding: utf-8 -*-

"""
python package for the analysis of absorption images
developed by members of the lithium project
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn

import imageio
import xml.etree.ElementTree as ET
import re

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy import constants as const
from PIL import Image
from alive_progress import alive_bar

from lithium.li import EvaluationHelpers


# constants

omega_T4 = 2 * const.pi * 27.3 # angular frequency of T4 harmonic [Hz]
m_Li = 9.9883414e-27           # mass of lithium [kg]
px_to_x = 1.09739368998628e-6  # effective pixel size in [m]

wl_laser = 671e-9  # resonant wavelength for imaging
sigma_factor = 1
sigma_zero = 3 * (wl_laser ** 2) / (2 * np.pi)
sigma_eff = sigma_factor * sigma_zero

A = px_to_x**2  # pixel size in m**2
gain = 1


# ~~~ FUNCTIONS FOR IMAGE PROCESSING ~~~ #

def rectangular_mask(image_shape, center, h, w):
    """
    Function:
        This function creates a rectangular mask to select the region of interest in T4 measurements.
        While 'image_shape' is automatically inherited from the original density images,
        'center', 'h' and 'w' have to be determined manually.

    Arguments:
        image_shape -- {array-like} shape of the full image
        center      -- {array-like} center of the rectangular mask
        h           -- {scalar} height of the rectangular mask
        w           -- {scalar} width of the rectangular mask

    Returns:
        {array-like} mask selecting the region of interest in T4 measurements
    """

    Y, X = np.ogrid[:image_shape[0], :image_shape[1]]
    mask = ~((Y >= center[0] - h) & (Y <= center[0] + h) & (X >= center[1] - w) & (X <= center[1] + w))

    return mask


def circular_mask(image_shape, center, radius):
    """
    Function:
        This function creates a circular mask to select the region of interest in inSitu measurements.
        While 'image_shape' is automatically inherited from the original density images,
        'center', 'radius' have to be determined manually.

    Arguments:
        image_shape -- {array-like} shape of the full image
        center      -- {array-like} center of the circular mask
        radius      -- {scalar} radius of the rectangular mask

    Returns:
        {array-like} mask selecting the region of interest in inSitu measurements
    """

    Y, X = np.ogrid[:image_shape[0], :image_shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask


def density_builder(images, keys, center, h, w, Csat_rate, illumination_time, progress_disable):
    """
    Function:
        This function creates a density image after reading in the paths of the raw images.
        It also applies a mask to the densities, selecting the region of interest in T4 measurements.

    Arguments:
        images            -- {dictionary} collection of  all image paths and specific values of loop variables
        keys              -- {list of strings} keys of all loop variables
        center            -- {array-like} center of the rectangular mask for region of interest
        h                 -- {scalar} height of the rectangular mask for region of interest
        w                 -- {scalar} width of the rectangular mask for region of interest
        Csat_rate         -- {scalar} saturation rate value for imaging
        illumination_time -- {scalar} imaging illumination time [s]
        progress          -- {boolean} progress bar

    Returns:
        {pandas dataframe} densities for all combinations of loop variables
    """

    # ~~~ CONSTANTS ~~~ #

    counts_sat = Csat_rate * illumination_time * 1e6

    # ~~~ DENSITY CALCULATION ~~~ #

    # dictionary
    images_prc = {}

    # create dictionary fields for the loop variables
    for key in keys:
        images_prc[key] = []

    # create a new fields for the calculated density and individual atoms and bright images
    images_prc["density"] = []
    images_prc["atoms"] = []
    images_prc["bright"] = []

    # fill the dictionary
    with alive_bar(len(images), force_tty = True, spinner = "twirl", disable = progress_disable) as bar:
        for image_set in images:

            # density calculation
            density = EvaluationHelpers.makeDensityImage(image_set["atoms"], image_set["reference"], counts_sat, sigma_eff,
                                                         image_set["atom_background"], image_set["ref_background"]) * A

            # atoms and bright calculation (for fluctuation)
            atoms = EvaluationHelpers.loadImageRemoveDark(image_set["atoms"], image_set["atom_background"])
            bright = EvaluationHelpers.loadImageRemoveDark(image_set["reference"], image_set["ref_background"])

            # ROI mask for T4 measurement
            T4_mask = rectangular_mask(np.shape(density), center, h, w)

            # append density, atoms and bright as masked array
            images_prc["density"].append(ma.array(density, mask = T4_mask))
            images_prc["atoms"].append(ma.array(atoms, mask = T4_mask))
            images_prc["bright"].append(ma.array(bright, mask = T4_mask))

            for key in keys:
                images_prc[key].append(image_set[key])

            bar()

    # make pandas dataframe out of dictionary
    images_prc = pd.DataFrame(images_prc)

    return images_prc


def filter(images, threshold):
    """
    Function:
        This function filters the density images for missed shots.
        This is done by calculating the number of atoms in the region of interest,
        where entries with atom numbers lower than the threshold are dropped.
        The threshold has to be determined manually.

    Arguments:
        images    -- {pandas dataframe} densities for all combinations of loop variables
        threshold -- {scalar} threshold for filtering missed shots

    Returns:
        {pandas dataframe} densities for all combinations of loop variables filtered for missed shots
    """

    for i, im in images.iterrows():

        # compare atom number in ROI to threshold
        if np.sum(im["density"].compressed()) < threshold:
            images = images.drop([i])

    return images


def group(images, keys, key_kill):
    """
    Function:
        This function averages the densities for all combinations of loop variables with respect to the variable
        'key_kill'. The averaging is usually performed over the iterations such that 'key_kill' = 'i',
        though it can be any of the loop variables. The averaging is done by grouping the entries in the input
        dataframe by the loop variables meant to be kept and averaging over the densities in the respective groups.

    Arguments:
        images   -- {pandas dataframe} densities for all combinations of loop variables filtered for missed shots
        keys     -- {array-like} keys of all loop variables
        key_kill -- {string} loop variable to be averaged over and "killed" from the dataframe

    Returns:
        {pandas dataframe} densities for all combinations of loop variables averaged over key_kill
    """

    counts_sat = 56

    key_group = keys.copy()
    key_group.remove(key_kill)

    if len(keys) == 1:

        images_grp = {}

        # averages
        images_grp["density"] = [images["density"].mean(numeric_only = False)]
        images_grp["atoms"] = [images["atoms"].mean(numeric_only=False)]
        images_grp["bright"] = [images["bright"].mean(numeric_only=False)]

        # variances for fluctuation calculation
        # images_grp["atoms_var"] = [images["atoms"].std(numeric_only = True) ** 2]
        # images_grp["bright_var"] = [images["bright"].std(numeric_only = True) ** 2]

        images_grp["atoms_var"] = [np.std(images["atoms"].to_numpy())**2]
        images_grp["bright_var"] = [np.std(images["bright"].to_numpy())**2]
        images_grp["fringe_var"] = [images_grp["bright_var"][0] - gain * images_grp["bright"][0]]

        images_grp["number_var"] = [(A / sigma_eff * (1/images_grp["atoms"][0] + 1/counts_sat))**2 * (images_grp["atoms_var"][0] - gain * images_grp["atoms"][0] - images_grp["fringe_var"][0])]

        images_grp = pd.DataFrame(images_grp)

    else:
        # group by group keys, calculate mean of densities, reshape dataframe to grouped dataframe
        images_grp = images.groupby(key_group).mean(numeric_only = False).reset_index().drop(columns = [key_kill])

    return images_grp


def gauss(x, a, b, c):
    """
    Function:
        Gaussian

    Arguments:
        x -- {array-like} x coordinate
        a -- {scalar} amplitude parameter
        b -- {scalar} mean parameter
        c -- {scalar} standard deviation parameter
        d -- {scalar} vertical offset parameter (not used)

    Returns:
         {array-like} Gaussion evaluated at x
    """

    return a * np.exp(-(x - b)**2 / c) # + d


def parab(x, e, b, f):
    """
    Function:
        Parabola where negative points are masked.

    Arguments:
        x -- {array-like} x coordinate
        e -- {scalar} amplitude parameter
        b -- {scalar} mean parameter
        f -- {scalar} width parameter

    Returns:
        {array-like} parabola evaluated at x
    """

    # parab = -e * (x - b)**2 + f
    parab = 1 - ((x - b)/f)**2

    # mask points where parabola is negative
    mask = (parab < 0)
    parab[mask] = 0

    parab = e * parab**(3/2)

    return parab


def gauss_parab(x, a, b, c, e, f):
    """
    Function:
        Gaussian plus parabola

    Arguments:
        x -- {array-like} x coordinate
        a -- {scalar} amplitude parameter Gaussian
        b -- {scalar} mean parameter
        c -- {scalar} standard deviation parameter Gaussian
        d -- {scalar} vertical offset parameter Gaussian
        e -- {scalar} amplitude parameter parabola
        f -- {scalar} width parameter parabola

    Returns:
        array-like, Gaussian + parabola evaluated at x
    """

    return gauss(x, a, b, c) + parab(x, e, b, f)


def running_average(x, w):
    """
    Function:
        This function is a smoothing function that computes the running average of an input array.
        This is done by convolving nearest neighbors in a window of size 'w'.

    Arguments:
        x -- {array-like}
        w -- {scalar} convolution window size

    Returns:
        {array-like} running average of input array
    """

    return np.convolve(x, np.ones(w), 'same') / w


def T4_fit(images):
    """
    Function:
        This function determines the T4 peaks of the densities using fitting and running average methods.
        The densities in the region of interest are averaged along the beatle to yield an averaged density profile.
        The peaks are determined using both a gauss + parabola fit as well as a double running average.

    Arguments:
        images -- {pandas dataframe} averaged densities for all combinations of loop variables

    Returns:
        {pandas dataframe} additionally containing fit parameters and T4 peaks
    """

    images_fit = images.copy()

    # add new columns
    T4_params = []
    T4_peak = []
    T4_run_peak = []
    temperature = []

    for i, im in images_fit.iterrows():

        # calculate averaged density profile
        T4 = np.mean(im["density"], axis = 1).compressed()
        pos = np.arange(0, len(T4))

        # peak from gauss + parabola fit
        popt, pcov = curve_fit(gauss_parab, pos, T4, p0 = [0.5, 50, 700, 1, 5], bounds=([0, 40, 0, 0, 0], [1, 60, 1000, 3, 10]))

        T4_params.append(popt)
        T4_peak.append(popt[0] + popt[3])

        # peak from double running average
        T4_run_peak.append(np.max(running_average(running_average(T4, 5), 5)))

        # calculate temperature
        T = popt[2] * px_to_x**2 * m_Li * omega_T4**2 / const.k * 1e9
        T_err = px_to_x**2 * m_Li * omega_T4**2 / const.k * 1e9 * np.sqrt(np.diag(pcov)[2])
        temperature.append([T, T_err])

    images_fit["T4_params"] = T4_params
    images_fit["T4_peak"] = T4_peak
    images_fit["T4_run_peak"] = T4_run_peak
    images_fit["temperature"] = temperature

    return images_fit


def response(images):
    """
    Function:
        This function calculates the response from the T4 peaks extracted
        from both fitting and running average methods.

    Arguments:
        images -- {pandas dataframe} densities for all combinations of loop variables plus T4 peaks

    Returns:
        {pandas dataframe} additionally containing response from T4 peaks
    """

    images_res = images.copy()

    a = np.array(images_res["T4_peak"])
    b = np.array(images_res[images_res["Acc_heat_freq"] == 0.0]["T4_peak"])
    a0  = np.tile(b, len(images_res) // len(b))

    a_run = np.array(images_res["T4_run_peak"])
    b_run = np.array(images_res[images_res["Acc_heat_freq"] == 0.0]["T4_run_peak"])
    a0_run  = np.tile(b_run, len(images_res) // len(b_run))

    images_res["response"] = a0 / a - np.ones(len(a))
    images_res["response_run"] = a0_run / a_run - np.ones(len(a_run))

    return images_res


def visualize(images, index, columns, values, title, vmin = 0, vmax = 1, cmap = "viridis"):
    """
    Function:
        This function visualizes the response as a function of all loop variables in a heatmap.

    Arguments:
        images  -- {pandas dataframe, containing respsonse from T4 peaks
        index   -- {string} loop variable on y-axis
        columns -- {string} loop variable on x-axis
        values  -- {string} loop variable as heatmap values
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







#Function for reading images
def sorting(array,sortingName):
    A=[]                                                                                                                
    for k in range(len(array)):
        if sortingName in array[k]:
            A.append(array[k])
            A.sort()
    return A

#Function for reading images
def sorting_2(array,sortingName,sortingName2):
    A=[]
    for k in range(len(array)):
        if sortingName in array[k]:
            if sortingName2 in array[k]:
                A.append(array[k])
                A.sort()
    return A

def average(files):
    aa=np.zeros(img.imread(files[0]).shape)
    for file in files:
        a=img.imread(file)
        aa=a+aa
    return aa/len(files) 

def loop_parameters(string):
    matches=[]
    for n in string:
        n=float(re.search(r"[-+]?(?:\d*\.*\d+)",n).group())
        matches.append(n)
    return matches

def loop_variables(image):
    '''Takes an image from computer control, extracts the xml and returns from the dictionary the 
    variable name and the corresponding loop'''
    imInfo=Image.open(image).info
    l1=imInfo['Control'].split()
    s1="<loops>"
    s2="</loops>"
    matched_indexes = []
    i = 0
    length = len(l1)
    while i < length:
        if s1 == l1[i] or s2 == l1[i]:
            matched_indexes.append(i)
        i += 1
    loop_variables=l1[matched_indexes[0]:matched_indexes[1]]
    var=[match for match in loop_variables if "variablename" in match]
    start=[match for match in loop_variables if "from" in match]
    end=[match for match in loop_variables if "to" in match]
    steps=[match for match in loop_variables if "steps" in match]
    start=loop_parameters(start)
    end=loop_parameters(end)
    steps=loop_parameters(steps)
    T=np.array([start,end,steps])
    
    var_arrayN=[]
    for column in T.T:
        a=np.linspace(column[0],column[1],int(column[2]))
        var_arrayN.append(a)
    
    var_array=[]
    for n in var:
        result = re.search('<variablename>(.*)</variablename>', n).group(1)
        var_array.append(result)
    df=pd.DataFrame(var_arrayN).T
    df.columns = var_array
    # print(df)
    return var_arrayN,var_array,df

def ReadImage(imName,getFormulae=False,fast=False):
    '''Takes a path to an image as a string and returns the image as an array.
    A list of variables from ExpWiz are extracted from xml and returned as a 
    dictionary (variables[name] = value).
    The effective pixel size (accounting for magnification) is returned.
    optional: getFormulae - returns formulae defining all parameters calculated
    in camera control as well as all fits called on the image. formulae is a
    dictionary with keys giving the names of defined variables or fit# where #
    is a number starting from one and going up to formulae['numFits'].
    If fast=True then most of the processing is skipped and the pixel data is
    returned in an array with nothing else.
    
    ReadImage(imName,getFormulae=False,fast=False,Dictionary=True)
    
    return im, variables, pixelSize, *formulae'''
    im = np.array(imageio.imread(imName).astype(float))
    if (fast):
        return im
    if not(getFormulae):
        variables, pixelSize = GetImageMetadata(imName)
        return im, variables, pixelSize
    else:
        variables, pixelSize, formulae = GetImageMetadata(imName,getFormulae=True)
        return im, variables, pixelSize, formulae

def GetImageMetadata(imName,getFormulae=False):
    '''Takes a path to an image as a string and returns metadata.
    A list of variables from ExpWiz are extracted from xml and returned as a 
    dictionary (variables[name] = value).
    Print list of variables using print(variables.keys())
    The effective pixel size (accounting for magnification) is returned.
    optional: getFormulae - returns formulae defining all parameters calculated
    in camera control as well as all fits called on the image. formulae is a
    dictionary with keys giving the names of defined variables or fit# where #
    is a number starting from one and going up to formulae['numFits'].
    
    GetImageMetadata(imName,getFormulae=False)
    
    return variables, pixelSize, *formulae'''
    imInfo = Image.open(imName).info
    #Get all variable names and values
    variables = []
    #<codefromJeff>
    ctr = ET.fromstring(imInfo['Control']) # .ctr file (stored as the header a.k.a. info) parsed as XML
    varis = ctr.find('.//variables') # the part of the XML that contains the variables from ExperimentControl
    #</codefromJeff>
    #From https://stackoverflow.com/questions/4664850/find-all-occurrences-of-a-substring-in-python
    vind1 = [m.start() for m in re.finditer('<variable>\n      <name>', imInfo['Control'])]
    vind2 = [m.start() for m in re.finditer('</name>\n      <value>', imInfo['Control'])]
    numVars = len(vind1)
    variables = {}
    itr = 0
    while (itr<numVars):
        vname = imInfo['Control'][(vind1[itr]+len('<variable>\n      <name>')):vind2[itr]]
        #<codefromJeff>
        vvalue = float(varis.find(f'.//variable[name="{vname}"]').find('value').text) # the value of the variable
        #</codefromJeff>
        variables[vname] = vvalue
        itr+=1
    vname = 'CreationTime'
    vvalue = imInfo[vname]
    variables[vname] = vvalue
    #Get the pixel size in m
    pixelSize = ()
    for x in imInfo['dpi']:
        pixelSize += (0.0254/x,)
    if not(getFormulae):
        return variables, pixelSize
    else:
        processors = ctr.find('.//imaging').find('.//processors')
        formulae = {}
        nFit = 1
        for processor in processors:
            if ((processor.find('.//formula') is not None) and (processor.find('.//uselimits') is None)):
                formulae[processor.find('.//variablename').text] = processor.find('.//formula').text
            elif ((processor.find('.//parameters') is not None) and (processor.find('.//image').text==imInfo['Title'])):
                fit = {}
                basename = processor.find('.//basename').text
                fit['fitType'] = processor.attrib['type']
                fit['parameters'] = [f'{basename}{x.attrib["name"]}' for x in processor.find('.//parameters')]
                fit['results'] = [f'{basename}{x.attrib["name"]}' for x in processor.find('.//results')]
                formulae[f'fit{nFit}'] = fit
                nFit+=1
        formulae['numFits'] = nFit - 1
        return variables, pixelSize, formulae

def radial_average(image,rmax,get_center=False,plot=False):
    # Calculate the center of the image
    '''Perform radial average of an image. 
    get_center=True, the center of the radial average
    is given by half of the shape of the image. 
    get_center=False performs a gaussian filtering on the image
    and selects the max and min pixel of the filtered image. The default value is choosen to 4, however it       can be changed in the gaussian filter function. 
    rmax=defines the largest radius where the radial average is done. 
    Takes an image which was already read as an array. It returns the radial average of the image'''
    if get_center: 
        center = np.array(image.shape)/2
    else:
        center=center_gaussianfilter(image,4,rmax,plot)
        
    # Calculate the maximum radius
    radius = rmax
    
    # Create a 2D array of indices
    y, x = np.indices(image.shape)
    
    # Calculate the distance of each point from the center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create an array to store the radial profile
    radial_profile = np.zeros(int(radius))
    
    # Loop over each radius
    for i in range(int(radius)):
        # Find all pixels with radius within this range
        pixels = np.where((r >= i) & (r < i+1))
        
        # Calculate the mean intensity of those pixels
        mean_intensity = np.mean(image[pixels])
        
        # Store the mean intensity in the radial profile array
        radial_profile[i] = mean_intensity    
    return radial_profile

def center_gaussianfilter(image,sigma,rmax,plot=False):
    image_filter=gaussian_filter(image,sigma=sigma)
    max_rows, max_cols = np.where(image_filter==np.max(image_filter))
    if plot:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        ax.axis('off')
        circ=plt.Circle( (max_rows, max_cols),rmax ,fill = False,color='red')
        ax.add_patch(circ)
        plt.show()
    return max_rows, max_cols