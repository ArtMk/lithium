import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image #for reading images
from scipy.optimize import curve_fit
from scipy.ndimage import rotate

#################################################################
def createImageInfoList(folders, variableNamesList, atomImgName = 'AtomsM', refImgName = 'BrightM', atomBgImgName = 'AtomsDarkM', refBgImgName = "BrightDarkM"):
    '''
    Iterate over all given folders and files within the folders:
    try to recognize what kind of image it is (atom / reference / background) and extract the variable values of the run
    return a list of dictionaries, with each dictionary corresponding to one file
    dictionary: includes the variables values and the exact filepaths to atoms/reference/background images
    '''

    infoList = []

    for folder in folders:
        fl = sorted(os.listdir(folder))
        for fn in fl:
            if atomImgName in fn:

                varValues = extractVariableValuesFromSingleString(fn[:-4], variableNamesList)

                d = {}

                for i in range(len(variableNamesList)):
                    d[variableNamesList[i]] = varValues[i]

                d['atoms'] = os.path.join(folder, fn)
                d['reference'] = os.path.join(folder, fn.replace(atomImgName, refImgName))
                d['atom_background'] = os.path.join(folder, fn.replace(atomImgName, atomBgImgName))
                d['ref_background'] = os.path.join(folder, fn.replace(atomImgName, refBgImgName))

                infoList.append(d)

    return infoList

#################################################################
def aggregateRunsByVariable(variableName, infoList):
    '''
    variableName: variablet to remove
    return will be a list of entries, where all but the given variable are still contained.
    Instead of single image names for atoms / reference / background, there will be a list [[atoms1, ref1, bg1], [atoms2, ref2, bg2], ...]
    '''
    variables = list(infoList[0].keys())
    variables.remove('atoms')
    variables.remove('reference')
    variables.remove('atom_background')
    variables.remove('ref_background')
    variables.remove(variableName)
    
    #Make a new list of infos. The new list contains of dictionaries again
    #Each dictionary contains the values of the remaining variables as well as a list of grouped atoms/ref/background images that match these remaining variables
    newInfoList = []
    for entry in infoList:
        addedEntry = False
        for newEntry in newInfoList:
            matched = True
            for var in variables:
                if entry[var] != newEntry[var]:
                    matched = False
            if matched:
                newEntry['files'].append([entry['atoms'], entry['reference'], entry['atom_background'],  entry['ref_background']])
                addedEntry = True
        if not addedEntry:
            newEntry = {}
            for var in variables:
                newEntry[var] = entry[var]
            newEntry['files'] = []
            newEntry['files'].append([entry['atoms'], entry['reference'], entry['atom_background'],  entry['ref_background']]) 
            newInfoList.append(newEntry)
    return newInfoList

#################################################################
def makeDensityImage(atomsImgPath, refImgPath, counts_sat, sigma_eff, atomsBgImgPath = None, refBgImgPath = None, imgOffset = 0):
    '''
    Create Density Image by optical density = -1 * ln(atomsImg/referenceImg)
    If present, a background image is substracted from the atoms and reference image beforehand
    '''
    atoms = loadAbsorptionDensityImage(atomsImgPath, rescale = False) - imgOffset
    ref = loadAbsorptionDensityImage(refImgPath, rescale = False) - imgOffset
    
    if atomsBgImgPath != None:
        atoms_bg = loadAbsorptionDensityImage(atomsBgImgPath, rescale = False)
    else:
        atoms_bg = np.zeros(atoms.shape)

    if refBgImgPath != None:
        ref_bg = loadAbsorptionDensityImage(refBgImgPath, rescale = False)
    else:
        ref_bg = np.zeros(atoms.shape)
    
    #Potentially clean the images or mask hot pixels or so?
    atomsCleaned = atoms - atoms_bg
    refCleaned = ref - ref_bg

    atomsCleaned = np.maximum(atomsCleaned, 1)
    refCleaned = np.maximum(refCleaned, 1)

    density = (- np.log(atomsCleaned/refCleaned) + (refCleaned - atomsCleaned)/counts_sat)/sigma_eff
    return density

##################################################################

def loadImageRemoveDark(atomsImgPath, atomsBgImgPath = None, imgOffset = 0):
    '''
    load an absorption/reference image and subtract its corresponding dark image
    '''
    atoms = loadAbsorptionDensityImage(atomsImgPath, rescale = False) - imgOffset
    
    
    if atomsBgImgPath != None:
        atoms_bg = loadAbsorptionDensityImage(atomsBgImgPath, rescale = False)
    else:
        atoms_bg = np.zeros(atoms.shape)
    
    #Potentially clean the images or mask hot pixels or so?
    atomsCleaned = atoms - atoms_bg


    atomsCleaned = np.maximum(atomsCleaned, 1)

    return atomsCleaned
#################################################################
def makeAveragedDensityImage(filenameList, averagingOrder = 'normal'):
    '''
    create a averaged density image from the raw file names given in the list
    List has to be of the shape: [ [atoms1, ref1, bg1], [atoms2, ref2, bg2], [atoms3, ref3, bg3], ...]
    set averagingOrder = 'individual' to first average all atoms images, all references, ... and then calculate the average density
    '''
    imgShape = makeDensityImage(filenameList[0][0], filenameList[0][1]).shape
    
    if averagingOrder != 'individual':
        avDensity = np.zeros(imgShape)
        for i in range(len(filenameList)):
            avDensity += makeDensityImage(filenameList[i][0], filenameList[i][1], filenameList[i][2])
        avDensity = avDensity / len(filenameList)
    else:
        avAtoms = np.zeros(imgShape)
        avRef = np.zeros(imgShape)
        avBG = np.zeros(imgShape)
        for entry in filenameList:
            avAtoms += loadAbsorptionDensityImage(entry[0], rescale = False)
            avRef += loadAbsorptionDensityImage(entry[1], rescale = False)
            if entry[2] != None:
                avBG += loadAbsorptionDensityImage(entry[2], rescale = False)
        avAtoms = avAtoms / len(filenameList)
        avRef = avRef / len(filenameList)
        avBG = avBG / len(filenameList)
        
        atomsCleaned = avAtoms - avBG
        refCleaned = avRef - avBG
        atomsCleaned = np.maximum(atomsCleaned, 1)
        refCleaned = np.maximum(refCleaned, 1)
        avDensity = -1 * np.log(atomsCleaned / refCleaned)
    return avDensity

#################################################################
def cropImage(img, xOffset, yOffset, xSize, ySize):
    """
    Crop an image to a smaller size, starting at the given offset
    """
    try:
        newImg = img[yOffset:yOffset+ySize, xOffset:xOffset+xSize]
    except:
        newImg = img
        print('Problem when cropping the image: crop size too large?')
    return newImg

#################################################################
def extractVariableValue(filename, varname):
    filename = filename[:-4]
    ind = filename.find(varname)
    secondhalf = filename[ind+len(varname)+1:]
    varvalend = secondhalf.find('_')
    varval = secondhalf
    if varvalend > 0:
        varval = secondhalf[:varvalend]
    return float(varval)

def loadAbsorptionDensityImage(filepath, rescale = False):
    data = np.asarray(Image.open(filepath), dtype = int)
    if rescale:
        data = data - 5000
        data = data / 18141.88
    return data

def sortFilenamesByVariable(filelist, variableName):
    fndict = {}
    for fn in filelist:
        varVal = extractVariableValue(fn, variableName)
        if varVal not in fndict.keys():
            fndict[varVal] = []
        fndict[varVal].append(fn)
    return fndict

def averageImages(filelist, filepath, imshape = None):
    if imshape == None:
        temp = np.asarray(Image.open(filepath + filelist[0]))
        imshape = temp.shape
    av = np.zeros(imshape)
    for fn in filelist:
        av += np.asarray(Image.open(filepath + fn))
    av = av / len(filelist)
    return av

def gaussian(x, amplitude, waist, xCenter, offset):
    '''
    implements a gaussian in 1D
    waist = 1/e^2 radius
    '''
    return amplitude * np.exp(-2 * (x - xCenter)**2 / waist**2) + offset
    
def tripleGaussian(x, ampCenter, widthCenter, ampPeak, widthPeak, xCenter, deltaX, offset):
    '''
    implements a symmetric triple gaussian in 1D with a central peak and two sidepeaks
    '''
    leftPeak = ampPeak * np.exp(-2 * (x - xCenter + deltaX)**2 / widthPeak**2)
    rightPeak = ampPeak * np.exp(-2 * (x - xCenter - deltaX)**2 / widthPeak**2)
    centerPeak = ampCenter * np.exp(-2 * (x - xCenter)**2 / widthCenter**2)
    return offset + leftPeak + rightPeak + centerPeak

def rotateImage45Deg(imagedata, cutRotatedImage = True):
    '''
    Rotate an image by 45°, and possible cut the new image such that only original data is remaining
    '''
    rotated = rotate(imagedata, angle=45)
    if cutRotatedImage:
        ys = int(round(rotated.shape[0]*0.25))
        ye = int(round(rotated.shape[0]*0.75))
        xs = int(round(rotated.shape[1]*0.25))
        xe = int(round(rotated.shape[1]*0.75))
        rotated = rotated[ys:ye, xs:xe]
    return rotated

def diagonalLinesum(imagedata, cutRotatedImage = True):
    '''
    rotate an image by 45° and then perform the horizontal and vertical linesums.
    if cutRotatedImage == True, the rotated image is cut down to a rectangular part that contains only the original data,
    i.e. the zero padding is not relevant
    Only tested for square images.
    vertical and horizontal are not understood yet, i.e. which one is left bottom / right top and which one is the other one
    '''
    rotated = rotateImage45Deg(imageData, cutRotatedImage)
    lsv = np.sum(rotated, axis = 1)
    lsh = np.sum(rotated, axis = 0)
    return lsv, lsh

########################
### Fitting Functions
########################
def fitGaussian(data, x = [], p0 = None, fitMinimum = False):
    '''
    Data: actual data values
    x: x axis; if not given, the function just uses the indices (as with camera pixels)
    
    return: popt, errors
    popt: (amplitude, waist, xCenter, offset)
    '''
    if len(x) != len(data):
        x = np.linspace(0, len(data)-1, len(data))
    
    if p0 == None:
        #gaussian(x, amplitude, waist, xCenter, offset)
        p0 = [np.max(data) - np.min(data), 0.2 * (np.max(x) - np.min(x)), x[np.argmax(data)], np.min(data)]
        if fitMinimum:
            p0 = [np.min(data) - np.max(data), 0.2 * (np.max(x) - np.min(x)), x[np.argmin(data)], np.max(data)]
    popt, pcov = curve_fit(gaussian, x, data, p0 = p0)
    errors = [pcov[i][i]**0.5 for i in range(len(popt))]
    return popt, errors


########################
### CSV Preprocessing
########################
def extractVariableNames(s, existingVars = []):
    ''' Extract the variable names from the sequence name. Might not always be right '''
    r = s.split('_')
    converted = []
    isNumericArr = []
    #iterate over the split string: for each entry, decide if it is a number or not. variable names are always followed by numbers
    for i in range(len(r)):
        e = r[i]
        try:
            e = float(e)
            if i > 0:
                converted.append(e)
                isNumericArr.append(True)
        except:
            if i > 0:
                converted.append(e)
                isNumericArr.append(False)


    #Rebuild variable names
    variables = []
    current = ''
    for i in range(len(converted)):
        if not isNumericArr[i]:
            current += converted[i] + '_'
        else:
            if len(current) > 1:
                variables.append(current[:-1]) #remove last _ again; only when there really is a name
            current = ''

    return variables

def extractVariableValuesFromSingleString(s, variables):

    values = []

    # remove .png at the end of file name
    # string = s[:-4]

    for var in variables:

        # split off string right after variable name
        split = s.split("_" + var + "_")[1]

        # select number as first element of split
        value = split.split("_")[0]
        values.append(float(value))

    return values

def extractvariablesFromNames(nameslist, variables):
    ''' extract the values from the list of run names '''
    valuesMat = [[] for var in variables]
    for name in nameslist:
        vals = extractVariableValuesFromSingleString(name, variables)
        for i in range(len(variables)):
            valuesMat[i].append(vals[i])
    return valuesMat

def addForgottenColumns(df):
    ''' Try to extract the variable names and values that have not been added by camera control. Might often fail '''
    # extract the variable names contained in the sequence name.
    # Might be errorenous, i.e. part of the sequence name might be interpretated as variable name
    variableNames = extractVariableNames(df['Sequence'].loc[0])
    variables = [] # list of variables not contained in the df yet, but still potentially needed
    #check for all extracted variables if they are already present, and if not, if the name ends on "_i";
    #in the latter case, it most likely was not a real variable
    for var in variableNames:
        if var not in df.keys():
            if var[-2:] != '_i':
                variables.append(var)
    if len(variables) > 0:
        valuesMatrix = extractvariablesFromNames(df['Sequence'], variables)
        for i in range(len(variables)):
            print('extracted and added the variable', variables[i],'to the data')
            df[variables[i]] = valuesMatrix[i]
    return df

def dropUnusedColumns(df):
    ''' drop all columns that always contain the same data, as they are probably not interesting '''
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df

def removeDuplicateColumns(df):
    ''' remove duplicate columns from the dataframe '''
    # in case of duplicates: pandas adds ".1", ".2",... to the column name when the name already exists
    keylist = []
    toDrop = []
    # find all keys to drop (dont drop while iterating over the dataframe)
    for key in df.keys():
        if key.split('.')[0] in keylist:
            toDrop.append(key)
        else:
            keylist.append(key)
    #actually drop them (dont drop while iterating over the dataframe)
    for key in toDrop:
        df.drop(key, axis = 1, inplace = True)
    return df