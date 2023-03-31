# ==============================================================================
# This is a tool for automatically processing radiochromic film by Chris Williams
# 
# It is targetted at processing small film samples that are uniformly irradiated
# 
# 
# 
# 
# 
# ==============================================================================

import os

import pickle as pickle
import glob
import pathlib
import yaml
import re
import datetime

from scipy import misc
import scipy.ndimage as ndimage
from scipy import odr
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology as mph

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Tuple, List, Optional, Any, Type
import imageio.v3 as iio

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#if len(logger.handlers) == 0:
#    logger.addHandler(ch)

# %%
# ==============================================================================
# Settings
# The user should (and must for the program to work) change these
# variables to reflect the parameters of the measurement.
# Directories should point to existing directories
# All directories should be separated with a '/', not a '\'
# 
# The window or dynamic window must be chosen according to the image files
# that are to be processed. However, only one of those two options is necessary
# 
# For each measurement batch there are other settings that need to be entered
# in the form of an index.txt, which is located in the source directory along
# with the images. An example index file should be avaiable. If not, there is 
# a detailed description of how to write one in the indexing section.
# 
# 
# ==============================================================================


#
def load_image(fn: Union[str, bytes, os.PathLike]) -> np.array:
    """Load an image to a numpy array"""
    logging.debug('Loading file: %s' % fn)
    if not pathlib.Path(fn).exists():
        raise(FileNotFoundError('Could not load file: %s' % fn))
    im = cv2.imread(str(fn), -1)
    im = np.float32(im)

    return im[:, :, ::-1]


# %%
# Identify the film region in a scanned image
def make_film_mask(im):
    bim = im[:, :, 2]
    val = threshold_otsu(bim)
    m = (bim > 0) & (bim < val)
    m = mph.remove_small_holes(m)
    m = mph.remove_small_objects(m)
    m = ndimage.binary_erosion(m, iterations=8)
    #    if logging:
    #        plt.imshow(bim*~m)
    #        plt.show()
    return m


# %%
# Herein the ROI is found dynamically according to sorcerous calculations
# return bounds as two absolute pixel values
def make_roi(im: np.array, m: np.array, settings):
    """Choose a region of interest within a piece of irradiated radiochromic film for analysis based on the minimum
    pixel values (i.e. the darkest region on the radiochromic film strip)"""

    # Choose the region of interest based on the pixels falling within 1.05
    # of the central region, with an 8 pixel erosion
    rim = im[:, :, 0]

    buffer = 1
    ratio = 1.045
    ignore_pixels = 150

    pixels = np.argsort(rim[m])
    cutoff = rim[m][pixels[ignore_pixels]] * ratio
    t = rim < cutoff
    m2 = t & m
    m2 = mph.remove_small_holes(m2)
    m2 = ndimage.binary_erosion(m2, iterations=buffer)
    if settings['dynamic_show_roi']:
        plt.imshow(rim * ~m2)
        plt.show()
    if m2.any():
        return m2
    return m


# %%
def threshold(im, m):
    """
    Apply the Otsu threshold to, return a mask of the union of the original mask and the area below/above
    :param im:
    :param m:
    :return:
    """
    rim = im[:, :, 0]
    t_val = threshold_otsu(rim[m])
    t_mask = rim < t_val
    upper_mask = ~t_mask & m
    lower_mask = t_mask & m
    return lower_mask, upper_mask


# %%

def multi_threshold(im: np.ndarray, m: np.ndarray, mode: str = 'countmax') -> np.ndarray:
    rim = im[:, :, 0]

    # Threshold once
    l, u = threshold(im, m)
    # Threshold each thresholded image
    ll, lu = threshold(im, l)
    ul, uu = threshold(im, u)

    # Set up a dict containing a mask based on each threshold
    # Calculate some stats for each mask
    dat = {}
    a = [l, u, ll, lu, ul, uu]
    b = ['l', 'u', 'll', 'lu', 'ul', 'uu']
    for i in range(len(a)):
        dat[b[i]] = {}
        dat[b[i]]['m'] = a[i]
        dat[b[i]]['mean'] = rim[dat[b[i]]['m']].mean()
        dat[b[i]]['std'] = rim[dat[b[i]]['m']].std()
        dat[b[i]]['count'] = dat[b[i]]['m'].sum()

    # Create dataframe based on dict for further processing
    df = pd.DataFrame.from_dict(dat, orient='index')

    # countmax mode
    # Chooses from the secondary thresholds, returns the segment containing the
    # number of pixels. Requires that the bulk area of the image is the ROI
    # Should filter out both unexposed regions and scan overlap
    if mode == 'countmax':
        m = (df.index != 'l') & (df.index != 'u')
        mval = df[m]['count'].max()
        mask_out = df[df['count'] == mval].m.asobject[0]
        return mask_out
    else:
        print('No option selected, multi-threshold returning original mask')
        return m


# %%
def enumerate_mask_grid(m: np.ndarray, number_cols: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    # m: mask of particles in a grid
    # number_cols: number of columns in grid

    # Use the enumerate function to apply a number to each particle in the mask
    t, n = ndimage.measurements.label(m)

    # Find the central position of each enumerated particle in the mask
    particle_dict = {}
    for i in np.arange(n) + 1:
        yxargs = np.argwhere(t == i)
        y, x = (yxargs.max(axis=0) + yxargs.min(axis=0)) / 2
        particle_dict[i] = {'x': x, 'y': y}

    # Create a dataframe based on the particle positions
    df = pd.DataFrame.from_dict(particle_dict, orient='index')

    # Calculate number of rows, based on total number of particles and columns
    number_rows = int((n - (n % number_cols)) / number_cols + 1)

    # Reorder the index so that it is in numerical order, top to bottom
    df.sort_values('y', inplace=True)

    # Then within each row, create an index ordering left to right
    index = pd.Int64Index([])
    for i in range(int(number_rows)):
        l = i * number_cols
        u = (i + 1) * number_cols
        if u > n:
            u = n
        index = index.append(df[l:u].sort_values('x').index)
    df = df.reindex(index)
    df['N'] = np.arange(len(index)) + 1

    # Provide a plot of the xs, ys and labels if logging is enabled
    if logging:
        fig, ax = plt.subplots()
        ax.set_xbound(lower=0, upper=1200)
        ax.set_ybound(lower=0, upper=1200)
        for i in np.arange(n) + 1:
            ax.plot(df.x[i], df.y[i], 'o')
            ax.text(df.x[i], df.y[i], df.N[i])
        fig.show()

    # Rejigger the dataframe to have a label column, which corresponds to the
    # values in the enumerated mask, t
    df = df.sort_values('N').reset_index()
    df = df.rename(columns={'index': 'label'})
    if not df.N.all():
        print('Automatic file loading failed. Input settings')
    return df, t, n


# %%

def array_file_process(path: Union[os.PathLike, pathlib.Path, str], settings):
    # Process filename
    fn = path.split('\\')[-1]
    fn = path.split('/')[-1]
    fn = fn.split('.')[0]
    fn = fn.split('_')
    # Get batch name
    batch_name = fn[0]

    output = {}

    im = load_image(path)
    # background image in this case should just be a single large sheet of film.
    # Maybe we can work on using cut up film? Don't know. Sounds hard.
    bim = load_image(path.replace('after', 'before'))

    m = make_film_mask(im)
    df, t, n = enumerate_mask_grid(m, settings['array_mode_columns'])

    # Get after and before data from im and bim
    for i in df.index:
        strip_id = batch_name + '_' + "{0:0>3}".format(i + 1)
        m = (t == df.label[i])
        m = multi_threshold(im, m)
        m = make_roi(im, m, settings)
        output[strip_id] = {}
        output[strip_id]['mask'] = m
        output[strip_id]['mean_after'] = im[:, :, 0][m].mean()
        output[strip_id]['std_after'] = im[:, :, 0][m].std()
        output[strip_id]['mean_before'] = bim[:, :, 0][m].mean()
        output[strip_id]['std_before'] = bim[:, :, 0][m].std()
    return output


def file_process(path: Union[os.PathLike, str], settings: dict) -> dict:
    """
    Process an image file and compute the peak dose region, corresponding to the
    lowest pixel values within the area defined in the settings.

    :param path:
    :param settings:
    :return:
    """
    path = pathlib.Path(path)
    output = {}
    # Process filename
    fn = path.with_suffix('').name
    fn = fn.split('_')

    # Get date
    date = fn[0]
    # Get film number
    strip_no = fn[2]
    strip_id = fn[0] + '_' + fn[2]
    if not strip_id in output:
        output[strip_id] = {}
    output[strip_id]['date'] = date
    output[strip_id]['number'] = strip_no

    # Get the image up in here
    try:
        im = load_image(path)
    except:
        logger.error('Could not load image at %s' % path)
        return {}

    # Remove dead pixels.
    # im = remove_dead_pixels(im)

    # select the ROI using coordinates from settings file. Initially, lets choose the RED channel
    if settings['dynamic']:
        if fn[1] == 'after':
            mask = make_film_mask(im)
            mask = make_roi(im, mask, settings)
            output[strip_id]['mask'] = mask
        else:
            try:
                mask = output[strip_id]['mask']
            except KeyError:
                try:
                    mask = make_film_mask(im)
                    mask = make_roi(im, mask, settings)
                    output[strip_id]['mask'] = mask

                except: # too much excepting. do we need to make exceptions here? just fix the inputs! defaulting to manual seems like a bad idea.
                    mask = np.zeros(im.shape[:-1])
                    # Set all values in the 'manual roi' area to 1
                    roi = settings['manual_roi']
                    mask[roi[2]:roi[2] + roi[3], roi[0]:roi[0] + roi[1]] = 1
                    logger.error('Could not set window mode for %s - defaulting to manual', strip_id)

    else:
        mask = np.zeros(im.shape[:-1])
        # Set all values in the 'manual roi' area to 1
        roi = settings['manual_roi']
        mask[roi[2]:roi[2] + roi[3], roi[0]:roi[0] + roi[1]] = 1

    mean_val = im[:, :, 0][mask].mean()
    std_val = im[:, :, 0][mask].std()

    output[strip_id]['mean_' + fn[1]] = mean_val
    output[strip_id]['std_' + fn[1]] = std_val

    # return the updated dictionary
    return output


# Process the results dictionary
# at the moment, just iterate through and find change in pixval and overall std
def process_results(results):
    for film_id in results:
        try:
            results[film_id]['delta'] = results[film_id]['mean_before'] - results[film_id]['mean_after']
            results[film_id]['std'] = np.sqrt(
                np.square(results[film_id]['std_before']) + np.square(results[film_id]['std_after']))
        except KeyError:
            logger.warning('Unpaired before/after image found for film id %s' % film_id)
            # insert code for case where no before/after image exists
            # print('Film ID '+ film_id + ' does not have before scan. Trying strip 001')
            try:
                results[film_id]['delta'] = results[film_id[:-3] + '001']['mean_before'] - results[film_id][
                    'mean_after']
                results[film_id]['std'] = np.sqrt(
                    np.square(results[film_id[:-3] + '001']['std_before']) + np.square(results[film_id]['std_after']))
            except Exception as e:
                logger.error(e)
                logger.error(
                    'Film ID ' + film_id + ' does not have before scan and could not be processed using strip 001 from batch')
    return results


# %%


# ==============================================================================
# Open the index file and make a list of lists out of the contents
# Note: index file must have the following format:
# line 1: measurement_type,batch_name
# line 2+:exposure_id, film_id, film_id,...,
# 
# For example, a calibration measurement:
# 1 calibration,120kVpCT_0205021
# 2 103.2mGy,0315_001,0315_002,0315_003
# 3 221.4mGy,0315_004,0315_005,0315_006,0305_007,0305_008
# ...
# 
# or for a measurement example
# 
# 1 measurement,120kVpCT_0205021
# 2 chest_1,80kVp,0418_001,0418_002
# 3 chest_2,120kVp,0418_003,0418_004
# ...
# 
# 
# Notes:
#  - measurement_type is either calibration, for an initial calibration
#    measurement which will be saved as the calibration_name, or measurement for 
#    any subsequent measurements using the calibration_name calibration function
#  - For measurements, the first line will be a description rather than a dose
#    e.g. 
#  - each measurement point can have an arbitrary number of film strips associated
# ==============================================================================

def get_index(fn: Union[os.PathLike, str]) -> list:
    """
    Open the index file and make a list of lists out of the contents
    :param fn: The file containing the index data
    :return: a list containing the content of hte index file
    """
    with open(fn) as f:
        content = f.readlines()
        # Strip trailing nonsense
    content = [x.strip().split(',') for x in content]
    content = [list(filter(None, c)) for c in content]
    content = list(filter(None, content))
    return content


# Calculate the weighted mean of the sample set, based on the relative uncertainties as well as the overall uncertainty

def combine_samples(film_id_list: list, results: dict) -> Tuple[float, float]:
    """
    Lookup the film ids given in the list of results. Combine the results and produce a single
    mean dose and measurement uncertainty.

    :param film_id_list: list of film ids, which are defined
    :param results: dictionary of processed film results
    :return: tuple containing the mean dose and overall uncertainty
    """
    if len(film_id_list) == 0:
        raise (ValueError('Film ids not supplied'))
    total_mean = 0
    std_sum_inv_squares = 0
    std_sum_inv = 0
    scanner_mean = []
    for film_id in film_id_list:
        std = results[film_id]['std']
        mean = results[film_id]['delta']
        std_sum_inv_squares += 1 / np.square(std)
        std_sum_inv += 1 / std
        total_mean += mean / np.square(std)
        scanner_mean.append(mean)
        scanner_std = np.std(scanner_mean)
    pixel_std = np.sqrt(1 / std_sum_inv)
    total_mean = total_mean / std_sum_inv_squares
    total_std = np.sqrt(np.square(pixel_std) + np.square(scanner_std))
    if len(film_id_list) == 1:
        total_std += total_mean * .05
    return total_mean, total_std


# Iterate through the index and return the weighted mean and total uncertainty for each sample set
def index_samples(results: dict, measurement_folder: Union[os.PathLike, str], settings) -> Tuple:
    """
    Given a set of individual film strip measurement results, combine similar measurements
    in order to evaluate uncertainty.
    
    Should have used pandas.

    :param results: The processed results of film strips
    :param measurement_folder: The folder containing the images and the index for the measurement set
    :return: tuple containing a dataframe with results, the batch name, and calibration objects
    """
    measurement_folder = pathlib.Path(measurement_folder)
    logger.debug('Indexing measurements in %s' % measurement_folder)
    index = get_index(measurement_folder / 'index.txt')
    logger.debug(index)

    labels = []
    data = []
    index_header = index.pop(0)
    batch_type = index_header[0]
    batch_name = index_header[1]

    if batch_type not in ['calibration', 'measurement']:
        logger.error(batch_type + 'is not a valid operation. Typo?!')

    cal_list = []

    for i in index:
        measurement_id = i[0]
        if batch_type == 'measurement':
            cal_list.append(i.pop(1))
        else:
            measurement_id = extract_single_float(measurement_id)
        file_id_list = i[1:]
        mean, std = combine_samples(file_id_list, results)
        labels.append(measurement_id.split())
        data.append([measurement_id, mean / 2**settings['bit_depth'], std / 2**settings['bit_depth']])
    if batch_type == 'calibration':

        data = np.array(data).astype(float)
        cal = make_calibration(data, settings['calibration_folder'], batch_name)
        doses = None
    else:
        cal, doses, doseerr = apply_calibration(data, cal_list, settings['calibration_folder'])
        
    output_data = pd.DataFrame(data)
    output_data.columns = ['measurement_id', 'delta_R','delta_R_uncertainty']
    output_data['batch_id'] = batch_name
    output_data = output_data.iloc[:,[-1, 0,1,2]]
    
    if batch_type != 'calibration':
        output_data['measured_dose'] = doses
        output_data['measured_dose_uncertainty'] = doseerr

    return output_data, batch_name, batch_type, cal


def make_calibration(data: np.array,
                     calibration_folder: Union[os.PathLike, str],
                     calibration_name: str
                      ) -> object:

        # create calibration function from data
    cal = Calibration(calibration_name, calibration_folder,
                      R=data[:, 1], D=data[:, 0], sigma=data[:, 2])
    return cal


def apply_calibration(data: np.array,
                      cal_list: list = None,
                      calibration_folder: Union[os.PathLike, str] = ''
                      ) -> Tuple[Union[Any, List[Any]], np.ndarray, np.ndarray]:
    """
    Apply a calibration function to a set of data which includes measured reflectance values

    :param data: list of lists. each sublist contains: name, mean pixel value, std pixel value
    :param cal_list: list of the names of the calibration function for each item in data
    :param calibration_folder: The folder used where the calibration results are stored
    :return: tuple containing: calibration functions used, list of doses, a list of uncertainties
    """

    cal = []
    doses = []
    doseerr = []
    for i, R in enumerate(data):
        c = Calibration(cal_list[i], calibration_folder)
        cal.append(c)
        doses.append(c.get_dose(R[1]))
        doseerr.append(c.get_total_err(R[1], R[2]))
    return cal, doses, doseerr

# %%
# Dead pixel functions

# Remove dead pixels
# Refers to previously calculated array, saved in a text file in a certain directory.
# Not currently working =[
def remove_dead_pixels(im, pixel_list_fn):
    dead_pixel_list = np.loadtxt(pixel_list_fn, delimiter=',')
    # Choose an averaging filter for all 8 adjacent pixels
    my_filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # (essentially) convolve the image with my filter in order to create an averaged image
    mean_im = ndimage.generic_filter(im, np.mean, footprint=my_filter)

    # iterate over all dead pixels. For each one, replace the same pixel in the orginal image with the equivalent in the averaged image
    for row in dead_pixel_list:
        x, y = row
        im[x, y] = mean_im[x, y]

    return im


# Find dead pixels from background image
def get_dead_pixels(fn, pixel_list_fn):
    im = load_image(fn)

    # todo: lets make this a LOT more nuanced
    mean = im.mean
    dead_pixel_list = np.argwhere(im < 255 * .9)
    np.savetxt(pixel_list_fn, dead_pixel_list, delimiter=',', fmt='%s')
    return dead_pixel_list


# %%
# Window an image based on a rectangle
# rect =

def crop_image(image, rectangle):
    x, dx, y, dy = rectangle
    cropped_image = image[y:y + dy, x:x + dx, 0]
    return cropped_image


# Test window, showing before and after images

def show_window(image, rect, mode='relative'):
    if mode == 'relative':
        x, dx, y, dy = rect
    elif mode == 'absolute':
        x, x2, y, y2 = rect
        dx = x2 - x
        dy = y2 - y
    else:
        raise (ValueError('Acceptable mode not supplied. Modes are "relative" and "absolute"'))
    im_windowed = np.copy(image)
    im_windowed[y:y + dy, x:x + dx, :] = 1
    im_temp = misc.toimage(im_windowed)
    plt.imshow(im_temp)

    return im_temp


# %%
# Take an image and a calibration function then turn it into a dose map
def make_dosemap(im_after, im_before, cal, settings, unit = 1000):
    logger.debug(cal)
    diff_im = (im_before-im_after)[:,:,0] / 2**settings['bit_depth']
    diff_im[diff_im < 0] = 0
    dose_im = cal.get_dose(diff_im)
    dose_im[dose_im<0] = 0
    return dose_im


def save_dosemap(im, ffn):
    # now take the dosemap and save it to a file
    
    iio.imwrite(ffn.with_suffix('.tif'), im.astype('float32'))
    


# misc.toimage(test, cmin=0, cmax=255,mode='I').save("tmp.png")

def folder_to_dosemaps(path, calibration_folder, settings, unit=1000):
    path = pathlib.Path(path)
    index = get_index(pathlib.Path(path) / 'index.txt')
    index_header = index.pop(0)
    if index_header[0] == 'calibration':
        return
    
    try:
        batch_name = index_header[1]
    except IndexError:
        batch_name = path.name
        
    output_folder = pathlib.Path(settings['output_folder']) / 'dose_maps' / batch_name
    for line in index:

        if index_header[0] == 'measurement':
            measurement_id = line.pop(0)
        cal_name = line.pop(0)
        cal = Calibration(cal_name, calibration_folder)
        for film_id in line:
            dose_image = file_to_dosemap(path, film_id + '.tif', cal, settings, unit)


def file_to_dosemap(path, fn, cal, settings, unit=1000):
    path = pathlib.Path(path)
    

    #except:
    #    logger.error('error with file %s' % (path / fn))
        
    # Load background image
    # Try before file
    after_fn = fn.replace('_','_after_')
    after_im = load_image(path / after_fn)
    before_fn = fn.replace('_','_before_')
    # Load the background 'before' image associated with this file id
    if (path / before_fn).exists():
        before_im = load_image(path / before_fn)
    #otherwise, try loading the default background image for the measurement set, if it exists
    elif (path / 'default_background.tif').exists():
        logger.info('Default batch background image used for %s' % (path / fn))
        before_fn = path / 'default_background.tif'
        before_im = load_image(before_fn)
    # otherwise, try loading the first before measurement from the set, if it exists 
    #elif path / (backfnos.path.exists(backfn[:-6] + '01' + backfn[-4:]):
    #    logger.info('Default batch background image used for %s' % (path / fn))
    #    backfn = backfn[:-6] + '01' + backfn[-4:]
    #    imback = load_image(path / backfn)
    # default to using the backup background pixel value. accuracy is reduced in this case
    else:
        logger.warning('Backup backrgound pixel value used for %s' % (path / fn))
        before_im = np.array([settings['backup_background_pixel_value']])  # manual backup value for average measured
 #   except:
#        logger.error('Could not load before image for dosemap' + fn)
    dose_image = make_dosemap(after_im, before_im, cal, settings, unit)
    output_folder = pathlib.Path(settings['output_folder']) / 'dose_maps' / path.name
                                 
    os.makedirs(output_folder, exist_ok=True)
    save_dosemap(dose_image, output_folder / (fn.replace('_after','').replace('.tif','_dosemap.tif')))
    return dose_image

# %%

def crop_to_film(im):
    # this needs to have code that crops an image file to the section that contains film.
    return im


# %%
# This class allows the creation of a calibration function.
# By calling get_dose(R), it is possible to manually check the dose for a specific strip

# class functions, it is possible to apply the generated function to 

class Calibration:
    name: str = ''
    calibration_folder: Union[os.PathLike, str] = None
    D: np.array = None
    R: np.array = None
    sigma: np.array = None
    a: float = None
    b: float = None
    fit_cov = None
    p = None
    

    # Initiate the calibration
    # Requires the name parameter, which represents the calibration function
    # If the name has previously been used, loads the calibration function for that name
    # If keywords in the form of R,D and sigma are passed
    # create a new calibration function overwriting the existing one
    def __init__(self, name, calibration_folder, **kwargs):
        self.name = name
        self.calibration_folder = calibration_folder
        if 'R' in kwargs:
            self.new_calibration(**kwargs)
        elif not name == '':
            self.load_calibration()
        else:
            print('No calibration function was initiated')

    # Create a new function using key word arguments of the form: R = Reflectionvals, D=Dosevals,sigma=uncertaintyvals
    # If any previous calibration function existed with this name, it will be overwritten.
    def new_calibration(self, **kwargs):
        self.D = kwargs['D']
        self.R = kwargs['R']
        self.sigma = kwargs['sigma']
        if kwargs['D'][0] == 0:
            kwargs['D'] = kwargs['D'][1:]
            kwargs['R'] = kwargs['R'][1:]
            kwargs['sigma'] = kwargs['sigma'][1:]
        linear = odr.Model(self.fit_function)
        mydata = odr.RealData(kwargs['R'], kwargs['D'], sx=kwargs['sigma'])
        myodr = odr.ODR(mydata, linear, beta0=[60, -2])
        myoutput = myodr.run()
        cov = myoutput.cov_beta
        sd = myoutput.sd_beta
        self.p = myoutput.beta
        self.a = self.p[0]
        self.b = self.p[1]
        self.fit_cov = cov
        self.fit_sd = sd
        # [self.a,self.b],self.fit_cov = curve_fit(self.fit_function,kwargs['R'],kwargs['D'],sigma=kwargs['sigma'],)
        print(self.a)
        print(self.b)
        if not self.name == '':
            self.save_calibration()

    # This returns a function based on the variables R and the constants a and b.
    # By changing the form of this function, the form of the fit can be altered
    # Several alternative fit options have been included but commented out
    def fit_function(self, A, R):
        a = A[0]
        b = A[1]
        # Rational function. Looks like the best fit so far
        return a * R / (1 + b * R)

    # Log function. This is quite bad
    #    return a+b*R/np.log(R)
    # Exponential function. ok fit
    #        return a*R*np.exp(b*R)

    # Rational function with a power argument. For science.
    #        return a*R/(1+b*R)
    # something funky
    #        return a*np.power(R,b)/(1-np.log(R*c))

    # Calling this function returns the dose for a given reflectance, according to the calibration function.
    def get_dose(self, R):
        return self.fit_function([self.a, self.b], R)

    def get_a_err(self, R):
        return self.fit_sd[0] / self.a

    def get_b_err(self, R):
        return R * self.fit_sd[1] / (1 + self.b * R)

    def get_fit_err(self, R):
        return np.sqrt(self.get_a_err(R) ** 2 + self.get_b_err(R) ** 2)

    def get_exp_err(self, R, sigma):
        return sigma / R / (1 + self.b * R)

    def get_total_err(self, R, sigma):
        return np.sqrt(self.get_a_err(R) ** 2 + self.get_b_err(R) ** 2 + self.get_exp_err(R, sigma) ** 2)

    # Save the calibration function to disk.
    # Todo not robust
    def save_calibration(self):
        path = pathlib.Path(self.calibration_folder)
        if not path.exists():
            os.makedirs(path)
        with open(path/(self.name+'.p'), "wb") as f:
            pickle.dump([self.a, self.b, self.fit_cov, self.R, self.D, self.sigma, self.fit_sd],
                    f)

    #        np.savetxt(home+'dat/cali/fit/'+name+'.csv', dead_pixel_list, delimiter=',',fmt='%s')

    # Load a calibration function from a disk
    # todo not robust
    def load_calibration(self):
        path = pathlib.Path(self.calibration_folder)
        try:
            with open(path / (self.name + '.p'), "rb") as f:
                [self.a, self.b, self.fit_cov, self.R, self.D, self.sigma, self.fit_sd] = pickle.load(f)
        except:
            print('Could not load calibration with name ' + self.name)

    # Show the calibration function in the inline spyder window
    # todo save plot to output folder
    def show_calibration(self, ax=None, **kwargs):

        R = np.linspace(0.01, 0.48, 90)
        if not ax:
            fig, ax = plt.subplots()
        curve, = ax.plot(R, self.fit_function([self.a, self.b], R), label=self.name, zorder=1)
        col = curve.get_color()
        ax.errorbar(self.R, self.D, xerr=self.sigma, yerr=self.D * 0.03, color=col, marker='o', linestyle='none',
                     markersize=2, capsize=2, zorder=2)
        ax.set_xlim([0, 0.45])
        ax.set_ylim([0,400])
        ax.set_ylabel('Dose (mGy)')
        ax.set_xlabel(r'$\Delta R$')
        return ax

    def show_fit_uncertainty(self, ax=None, **kwargs):
        R = np.linspace(0.00, 0.45, 90)
        if ax is None:
            fig, ax = plt.subplots()
        curve, = ax.plot(R, self.get_fit_err(R) * 100, label='Fit uncertainty', zorder=1)
        col = curve.get_color()
        # plt.plot(self.R,self.get_exp_err(self.R),color = col, marker ='o',linestyle = 'none',markersize = 2,zorder=2)
        ax.plot(self.R, self.get_exp_err(self.R, self.sigma) * 100, 'x', label='Experimental uncertainty', color=col)

        ax.plot(self.R, self.get_total_err(self.R, self.sigma) * 100, '.', label='Total uncertainty', color=col)
        ax.set_xlim([0, 0.45])
        ax.set_ylim([0, 12])
        ax.set_ylabel(r'Uncertainty ($\sigma/D$%)')
        ax.set_xlabel(r'$\Delta R$')
        return ax, curve
    
    
    def __repr__(self):
        return f"Calibration - {self.name}. a={self.a} b={self.b}"


def process_measurements_single(folder: Union[os.PathLike, pathlib.Path, str],
                                settings) -> Tuple:
    """
    Assume all Process all .tif or .tiff image files in the supplied folder.

    :param settings: settings dictionary for the measurement being performed
    :param folder: Folder containing scanned image of radiochromic film and an index file
    :return: tuple containing results
    """
    folder = pathlib.Path(folder)
    filenames = folder.glob('*.tif*')
    results = {}
    for fn in filenames:
        if not (('after' in fn.name) or ('before' in fn.name)):
            logger.info('Found a tif file not conforming to the naming convention xyzz_[before/after]_00x.tif: %s' % fn)
            continue
        logger.debug("Processing %s" % fn)
        result = file_process(fn, settings)
        measurement_id = list(result.keys())[0]
        if measurement_id in results:
            results[measurement_id].update(result[measurement_id])
        else:
            results.update(result)
    results = process_results(results)
    result_data, batch_name, batch_type, cal_functions = index_samples(results, folder, settings)
    return result_data, batch_name, batch_type, cal_functions


def process_measurements_array(path, fn, settings):
    results = {}
    array_file_process(fn, results)
    results = process_results(results)
    result_data, batch_name, batch_type, cal_functions = index_samples(results, path, settings)
    return result_data, batch_name, batch_type, cal_functions


# %%
def folder_to_measurements(path: Union[os.PathLike, str], settings) -> Tuple:
    index = get_index(path + '/index.txt')
    measurement_type = index[0][0]
    batch_name = index[0][1]

    try:
        array_fn = index[0][2]
    except:
        logger.info('array_mode not set, defaulting to single')
        array_fn = 'single'

    if array_fn == 'single':
        result_data, batch_name, batch_type, cal_functions = process_measurements_single(path, settings)
    else:  # Array mode engaged:
        raise(NotImplementedError('Array mode has not been fully implemented.'))
        result_data, batch_name, batch_type, cal_functions = process_measurements_array(path, array_fn, settings)
    
    # Save the results to csv
    csv_output_dir = pathlib.Path(settings['output_folder']) / 'data'
    if not csv_output_dir.exists():
        os.makedirs(csv_output_dir)
    result_data.to_csv(csv_output_dir / (batch_name + '.csv'), index=False)

    return result_data, batch_name, batch_type, cal_functions


# %%

def process_folder(path, settings):
    index = get_index(path + '/index.txt')
    measurement_type = index[0][0]

    if settings['make_dose_images'] not in ['measure_and_maps', 'only_measure', 'only_maps']:
        raise(ValueError('measure_and_maps, only_measure, only_maps are acceptable settings for make_dose_images'))
    if (settings['make_dose_images'] != 'only_measure')  and (measurement_type!='calibration'):
        folder_to_dosemaps(path, settings['calibration_folder'], settings)
    if (settings['make_dose_images'] != 'only_maps') and (measurement_type!='dosemap'):
        results = folder_to_measurements(path, settings)
        return results


def plot_calibrations(all_results, settings):
    
    calibration_folder = pathlib.Path(settings['calibration_folder'])
    os.makedirs(calibration_folder / 'plot', exist_ok=True)
    
    calibration_results = []
    for result in all_results:
        batch_type = result[2]
        if batch_type == 'calibration':
            calibration_results.append(result)
    if len(calibration_results) == 0:
        return
    
    collected_calibration_name = ' '.join([c[1] for c in calibration_results])

            
    fig, ax = plt.subplots()
    for result in calibration_results:
        result[3].show_calibration(ax)
        plt.legend(loc=2)
    fig.savefig(calibration_folder / 'plot' / (collected_calibration_name+'calibration_curves.png'), format='png', dpi=600)
    fig.show()
    
    
    fig, ax = plt.subplots()
    labels = []
    handles = []
    for result in calibration_results:
        cal_name = result[1]
        ax, curve = result[
            3].show_fit_uncertainty(ax)  # x=data[0:,1],y=data[:,0],xerr = data[:,2],yerr = 0.05*data[:,0] what were all these arguments even doing in this call?
        labels.append(cal_name)
        handles.append(curve)

    line = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')
    cross = plt.Line2D((0, 1), (0, 0), color='k', marker='x', linestyle='')
    plus = plt.Line2D((0, 1), (0, 0), color='k', marker='.', linestyle='')

    fig.legend([handle for i, handle in enumerate(handles)] + [line, cross, plus],
               [label for i, label in enumerate(labels)] + ['Fit uncertainty', 'Exp. uncertainty', 'Total uncertainty'])
    
    fig.savefig(calibration_folder / 'plot' / (collected_calibration_name+'calibration_uncertainty.png'), format='png', dpi=600)

    #plt.savefig('cal_unc.eps', format='eps', dpi=600)
    fig.show()

# %%
# Goes through and shows a random strip window location. For proof checking of
# the dynamic window function
def test_windows(results, settings, im_fn):
    for strip_id in results:
        fn = strip_id[0:4] + '_before' + strip_id[4:] + '.tif'
        im = load_image(im_fn)
        bounds = results[strip_id]['window']
        rectangle = [settings['dynamic_x_range'][0], settings['dynamic_x_range'][1],
                     bounds[0], bounds[1]]
        show_window(im, rectangle, 'absolute')


def extract_single_float(x):
    return re.sub('[^\d|\.]','', x)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def load_yml_settings(fn: Union[os.PathLike, str]) -> dict:
    try:
        with open(fn, 'r') as f:
            settings = yaml.load(f, yaml.SafeLoader)
    except FileNotFoundError as e:
        return {}
    return settings


def process_multiple_folders(folders, settings):
    all_results = []

    for folder in folders:
        result = process_folder(folder, settings)
        if result is not None:
            all_results.append(result)
    if settings['plot_calibration_results']:
        plot_calibrations(all_results, settings)
    if settings['single_csv_output']:
        if len(all_results) > 0:
            output = pd.concat([r[0] for r in all_results], axis=0)
            output.to_csv(
                pathlib.Path(settings['output_folder']) / 'data' /
                ('merged_data_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
            )
    return all_results


def run_script(args):
    # Load settings from the yml file
    with open(args.yml_input, 'r') as f:
        settings = yaml.load(f, yaml.SafeLoader)

    # Override the yml if the command line args were used
    for arg in vars(args):
        if getattr(args, arg):
            settings[arg] = getattr(args, arg)

    logger.debug(str(settings))

    all_results = process_multiple_folders(settings['input_folders'], settings)
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='RadiochromicFilmProcessor',
        description='Processes individual strips of radiochromic film. Manages film calibrations')
    parser.add_argument('yml_input', default='film_process.yml', nargs='*')
    parser.add_argument('--calibration_folder', required=False)
    parser.add_argument('--input_folders', required=False)
    parser.add_argument('--output_folder', required=False)
    parser.add_argument('--bit_depth', required=False)
    parser.add_argument('--dynamic_position', required=False)
    parser.add_argument('--dose_images', required=False)
    parser.add_argument('--single_csv_output', required=False)
    
    args = parser.parse_args()

    out = run_script(args)
