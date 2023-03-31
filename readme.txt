This is a python script used for processing scanned images of radiochromic film.
Written by Chris Williams


requires:
Python version 3.8+
requires:
 - scipy
 - skimage
 - numpy
 - matplotlib
 - PIL
 - opencv
 - 

Key functions:
 - Calibration: generate a calibration curve from a series of film measurements
 - Measurement: measure radiation dose from film strips
 - Produce dose maps: allow qualitative assessment/manual inspection of dose within the irradiated region
Each batch of measurements must conform to one of the above functions.


Glossary
Batch - A batch of film samples is a grouping of film.
      - Calibration functions limit the contents of each batch (one calibration curve per batch)
      - Otherwise, batches are just a convenient method for collecting measurements in a sub-group
      - Ideally, each batch consists of images with the same resolution. However, this is not strictly necessary.
Set - A set of measurements refers to a single measurement point
    - This consists of 1+ pieces of film exposed using the same conditions
    - Measurement sets greater than 1 film strip are used to assess precision/uncertainty


How to perform film measurements and scan film strips:
 - Images must be scanned into .tif format with a digital scanner
 - Images should preferably be 16 bit depth. If another image depth is used, the yml settings file must be updated
 - All film should be positioned at the exact same position within the image
  > use a template for film placement in scanner
 - Each image should contain one piece of radiochromic film
 - Ensure any image processing effects are disabled
 - Scan each piece of film before and after exposure for the best achievable accuracy
  > Ensure each scan field of view is the same size for the before/after images. Difference in image resolution will break the script.
  > At the very least, scan a single sample piece of film before and after exposure
 - Repeat each measurement x3, so 6 scanned images per measurement
  > Skipping this step significantly reduces the ability to estimate measurement uncertainty
 - Expose each film strip, preferably to a uniform radiation exposure
  > Horizontally uniform radiation exposures are fine in 'dynamic' mode.

  
Setting up the folder structure:
 - An example folder structure is included in the github repository.
 - The example folder structure can just be used as-is
 - An example yml setting file is included named 'film_process.yml'
 - The default action of the script is to load 'film_process.yml' from the working directory
 - The example yml setting file default behavior is to process the contents of the example folder structure
 - Each measurement batch must be defined in the yml settings file used to call the script 
 - Each measurement batch must have before/after images of each measurement, and an index.txt file which describes the measurement
 
 
File naming convention
THE SCRIPT WILL NOT PROCESS FILES WHICH DEVIATE FROM THIS NAMING CONVENTION
image names
[batch]_[after/before]_[number id].tif
'default_background.tif' can optionally be included in each directory.
If 'before' images are unavailable, the default background image will be used as the before image instead
eg:
testbatch_after_001.tif would be an image of the first film piece after exposure.
testbatch_before_001.tif would be the first film piece before exposure.
in the index, this before/after pair would have the film_id 'testbatch_001'


How to use the script:
 - Each folder must contain a measurement batch.
  > Calibration type batches must contain data for a single calibration curve
  > e.g. to have a calibration curve at 4 different kVp settings, 4 different measurement batches are required.
 - There is no strict limit or constraint on how many items/measurements can be included in a measurement or dosemap measurement batch
 - File names must strictly match the format provided.
 - An index.txt file must exist in the same directory, describing the images. See below.
 - a yml file describing the settings for the analysis must be created.
 - the script is then called with the yml file as the argument


In addition, each batch requires an 'index.txt' file
Optionally, each batch can have a 'default_background.tif' file, typically required for cases when the 'before' images were lost or forgotten.


Calibration function
 - The program can be used to calibrate film by exposing a batch to a series of known doses.
 - The calibration index.txt file must detail which images correspond to each measurement.
 - Each measurement batch can only contain a single calibration curve
 - Film strips should be exposed across the necessary dynamic range for any measurements which will be needed
 - Expose at least 1, preferably 3 pieces of film at each dose level.
 - The calibration function created will consist of 2 parameters and their corresponding uncertainty.
 - The form of the calibration is Dose = a * Delta R / (1 + b * Delta R)
 - Where Delta R is the change in reflectance (pixel value before - pixel value after)/2^bit_depth
 - a & b are the fitted parameters.
 - For calibration and uncertainty calculation discussions, please see thesis chapter 4
 - After running the script to calibrate, the calibration will be saved in [output_folder]/cal/[batch_name].p
 - Plots of the calibrations will be saved in [output_folder]/cal/plot/[batch_name].png (if the associated setting is enabled)
  > Saved calibrations can be used to perform measurements and create dose maps


Measurement function

 - The program can be used to measure an unknown dose using previously established calibration.
 - The index.txt file must detail a name and calibration function for each measurement set
 - The results of the measurement will be dumped to a csv file in [output_folder]/data/[batch_name].csv
  

Dosemap function

 - The script can be used to convert exposed images to a dose image, using a previously established calibration function.
 - The resulting images will be placed in [output_folder]/dose_maps/[batch_name]/xx_xx.tif
 - The images can be loaded in Image J or with python imageio
 - The default unit of the dosemap images is mGy


index.txt
 - The index.txt file details the contents of a given directory.
 - Each measurement batch must contain index.txt
 - No output will be created for files that aren't mentioned in index.txt
 - Accurately filling out the index.txt file is necessary for measurement
 - Example index.txt files are available for all 3 measurement types
 - The following provides a detailed description of how these files work, should it prove necessary.
 - Note that the 'before' or 'after' part of the image filenames is omitted in index.txt.

#==============================================================================
# Open the index file and make a list of lists out of the contents
# The header of the index file must have the following format:
#
# line 1: measurement_type,batch_name
#
# Valid measurement types are 'measurement' and 'calibration'
#The subsequent lines will be different for calibration and measurement files:
#
#Calibration:
#Each set requires one line
#line2+: 	[dose]mGy,film_id_1,film_id_2,film_id_3,...,
#Film id is [batch]_[number id]
#eg: 		103.2mGy,0315_001,0315_002,0315_003
#Where 0315 is the batch name, and the set includes film pieces 001, 002 and 003.
#
# 
# A complete example calibration index, including 3 measurements:
# 1 calibration,120kVpCT_0205021
# 2 103.2mGy,0315_001,0315_002,0315_003
# 3 221.4mGy,0315_004,0315_005,0315_006,0305_007,0305_008 
# ...
# 
#Measurement:
#Each set requires one line
##line2+: 	[measurement name],[calibration name],film_id,film_id,...
#
# Measurement example:
# 
# 1 measurement,120kVpCT_0205021
# 2 chest_1,80kVp,0418_001,0418_002
# 3 chest_2,120kVp,0418_003,0418_004,0418_004
#
#Notes: The number of strips in each set can be 1+
#NO TRAILING WHITESPACE note to self: ignore trailing whitespace
#==============================================================================


index.txt folders which are only for dose maps:

example file:
Note: for index.txt, the 'after' part of the filename must be included
#==============================================================================
#dosemap
#120kVp,0304_022,0304_023
#80kVp,0304_024,0304_025
#100kVp,0304_026,0304_027
#140kVp,0304_028,0304_029
#==============================================================================

