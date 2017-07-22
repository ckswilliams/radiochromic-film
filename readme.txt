This is a python script used for processing scanned images of radiochromic film.
Written by Chris Williams


To use from exe, requires:
Windows

To use from source, requires:
Python version 3.6+
basic imports: sys, os, pickle, glob
scipy,skimage,numpy,matplotlib,PIL,cv2

Tested using winpython 3.6, available  athttps://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.1.0/WinPython-64bit-3.6.1.0Qt5.exe/download
Requires cv2
 - Download cv2 package from http://www.lfd.uci.edu/~gohlke/pythonlibs/
 - opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl
 - Open winpython control panel


Glossary
A batch of film samples describes all film that needs to be processed in one sitting, or the film necessary to create a single calibration. A batch of film should consist of strips having the same size, located in the same place within the images.
A set of measurements refers to a single measurement point, which can consist of 1+ pieces of film exposed using the same conditions.


Usage of this script requires the following:
 - All film scanned at the same position within the image
  > use a template for film placement in scanner
 - For each batch of film samples with the same size, at least one image before exposure
  > Additional dosimetric accuracy can be achieved by scanning each piece of film before expsoure
 - Image of each piece of film scanned after exposure
 - Each batch of images should be in a unique directory
 - The file naming convention must follow the guide below
 - A index.txt or mapindex.txt must exist in the same directory, describing the images
  > These can be created by referring to the sections below or templates provided in the sample directory.

Image filenames:

The following image naming convention is mandatory:
[batch]_[after/before]_[number id]

eg:
testbatch_after_001 would be an image of the first film piece after exposure.
testbatch_before_001 would be the first film piece before exposure.




Calibration

The program can be used to calibrate film by exposing a batch to a series of known doses.
The calibration index.txt file must detail which images correspond to each measurement.
The dose each group of 
For calbiration, the dose levels should follow an approximately geometric pattern covering the necessary dynamic range for any measurements required.
Expose at least 1, preferably 3 pieces of film at each level.
The calibration function created will consist of 2 parameters and their corresponding uncertainty.
The form of the calibration is Dose = a * Delta R / (1 + b * Delta R)
Where Delta R is the change in reflectance (pixel value before - pixel value after)
a & b are the fitted parameters.
For calibration and uncertainty calculation discussions, please see thesis chapter 4
todo: link
After running the script to calibrate, the function will be saved as cal/name according to the name provided in the index.txt file.
Saved calibrations can be used to perform measurements and create dose maps.


Measuring an unknown dose

The program can be used to measure an unknown dose using previously measured calibration values.
The index.txt file must detail a name and calibration function for each measurement set
The results of the measurement will be dumped to an excel spreadsheet in the output directory.
  
  

Creating a dosemap

The script can be used to convert exposed images to a dose image, using previously calbirated values.
Requires a mapindex.txt file which details which calbiration function should be used for each image.
The resulting images will be placed in a subdirectory of the images used to create them.
The default unit of the dosemap images is microGray - this will lead to clipping above 65536 uGy.
When measuring above 65536 uGy, the multiplicitive factor should be set in dosemap.txt.
Manually adding 100 to this will decrease the values output by a factor of 10.

  
index.txt
The appropriate method for creating index files is to use the template index.txt provided.
The following provides a detailed description of how these files work, should it prove necessary.
Note that the 'before' or 'after' part of the image filenames is omitted in index.txt.

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
#line2+: 	[dose]mGy,film_id,film_id
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


mapindex.txt

example file:
Note: for mapindex.txt, the 'after' part of the filename must be included
#==============================================================================
#dosemap
#120kVp,0304_after_022,0304_after_023
#80kVp,0304_after_024,0304_after_025
#100kVp,0304_after_026,0304_after_027
#140kVp,0304_after_028,0304_after_029
#==============================================================================

