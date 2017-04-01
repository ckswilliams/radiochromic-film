#==============================================================================
# This is a tool for automatically processing radiochromic film by Chris Williams
# 
# It is targetted at processing small film samples that are uniformly irradiated
# 
# 
# 
# 
# 
#==============================================================================



#Let's get some imports up in here

#System imports
import sys
import os

#File management imports
import pickle as pickle
import glob

#UI imports?

#Sciency imports
 
#This probably isn't a good way to do this. I think it is currently only used for wiener filter
from scipy import misc
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from scipy import signal
import scipy

#Make python into matlab imports
import numpy as np
import math
import matplotlib.pyplot as plt

#Image management imports (note that scipy does a lot of the legwork here too)
from PIL import Image
import tifffile as tiff



#%%
#todo ask user for settings using tkinter?

#%%
#==============================================================================
# Settings
# The user should (and must for the program to work) change these
# variables to reflect the parameters of the measurement.
# Directories should point to existing directories
# Directories should be separated with a '/', not a '\'
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
#==============================================================================



#Get home directory
#This allows the user to flexibly point to a certain location independent of 
#device such as a google drive directory where all the data is stored
#This can be completely omitted in favour of hard coding the directory settings
try:
    from win32com.shell import shell,shellcon
    userdir = shell.SHGetFolderPath(0, shellcon.CSIDL_PROFILE, None, 0)+'/'
    home = userdir + 'Google Drive/MsProj/'
except ImportError:
    print('failure to find home directory, using hardcoded default')
    home = 'C:/Users/CwPc/Google Drive/MsProj/'

    
    
#source directory. Currently points to a test
source = home + 'dat/cali/0203/'

#output directory
out = home + 'dat/cali/macro_out/0203/'

#Calibration directiory
#This shold point the location of previous calibrations, or where future
#calibrations should be stored
caldir = home + 'dat/cali/fit/'



#x and y window for analysis
#These are absolute values, and should be selected using imageJ or an
#equivalent program
[x1,x2] = [40,80]
[y1,y2] = [60,100]

#Provides the computer with the bitdepth of the image
bitdepth = np.power(2,16)

#These options are for when the user wants to choose the window dynamicaly
#I needed to do this to manage film measuring a thinly colimated beam
#For uniformly irradiated strips, this is not necessary.
#Turn on or off dynamic window mode
dynamic = True

#Pick a rectangle for within which to choose the dynamic window
#This should be the user's best guess for the outer bounds of where the maximum
#dose regions might be, despite any positional uncertainty
# Relative measurements i.e. [x,dx,y,dy]
rect = [64,33,42,79]

#Provides the length of the dynamic window in the non-dynamic direction (horizontal)
ROI_x = [52,106]





#%%
#load an image, return a numpy array
def load_image(fn):
    try:
        im = tiff.imread(fn)
        im = np.array(im)
    except:
        'Could not import ' + fn
        im = np.array([0])
    return im
    



#%%
#Herein the ROI is found dynamically according to sorcerous calculations
#return bounds as two absolute pixel values

def choose_ROI(im):
    #Choose the region of interest based on the pixels falling within 1.05
    #of the central region, with a 8 pixel (~1.5mm) buffer
    buffer = 8
    ratio = 1.05
    im = crop_image(im,rect)
    misc.toimage(im)
    im = np.median(im,axis = 1)
    pixel_list = np.argwhere(im<im.min()*ratio)
    bounds = [pixel_list[0][0]+buffer+rect[2],pixel_list[-1][0]-buffer+rect[2]]
    return bounds

#%%
#Load list of files in directory to filenames
def file_process(path, container,window_mode='dynamic'):

    #Process filename
    fn = path.split('\\')[-1]
    fn = fn.split('.')[0]
    fn = fn.split('_')
    #Get date
    date = fn[0]
    #Get film number
    strip_no = fn[2]
    strip_id = fn[0]+'_'+fn[2]
    if not strip_id in container:
        container[strip_id] = {}
    container[strip_id]['date'] = date
    container[strip_id]['number']= strip_no

    #Get the image up in here
    try:
        im = load_image(path)
    except:
        print('Could not load image at '+path)
        return

    #Remove dead pixels. 
    #im = remove_dead_pixels(im)
    
    #apply wiener filter channel by channel. Is this better than just applying it?
    #for channel in range(im.shape[2]-1):
    #    im[:,:,channel] = signal.wiener(im[:,:,channel])
    #apply wiener filter to red channel

    #select the ROI using coordinates from settings file. Initially, lets choose the RED channel
    #This is working wrong
    if window_mode == 'dynamic':
        if fn[1] == 'after':
            bounds = choose_ROI(im)
            window = im[bounds[0]:bounds[1],ROI_x[0]:ROI_x[1],0]
            container[strip_id]['window']=bounds
        else:
            try:
                bounds = container[strip_id]['window']
                window = im[bounds[0]:bounds[1],ROI_x[0]:ROI_x[1],0]
            except KeyError:
                print('Could not set window mode for '+strip_id +' - defaulting to manual')
                window_mode == 'manual'
    if window_mode == 'manual':
        window = im[y1:y2,x1:x2,0]

#    show_window(im,window)
#    
#    return
    
    mean_val = window.mean()
    std_val = window.std()

    


    container[strip_id]['mean_'+fn[1]] = mean_val
    container[strip_id]['std_'+fn[1]] = std_val
    
    #return the updated dictionary
    return container

#results = file_process(path, results)

#images = [Image.open(fn).convert('L') for fn in filenames]

#%%

#Process the results dictionary
#at the moment, just iterate through and find change in pixval and overall std
def process_results(results):
    for film_id in results:
        try:
            results[film_id]['delta'] = results[film_id]['mean_before']-results[film_id]['mean_after']
            results[film_id]['std'] = np.sqrt( np.square(results[film_id]['std_before'])+np.square(results[film_id]['std_after']) )
        except KeyError:
            #insert code for case where no before/after image exists
            #print('Film ID '+ film_id + ' does not have before scan. Trying strip 001')
            try:
                results[film_id]['delta'] = results[film_id[:-3]+'001']['mean_before']-results[film_id]['mean_after']
                results[film_id]['std'] = np.sqrt( np.square(results[film_id[:-3]+'001']['std_before'])+np.square(results[film_id]['std_after']) )
            except:
                print('Film ID '+ film_id + ' does not have before scan and could not be processed using strip 001 from batch')
    return results

          

#%%


#==============================================================================
# Open the index file and make a list of lists out of the contents
# Note: index file must have the following format:
# line 1: measurement_type,calibration_name
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
# 2 chest_1,0418_001,0418_002
# 3 chest_2,0418_003,0418_004
# ...
# 
# 
# Notes:
#  - measurement_type is either calibration, for an intial calibration 
#    measurement which will be saved as the calibration_name, or measurement for 
#    any subsequent measurements using the calibration_name calibration function
#  - For measurements, the first line will be a description rather than a dose
#    e.g. 
#  - each measurement point can have an arbitrary number of film strips associated
#==============================================================================

def get_index(fn):
    with open(fn) as f:
        content = f.readlines()
        #Strip trailing nonsense
    content = [x.strip().split(',') for x in content]
    return content


#Calculate the weighted mean of the sample set, based on the relative uncertainties as well as the overall uncertainty

def combine_samples(film_id_list,results):
    total_mean = 0
    std_sum_inv_squares = 0
    std_sum_inv = 0
    scanner_mean = []
    for film_id in film_id_list:
        std = results[film_id]['std']
        mean = results[film_id]['delta']
        std_sum_inv_squares += 1/np.square(std)
        std_sum_inv += 1/std
        total_mean += mean/np.square(std)
        scanner_mean.append(mean)
        scanner_std = np.std(scanner_mean)
    pixel_std =np.sqrt(1/std_sum_inv)
    total_mean= total_mean/std_sum_inv_squares
    total_std = np.sqrt(np.square(pixel_std)+np.square(scanner_std))
    return total_mean, total_std
           
#Iterate through the index and return the weighted mean and total uncertainty for each sample set
def index_samples(results,source = source):
    try:
        index = get_index(source+'/index.txt')
    except FileNotFoundError:
        print('Could not load index file, cannot index results')
        return []
    labels = []
    data = []
    settings = index.pop(0)
    for i in index:
        name = i[0]
        file_id_list = i[1:]
        mean, std = combine_samples(file_id_list,results)
        labels.append(name)
        data.append([name[:-3],mean/bitdepth,std/bitdepth])
    data = np.array(data).astype(float)
    cal,doses = apply_calibration(data,settings)
    np.savetxt(home + 'dat/cali/macro_out/output.csv', data, delimiter=',',fmt="%s")
    return [labels,data,settings,cal,doses]

def apply_calibration(data,settings):
    try:
        mode = settings[0]
        name = settings[1]
    except IndexError: 
        print('Index file line 1 does not have valid measurement mode or name or format')
        return
    if mode == 'calibration':
        #create calibration function from data
        cal = Calibration(name,R=data[:,1],D=data[:,0],sigma=data[:,2])
        doses = []
    elif mode == 'measurement':
        cal = Calibration(name)
        doses = [cal.get_dose(R) for R in data[:,1]]
        #Apply calibration function to all data
    else:
        print('mode was not either calibration or measurement')
    return cal,doses

#%%
#Dead pixel functions

#Remove dead pixels
#Refers to previously calculated array, saved in a text file in a certain directory.
#Not currently working =[
def remove_dead_pixels(im):
    dead_pixel_list = np.loadtxt(home + 'dat/cali/source/pixellist.csv', delimiter=',')
    #Choose an averaging filter for all 8 adjacent pixels
    my_filter = np.array([[1,1,1],[1,0,1],[1,1,1]])
    
    #(essentially) convolve the image with my filter in order to create an averaged image
    mean_im = ndimage.generic_filter(im, np.mean,footprint=my_filter)
    
    #iterate over all dead pixels. For each one, replace the same pixel in the orginal image with the equivalent in the averaged image
    for row in dead_pixel_list:
        x,y = row
        im[x,y] = mean_im[x,y]
    
    return(im)
    
#Find dead pixels from background image
def get_dead_pixels(fn):
    im = load_image(fn)
    
    #todo: lets make this a LOT more nuanced
    mean=im.mean
    dead_pixel_list = np.argwhere(im<255*.9)
    np.savetxt(home +'dat/cali/source/pixellist.csv', dead_pixel_list, delimiter=',',fmt='%s')
    return dead_pixel_list


#%%
#Window an image based on a rectangle
#rect = 

def crop_image(image,rectangle):
    x,dx,y,dy = rectangle
    cropped_image=image[y:y+dy,x:x+dx,0]
    return cropped_image


#Test window, showing before and after images

def show_window(image,rect,mode = 'relative'):
    if (mode == 'relative'):
        x,dx,y,dy = rect
    if (mode == 'absolute'):
        x,x2,y,y2 = rect
        dx = x2-x
        dy = y2-y
    im_windowed = np.copy(image)
    im_windowed[y:y+dy,x:x+dx,:]=1
    im_temp = misc.toimage(im_windowed)
    plt.imshow(im_temp)
    
    return im_temp
    
    
#%%
#Take an image and a calibration function then turn it into a dose map
def make_dosemap(im,imbackground,cal):
    im = imbackground/bitdepth - im/bitdepth
    im[im<0]=0
    im = cal.get_dose(im)
    im = im.astype(np.uint16)
    return im
    
def save_dosemap(doseim,ffn): 
    #now take the dosemap and save it to a file
    misc.toimage(doseim, high=np.max(doseim), low=np.min(doseim),mode='I').save(ffn[:-4]+'.png')
#misc.toimage(test, cmin=0, cmax=255,mode='I').save("tmp.png")
    
def folder_to_dosemaps(path):
    index = get_index(path+'index.txt')
    name=index.pop(0)
    for l in index:
        cal = l.pop(0)
        cal = Calibration(cal)
        for im in l:
            file_to_dosemap(path,im+'.tif',cal)

def file_to_dosemap(path,fn,cal):
    im = load_image(path + fn)[:,:,0]
    #Load background image
    #Try before file
    try:
        backfn = fn[:4]+'_before_'+fn[-7:]
        if os.path.exists(path+backfn):
            imback = load_image(path+backfn)[:,:,0]
        else:
            backfn = backfn[:-6]+'01'+backfn[-4:]
            imback = load_image(path+backfn)[:,:,0]
    except:
        print('Could not load before image for dosemap'+fn)
    imdose = make_dosemap(im,imback,cal)
    os.makedirs(path+'dosemap/',exist_ok=True)
    save_dosemap(imdose,path+'dosemap/'+fn[:-4]+'_dosemap.tif')

#test = file_to_dosemap(home+'dat/skin/frontratio/','2003_after_070.tif',testcal)
folder_to_dosemaps(home+'dat/skin/frontratio/')
        
#testim,testimback,testimdose = file_to_dosemap(home+'dat/cali/0203/','0203_after_005.tif',cal120)
#%%

crop_to_film(im):
    #this needs to have code that crops an image file to the section that contains film.
    return im


#%%
#This class allows the creation of a calibration function.
#By calling get_dose(R), it is possible to manually check the dose for a specific strip

# class functions, it is possible to apply the generated function to 

class Calibration:
    
    #Initiate the calibration
    #Requires the name parameter, which represents the calibration function
    #If the name has previously been used, loads the calibration function for that name
    #If keywords in the form of R,D and sigma are passed
    #create a new calibration function overwriting the existing one
    def __init__(self,name = '',**kwargs):
        self.name=name
        if 'R' in kwargs:
            self.new_calibration(**kwargs)
        elif not name == '':
            self.load_calibration()
        else:
            print('No calibration function was initiated')

    #Create a new function using key word arguments of the form: R = Reflectionvals, D=Dosevals,sigma=uncertaintyvals
    #If any previous calibration function existed with this name, it will be overwritten.
    def new_calibration(self,**kwargs):
        self.D = kwargs['D']
        self.R = kwargs['R']
        self.sigma = kwargs['sigma']
        if kwargs['D'][0]==0:
            kwargs['D'] = kwargs['D'][1:]
            kwargs['R'] = kwargs['R'][1:]
            kwargs['sigma'] = kwargs['sigma'][1:]
        [self.a,self.b],self.fit_cov = curve_fit(self.fit_function,kwargs['R'],kwargs['D'],sigma=kwargs['sigma'])
        print(self.a)
        print(self.b)
        if not self.name =='':
            self.save_calibration()
        
    #This returns a function based on the variables R and the constants a and b.
    #By changing the form of this function, the form of the fit can be altered
    #Several alternative fit options have been included but commented out
    def fit_function(self,R,a,b):
    #Log function. This is quite bad
    #    return a+b*R/np.log(R)
    #Exponential function. ok fit
#        return a*R*np.exp(b*R)
    #Rational function. Looks like the best fit so far
        return a*R/(1+b*R)
    #Rational function with a power argument. For science.
#        return a*R/(1+b*R)
    #something funky
#        return a*np.power(R,b)/(1-np.log(R*c))
        
    #Calling this function returns the dose for a given reflectance, according to the calibration function.
    def get_dose(self,R):
        return self.fit_function(R,self.a,self.b)
    
    #Save the calibration function to disk.
    #Todo not robust
    def save_calibration(self):
        pickle.dump([self.a,self.b,self.fit_cov,self.R,self.D,self.sigma],open(caldir+self.name+'.p',"wb"))
        
        
#        np.savetxt(home+'dat/cali/fit/'+name+'.csv', dead_pixel_list, delimiter=',',fmt='%s')
    
    #Load a calibration function from a disk
    #todo not robust
    def load_calibration(self):
        try:
            [self.a,self.b,self.fit_cov,self.R,self.D,self.sigma]= pickle.load(open(caldir+self.name+'.p','rb'))
        except:
            print('Could not load calibration with name '+self.name)
            
    
            
    #Show the calibration function in the inline spyder window
    #todo save plot to output folder
    def show_calibration(self,**kwargs):
        
        R=np.linspace(0.01,0.48,90)
        curve, = plt.plot(R,self.fit_function(R,self.a,self.b),label = self.name)
        col = curve.get_color()
        plt.errorbar(self.R,self.D,xerr=self.sigma,yerr=self.D*0.03,fmt=col+'+')
        plt.axis((0,0.47,0,400))
        plt.ylabel('Dose (mGy)')
        plt.xlabel(r'$\Delta R$')
        #plt.show()


        
        
#%%


#%%
#Run the program, call the stuff
filenames = glob.glob(source+'/*.tif')

results = {}


#Can I make this order the list?
for fn in filenames:
    file_process(fn, results,'dynamic')
#%%
results = process_results(results)
#%%

#test_labels, test_data, test_settings, test_cal,test_doses = index_samples(results)





#This batch runs a bunch of sub directories then plots all the calibration functions together
def run_directories(path):
    all_results = {}
    #['2003/80/','2003/100/','2003/140/','0203']
    for sd in ['2003/80/','2003/100/','0203','2003/140/']:
        filenames = glob.glob(path + sd+'/*.tif')
        
        results = {}
        
        
        #Can I make this order the list?
        for fn in filenames:
            file_process(fn, results,'dynamic')
        results = process_results(results)
        all_results[sd] = index_samples(results,path+sd)
        

    for e in all_results:
        cal_name = all_results[e][2][1]
        data = all_results[e][1]
        all_results[e][3].show_calibration(x=data[0:,1],y=data[:,0],xerr = data[:,2],yerr = 0.05*data[:,0])
        plt.legend(loc=2)
    plt.savefig('cals.png', format='png', dpi=600)
    return all_results
    
testdir = home + 'dat/cali/'
all_results = run_directories(testdir)
#%%

    
    #all_results[sd][3].show_calibration()

#test_data2=test_data
#test_cal.show_calibration(x=test_data[0:,1],y=test_data[:,0],xerr = test_data[:,2],yerr = 0.05*test_data[:,0])
#test_cal.show_calibration(x=test_data[0:,1],y=test_data[:,0],xerr = test_data[:,2],yerr = 0.05*test_data[:,0])










#cal120 = Calibration('120kVp')
#cal = Calibration('120kVp')
#cal = Calibration('120kVp',R=test_data[:,1],D=test_data[:,0],sigma=test_data[:,2])
#cal.save_calibration('120kVp')

#cal120.show_calibration(x=test_data[0:,1],y=test_data[:,0],xerr = test_data[:,2],yerr = 0.05*test_data[:,0])



#D=cal.get_dose(R)
#cal.get_dose(R)
#test_test[:,0],test_dose_numbers,xerr = test_test[:,1]
#print(llog(0.21792597,popt[0],popt[1]))
#array([ -4.41190628e-08,  -4.31020280e+02])
#%%

#all_before = []
#for entry in results:
#    try:
#        all_before.append(results[entry]['mean_before'])
#    except:
#        print('could not get mean before for strip '+entry)



        

#%%
#fn = home+'dat/cali/0203/0203_after_011.tif'
#test_image = load_image(fn)

#test = choose_ROI(test_image) 

#show_window(test_image,[ROI_x[0],ROI_x[1],test[0],test[1]],'absolute')


#This is where stuff that's not currently being used lives


#%%
#Goes through and shows a random strip window location. For proof checking of
#the dynamic window function
def test_windows(results):
    for strip_id in results:
        if not strip_id[4:] == '_014':
            continue
        fn = strip_id[0:4]+'_before'+strip_id[4:]+'.tif'
        print(fn)
        im = load_image(home+'dat/cali/0203/'+fn)
        bounds = results[strip_id]['window']
        rectangle = [ROI_x[0],ROI_x[1],bounds[0],bounds[1]]
        show_window(im,rectangle,'absolute')
    
#test_windows(results)

#%%
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
    
    
    

#%%
#example usage of skimage

#image = data.coins()
# ... or any other NumPy array!
#edges = filters.sobel(image)
#io.imshow(edges)
#io.show()



#%%
#choose directory using tkinter
#from tkinter import filedialog
#from tkinter import *
#source = filedialog.askdirectory()