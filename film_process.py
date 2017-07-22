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
from scipy import odr
import scipy
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology as mph


#Make python into matlab imports
import numpy as np
import math
import matplotlib.pyplot as plt

#Image management imports (note that scipy does a lot of the legwork here too)
from PIL import Image
#import tifffile as tiff

#plt.style.use('seaborn-paper')
#plt.style.use('default')


#todo ask user for settings using tkinter?

logging = False


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
out = 'out/'

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
    if logging:
        print(fn)
    try:
        im = cv2.imread(fn,-1)
        im = np.float32(im)
        
        for i in np.arange(im.shape[2]):
            im[:,:,i] = signal.wiener(im[:,:,i])
    except:
        'Could not import ' + fn
        im = np.array([0])
    return im[:,:,2]
    return im[:,:,::-1]
    



#%%
#Herein the ROI is found dynamically according to sorcerous calculations
#return bounds as two absolute pixel values

def choose_ROI(im):
    #Choose the region of interest based on the pixels falling within 1.05
    #of the central region, with a 8 pixel erosion
    buffer = 2
    ratio = 1.04
    ignore_pixels = 150
    film_threshold = 49711
    mask = im<film_threshold
    for _ in np.arange(12):
        mask = mph.binary_erosion(mask)
    pixels = np.argsort(im[mask])
    cutoff = im[mask][pixels[ignore_pixels]]*ratio
    t=im<cutoff
    mask2 = t & mask
    for _ in np.arange(buffer):
       mask2 = mph.binary_erosion(mask2)
    mask2 = mph.remove_small_holes(mask2)
    if logging:
        plt.imshow(im*~mask2)
        plt.show()
    if mask2.any():
        return mask2
    return mask


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
    
    #select the ROI using coordinates from settings file. Initially, lets choose the RED channel
    if window_mode == 'dynamic':
        if fn[1] == 'after':
            mask = choose_ROI(im)
            container[strip_id]['mask']=mask
        else:
            try:
                mask = container[strip_id]['mask']
            except KeyError:
                print('Could not set window mode for '+strip_id +' - defaulting to manual')
                window_mode = 'manual'
    
    if window_mode == 'manual':
        mask= np.zeros(im.shape)
        mask[y1:y2,x1:x2]=1



    mean_val = im[mask].mean()
    std_val = im[mask].std()

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
    content = list(filter(None,content))
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
    if len(film_id_list)==1:
        total_std+=total_mean*.05
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
    if settings[0] not in ['calibration','measurement']:
        print(settings[0] +'is not a valid operation. Typo?!')
        
    cal_list = []

    for i in index:
        name = i[0]
        if settings[0] == 'measurement':
            cal_list.append(i.pop(1))
        file_id_list = i[1:]
        mean, std = combine_samples(file_id_list,results)
        labels.append(name)
        if settings[0] =='calibration':
            name =name[:-3]
        data.append([name,mean/bitdepth,std/bitdepth])
    if settings[0]=='calibration':
        data = np.array(data).astype(float)
        cal,doses,doseerr = apply_calibration(data,settings)
    else:
        cal,doses,doseerr = apply_calibration(data,settings,cal_list)
        
    batchnames = np.repeat(settings[1],len(labels))[:,np.newaxis]
    csvout = np.append(batchnames,np.array(data),axis=1)
    if doses:
        csvout = np.append(csvout,np.array(doses)[:,np.newaxis],axis=1)
        csvout = np.append(csvout,np.array(doseerr)[:,np.newaxis],axis=1)
    np.savetxt('out/output'+settings[1]+'.csv', csvout, delimiter=',',fmt="%s")
    return [labels,data,settings,cal,doses]
        


def apply_calibration(data,settings,cal_list = []):
    try:
        mode = settings[0]
        name = settings[1]
    except IndexError: 
        print('Index file line 1 does not have valid measurement mode or name or format')
        return
    if mode == 'calibration':
        #create calibration function from data
        
        cal = Calibration(name,R=data[:,1],D=data[:,0],sigma=data[:,2])
        doses = None
        doseserr = None
    elif mode == 'measurement':
        cal = Calibration('120kVp')
        doses = [Calibration(cal_list[i]).get_dose(R[1]) for i,R in enumerate(data)]
        doseserr = [Calibration(cal_list[i]).get_total_err(R[1],R[2]) for i,R in enumerate(data)]

        #Apply calibration function to all data
    else:
        print('mode was not either calibration or measurement')
    return cal,doses,doseserr

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
def make_dosemap(im,imbackground,cal,unit=1000):
    im = imbackground/bitdepth - im/bitdepth
    im[im<0]=0
    im = cal.get_dose(im)*unit
    im[im>2**16]=2**16
    im = im.astype(np.uint16)
    return im
    
def save_dosemap(im,ffn): 
    #now take the dosemap and save it to a file
    try:
        if ffn[-4] == '.':
            ffn = ffn[:-4]
    except:
        pass
    misc.toimage(im, high=np.max(im), low=np.min(im),mode='I').save(ffn+'.png')
#misc.toimage(test, cmin=0, cmax=255,mode='I').save("tmp.png")
    
def folder_to_dosemaps(path,unit = 1000):
    index = get_index(path+'mapindex.txt')
    name=index.pop(0)
    for l in index:
        cal = l.pop(0)
        cal = Calibration(cal)
        for im in l:
            file_to_dosemap(path,im+'.tif',cal,unit)


def file_to_dosemap(path,fn,cal,unit=1000):
    try:
        im = load_image(path + fn)
    except:
        print('error with file ' +path+fn)
    #Load background image
    #Try before file
    try:
        fnsplit = fn.split('_')
        backfn = fnsplit[0]+'_before_'+fnsplit[-1]
        if os.path.exists(path+backfn):
            imback = load_image(path+backfn)
        elif os.path.exists(backfn[:-6]+'01'+backfn[-4:]):
            backfn = backfn[:-6]+'01'+backfn[-4:]
            imback = load_image(path+backfn)
        else:
            imback = np.array([45830])
    except:
        print('Could not load before image for dosemap'+fn)
    imdose = make_dosemap(im,imback,cal,unit)
    os.makedirs(path+'dosemap/',exist_ok=True)
    save_dosemap(imdose,path+'dosemap/'+fn[:-4]+'_dosemap.tif')

#testcal = Calibration('120kVp')
#file_to_dosemap(home+'dat/skin/geo/','2003_geo.tif',testcal)
#folder_to_dosemaps(home+'dat/skin/nocolls/')
        
#testim,testimback,testimdose = file_to_dosemap(home+'dat/cali/0203/','0203_after_005.tif',cal120)
#%%

def crop_to_film(im):
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
        linear = odr.Model(self.fit_function) 
        mydata = odr.RealData(kwargs['R'], kwargs['D'], sx = kwargs['sigma'])
        myodr = odr.ODR(mydata, linear, beta0 = [60, -2])
        myoutput = myodr.run()
        cov = myoutput.cov_beta
        sd  = myoutput.sd_beta
        self.p   = myoutput.beta 
        self.a = self.p[0]
        self.b = self.p[1]
        self.fit_cov = cov
        self.fit_sd = sd
#        [self.a,self.b],self.fit_cov = curve_fit(self.fit_function,kwargs['R'],kwargs['D'],sigma=kwargs['sigma'],)
        print(self.a)
        print(self.b)
        if not self.name =='':
            self.save_calibration()
        
    #This returns a function based on the variables R and the constants a and b.
    #By changing the form of this function, the form of the fit can be altered
    #Several alternative fit options have been included but commented out
    def fit_function(self,A,R):
        a=A[0]
        b=A[1]
    #Rational function. Looks like the best fit so far
        return a*R/(1+b*R)
    #Log function. This is quite bad
    #    return a+b*R/np.log(R)
    #Exponential function. ok fit
#        return a*R*np.exp(b*R)

    #Rational function with a power argument. For science.
#        return a*R/(1+b*R)
    #something funky
#        return a*np.power(R,b)/(1-np.log(R*c))
        
    #Calling this function returns the dose for a given reflectance, according to the calibration function.
    def get_dose(self,R):
        return self.fit_function([self.a,self.b],R)
    
    def get_a_err(self,R):
        return self.fit_sd[0]/self.a
    
    def get_b_err(self,R):
        return R*self.fit_sd[1]/(1+self.b*R)
    
    def get_fit_err(self,R):
        return np.sqrt(self.get_a_err(R)**2+self.get_b_err(R)**2)
    
    def get_exp_err(self,R,sigma):
        return sigma/R/(1+self.b*R)
    
    def get_total_err(self,R,sigma):
        return np.sqrt(self.get_a_err(R)**2+self.get_b_err(R)**2+self.get_exp_err(R,sigma)**2)
        
        
    
    #Save the calibration function to disk.
    #Todo not robust
    def save_calibration(self):
        pickle.dump([self.a,self.b,self.fit_cov,self.R,self.D,self.sigma,self.fit_sd],open(caldir+self.name+'.p',"wb"))
        
#        np.savetxt(home+'dat/cali/fit/'+name+'.csv', dead_pixel_list, delimiter=',',fmt='%s')
    
    #Load a calibration function from a disk
    #todo not robust
    def load_calibration(self):
        try:
            [self.a,self.b,self.fit_cov,self.R,self.D,self.sigma,self.fit_sd]= pickle.load(open(caldir+self.name+'.p','rb'))
        except:
            print('Could not load calibration with name '+self.name)
            
    
            
    #Show the calibration function in the inline spyder window
    #todo save plot to output folder
    def show_calibration(self,**kwargs):
        
        R=np.linspace(0.01,0.48,90)
        curve, = plt.plot(R,self.fit_function([self.a,self.b],R),label = self.name,zorder=1)
        col = curve.get_color()
        plt.errorbar(self.R,self.D,xerr=self.sigma,yerr=self.D*0.03,color = col, marker ='o',linestyle = 'none',markersize = 2,capsize=2,zorder=2)
        plt.axis((0,0.45,0,400))
        plt.ylabel('Dose (mGy)')
        plt.xlabel(r'$\Delta R$')
        
    def show_fit_uncertainty(self,**kwargs):
        R=np.linspace(0.00,0.45,90)
        curve, = plt.plot(R,self.get_fit_err(R)*100,label = 'Fit uncertainty',zorder=1)
        col = curve.get_color()
#        plt.plot(self.R,self.get_exp_err(self.R),color = col, marker ='o',linestyle = 'none',markersize = 2,zorder=2)
        plt.plot(self.R,self.get_exp_err(self.R,self.sigma)*100,'x',label = 'Experimental uncertainty',color=col)
        
        plt.plot(self.R,self.get_total_err(self.R,self.sigma)*100,'.',label = 'Total uncertainty',color=col)
        plt.axis((0,0.45,0,12))
        plt.ylabel(r'Uncertainty ($\sigma/D$%)')
        plt.xlabel(r'$\Delta R$')
        return curve




        
        
#%%


#%%
#Run the program, call the stuff

#folder_to_dosemaps(home+'dat/skin/2404/')

#test_labels, test_data, test_settings, test_cal,test_doses = index_samples(results)




def run_measurement(path):
    filenames = glob.glob(path +'/*.tif')
    results = {}
    for fn in filenames:
        file_process(fn, results,'dynamic')
    results = process_results(results)
    return index_samples(results,path)
#test = run_measurements(home+'dat/skin/1505/')
#%%

#%%
#This batch runs a bunch of sub directories then plots all the calibration functions together
def run_cal(path):
    all_results = {}
    cals = {}
    #['2003/80/','2003/100/','2003/140/','0203']['2003/80/','2003/100/','0203','2003/140/']
    listy = ['2003/80/','2003/100/','0203','2003/140/']
    for sd in listy:
        filenames = glob.glob(path + sd+'/*.tif')
        
        results = {}
        
        
        #Can I make this order the list?
        for fn in filenames:
            file_process(fn, results,'dynamic')
            
        results = process_results(results)
        all_results[sd] = index_samples(results,path+sd)
        cals[all_results[sd][2][1]] = all_results[sd][3]
        

    for sd in listy:
        cal_name = all_results[sd][2][1]
#        data = all_results[sd][1]
        all_results[sd][3].show_calibration() #x=data[0:,1],y=data[:,0],xerr = data[:,2],yerr = 0.05*data[:,0] what were all these arguments even doing in this call?
        plt.legend(loc=2)
#    plt.savefig('cals.png', format='png', dpi=600)
    plt.savefig('cals.eps', format='eps', dpi=600)
    plt.show()
    
    labels = []
    handles = []
    for sd in listy:
        cal_name = all_results[sd][2][1]
#        data = all_results[sd][1]
        curve = all_results[sd][3].show_fit_uncertainty() #x=data[0:,1],y=data[:,0],xerr = data[:,2],yerr = 0.05*data[:,0] what were all these arguments even doing in this call?
        labels.append(cal_name)
        handles.append(curve)
        
    
    line = plt.Line2D((0,1),(0,0), color='k', linestyle='-')
    cross = plt.Line2D((0,1),(0,0), color='k', marker='x',linestyle='')
    plus = plt.Line2D((0,1),(0,0), color='k', marker='.',linestyle='')
        
    plt.legend([handle for i,handle in enumerate(handles)]+[line,cross,plus],
          [label for i,label in enumerate(labels)]+['Fit uncertainty', 'Exp. uncertainty','Total uncertainty'])
        
        

        

#    plt.savefig('cals.png', format='png', dpi=600)
    
    plt.savefig('cal_unc.eps', format='eps', dpi=600)
    plt.show()
    
    return all_results,cals
    


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