import glob,os,sys
import pydicom
import numpy as np
from scipy import ndimage
from tifffile import imread,imwrite
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time,shutil,ssl
import requests
import urllib.request
import tkinter as tk
import logging
from tkinter.filedialog import askdirectory
from datetime import datetime
from dateutil import tz
from tensorflow.keras.models import load_model
import config


#######################################################
#Functions for startup, dataloading and assertion_check
#######################################################

#Update and interface functions
def ask_continue(mode):
    '''
    Asks the user if he wishes to continue the execution in case of 
    problems with the data.

    Requires user input: 
    
    y for yes 
    n for no
    
    Returns
    -------
    True if the user wants to continue
    '''
    
    if mode == "Multi":
        return True
    
    print(f'Do You want to continue (y) or close the Application (n)?\n')
    user_input = input()
    if user_input in ['y','yes', 'ja' ]:
        print('Continuing...')
        return True
    elif user_input in ['n', 'no', 'nein']:
        print('Shuting Down.')
        exit()
    else: 
        print('Please enter only "y" or "n"!')
        ask_continue()



def update_neural_nets():
    '''
    Updates the neuralnetworks by checking if there are any updates on the server.
    
    '''
    
    sources = ['https://biomedisa.org/media/img_hernie.h5',
               'https://biomedisa.org/media/Hernien_detector_x.h5',
               'https://biomedisa.org/media/Hernien_detector_z.h5']

    destinations = [f'{config.path_names["neuralnet"]}\\img_hernie.h5',
                   f'{config.path_names["neuralnet"]}\\hernien_detector_x.h5',
                   f'{config.path_names["neuralnet"]}\\hernien_detector_z.h5']

    for src, dst in zip(sources,destinations):
        update = False
        if os.path.exists(dst):
            # source
            response = requests.head(src)
            timestamp1 = response.headers.get('Last-Modified')
            timestamp1 = timestamp1.split(' ')
            del timestamp1[0]
            timestamp1 = ' '.join(timestamp1)
            for i, month in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']):
                if month in timestamp1:
                    timestamp1 = timestamp1.replace(month, '0%s'%(i+1))
            t1 = datetime.strptime(timestamp1, "%d %m %Y %H:%M:%S %Z")
            t1 = t1.replace(tzinfo=tz.tzutc())  # tell datetime it's in UTC time zone
            t1 = t1.astimezone(tz.tzlocal())    # convert to local time
            print("Web:", t1)

            # destination
            timestamp2 = os.path.getmtime(dst)
            timestamp2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp2))
            t2 = datetime.strptime(timestamp2, '%Y-%m-%d %H:%M:%S')
            t2 = t2.astimezone(tz.tzlocal())
            print("Local:", timestamp2)
            update = (t1>t2) 
        #difference = t1 - t2
        #print(t1>t2, difference, difference.days)

        if update or not os.path.exists(dst):
            with urllib.request.urlopen(src,context=ssl._create_unverified_context()) as response,open(dst,'wb') as out_file: 
                shutil.copyfileobj(response,out_file)

                
                
def load_directorys(main_folder):
    ''' 
    Asks the user to select a Dicom Dataset and builds the required 
    directory structur by extracting the information from the Dataset.

    Parameter
    ---------
    main_folder: string 
        Pathstring to the standard main directory

    Returns
    -------
    string
        Pathstring to the patients main directory
    '''

    #Ask the user for the Path to the Data via Tkinterface
    tk.Tk().withdraw()
    path_to_dir = askdirectory(title='Select Dataset')


    #Console output
    print('Loading Data...')
    logging.debug('Loading Data...')


    #Get the raw dcm files
    files = glob.glob(path_to_dir + '/**/*', recursive=True)

    #Set patients directory 
    for file in files:
        if os.path.isfile(file):
            ds = pydicom.filereader.dcmread(file)
            if not 'first_level' in locals():
                #name the directory containing all results after patient name + Birthdate     	      
                first_level = f'{main_folder}\\{ds.PatientName}_{ds.PatientBirthDate}'
                #first_level = first_level.replace('^','_')
                #first_level = first_level.replace('/',' ')
                #first_level = first_level.replace(' ','_')
                first_level = first_level.replace('ü','ue')
                first_level = first_level.replace('ä','ae')
                first_level = first_level.replace('ö','oe')
                first_level = first_level.replace('ß','ss')
            if not 'second_level' in locals():    
                second_level = f'{first_level}\\{ds.StudyDate}_{ds.StudyDescription}'
                #second_level = second_level.replace('/',' ')
            third_level = f'{second_level}\\{ds.SeriesNumber}_{ds.SeriesDescription}'
            #third_level = third_level.replace('/',' ') 
            #third_level = third_level.replace('|','_')
            #third_level = third_level.replace('*',' ')
            third_level = third_level.replace('#',' ') 
            
            #Check existence and create non existing directorys
            if not os.path.exists(first_level):                       
                os.mkdir(first_level)
            if not os.path.exists(second_level):                       
                os.mkdir(second_level)
            if not os.path.exists(third_level):
                os.mkdir(third_level)
            
            #Set the path to the current .dcm files and copy it 
            path_to_dest = f'{third_level}\\{str(ds.InstanceNumber).zfill(6)}.dcm'
            if not os.path.exists(path_to_dest):			
                shutil.copy(file, path_to_dest)

    logging.debug('Data Loaded successfuly.')

    return first_level

def create_patient_directory_auto(path_to_dicom,main_folder):
    '''
    Creates the main directory of a single Patient when programm is run in
    multi/auto mode.

    Parameters
    ----------
    path_to_dir: string
        pathstring to a patients dicom data
    main_folder: string
        pathstring to the programms main directory
    '''
    
    #Console output
    os.system('cls')
    print('Loading Data...')

    #Get the raw dcm files
    files = os.listdir(path_to_dicom)
    #Set patients directory 
    ds = pydicom.filereader.dcmread(f'{path_to_dicom}\\{files[1]}')
    #name the directory containing all results after patient name + Birthdate     	      
    first_level = f'{main_folder}\\{ds.PatientName}_{ds.PatientBirthDate}'
    '''
    first_level = first_level.replace('^','_')
    first_level = first_level.replace('/',' ')
    first_level = first_level.replace(' ','_')
    first_level = first_level.replace('ü','ue')
    first_level = first_level.replace('ä','ae')
    first_level = first_level.replace('ö','oe')
    first_level = first_level.replace('ß','ss')
    '''
    if not os.path.exists(first_level):                       
        os.mkdir(first_level)

    return first_level

def get_slice_dims(dcm_dir):
    '''
    Returns the three voxel sidelengths of the given dataset.

    Parameter
    ---------
    dcm_dir: string
        path to the directory of the dcm datset

    Returns
    -------
    z_res: string
        the voxel length in z-direction
    y_res: string
        the voxel length in y-direction
    x_res: string
        the voxel length in x-direction
    '''

    files = os.listdir(dcm_dir)
    #Load a .dcm file of the datset and extract voxel side lengths
    ds = pydicom.filereader.dcmread(f'{dcm_dir}\\{files[1]}')
    z_res = ds.SliceThickness
    y_res, x_res = ds.PixelSpacing

    return str(z_res), str(y_res), str(x_res)

def compare_slice_amount(nativ_dcm_dir, valsalva_dcm_dir,mode):
    '''
    Compare the amount of slices between the nativ and valsalva Dataset.
    Prompts the user in case of a missmatch.
    
    Parameters
    ----------
    nativ_dcm_dir: string
        Path to the directory containing the nativ data
    valsalva_dcm_dir: string
        Path to the directory containing the valsalva data
    
    Returns
    -------
    True if missmatch is detected
    
    Raises
    ------
    ask_continue()
        when the data has a missmatch
    '''
    
    nativ_slice_amount = len(os.listdir(nativ_dcm_dir))
    valsalva_slice_amount = len(os.listdir(valsalva_dcm_dir))
    if nativ_slice_amount > valsalva_slice_amount:
        print(f'There are {nativ_slice_amount - valsalva_slice_amount} more nativ than valsalva({valsalva_slice_amount}) scans! \n This will impact the result. \n\n')
        logging.warning(f'There are {nativ_slice_amount - valsalva_slice_amount} more nativ than valsalva({valsalva_slice_amount}) scans! \n This will impact the result. \n\n')
        return ask_continue(mode)
    elif nativ_slice_amount < valsalva_slice_amount:
        print(f'There are {valsalva_slice_amount - nativ_slice_amount} more valsalva than nativ({nativ_slice_amount}) scans! \n This will impact the result. \n\n')
        logging.warning(f'There are {valsalva_slice_amount - nativ_slice_amount} more valsalva than nativ({nativ_slice_amount}) scans! \n\n')        
        return ask_continue(mode)


################################################
#Creation of the mask for translation detection
################################################

def load_mask_data(path_to_data):

    if os.path.isdir(path_to_data):
        files = glob.glob(path_to_data+'/**/*', recursive=True)
        for name in files:
            try:
                ds = pydicom.filereader.dcmread(name)
            except:
                files.remove(name)
        img = ds.pixel_array
        data = np.zeros((len(files), ds.Rows, ds.Columns), dtype=img.dtype)
        header = [0] * len(files)
        for k, file_name in enumerate(files):
            ds = pydicom.filereader.dcmread(file_name)
            img = ds.pixel_array
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img[:,:,0]
            data[ds.InstanceNumber-1] = img
            header[ds.InstanceNumber-1] = ds
    else:
        data, header = None, None
    return data, header  

def reduce_blocksize(data):
    zsh, ysh, xsh = data.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        y, x = np.nonzero(data[k])
        if x.any():
            argmin_x = min(argmin_x, np.amin(x))
            argmax_x = max(argmax_x, np.amax(x))
            argmin_y = min(argmin_y, np.amin(y))
            argmax_y = max(argmax_y, np.amax(y))
            argmin_z = min(argmin_z, k)
            argmax_z = max(argmax_z, k)
    argmin_x = max(argmin_x - 1, 0)
    argmax_x = min(argmax_x + 1, xsh-1) + 1
    argmin_y = max(argmin_y - 1, 0)
    argmax_y = min(argmax_y + 1, ysh-1) + 1
    argmin_z = max(argmin_z - 1, 0)
    argmax_z = min(argmax_z + 1, zsh-1) + 1
    data = np.copy(data[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x])
    return data, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x


def clean(image, threshold=0.9):
    image_i = np.copy(image)
    allLabels = np.unique(image_i)
    mask = np.empty_like(image_i)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels[1:]:

        # get mask
        label = image_i==k
        mask.fill(0)
        mask[label] = 1

        # reduce size
        reduced, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(mask)

        # get clusters
        labeled_array, _ = ndimage.label(reduced, structure=s)
        size = np.bincount(labeled_array.ravel())

        # get reference size
        biggest_label = np.argmax(size[1:]) + 1
        label_size = size[biggest_label]

        # preserve large segments
        reduced.fill(0)
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size:
                reduced[labeled_array==l+1] = 1

        # get original size
        mask.fill(0)
        mask[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = reduced

        # write cleaned label to array
        image_i[label] = 0
        image_i[mask==1] = k

    return image_i

def fill(image, threshold=0.9):
    image_i = np.copy(image)
    allLabels = np.unique(image_i)
    mask = np.empty_like(image_i)
    s = [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]]
    for k in allLabels[1:]:

        # get mask
        label = image_i==k
        mask.fill(0)
        mask[label] = 1

        # reduce size
        reduced, argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(mask)

        # reference size
        label_size = np.sum(reduced)

        # invert
        reduced = 1 - reduced # background and holes of object

        # get clusters
        labeled_array, _ = ndimage.label(reduced, structure=s)
        size = np.bincount(labeled_array.ravel())
        biggest_label = np.argmax(size)

        # get label with all holes filled
        reduced.fill(1)
        reduced[labeled_array==biggest_label] = 0

        # preserve large holes
        for l, m in enumerate(size[1:]):
            if m > threshold * label_size and l+1 != biggest_label:
                reduced[labeled_array==l+1] = 0

        # get original size
        mask.fill(0)
        mask[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = reduced

        # write filled label to array
        image_i[label] = 0
        image_i[mask==1] = k

    return image_i

def threshold(img):
    # Image -> Adjust -> Threshold (800-65535)
    zsh, ysh, xsh = img.shape
    a = np.zeros_like(img)
    a[img>800] = 1
    a = np.pad(a,pad_width=1,mode='constant',constant_values=0)
    a[0] = 1
    a[-1] = 1
    a = fill(a, 0.9)
    a[0] = 0
    a[-1] = 0
    a = clean(a, 0.9)
    a = a[1:-1,1:-1,1:-1]
    return a

def create_mask(path_to_dcm,path_to_labels,path_to_mask):
    data, header = load_mask_data(path_to_dcm)
    body_outline = threshold(data)
    muscle_mask = imread(path_to_labels)
    muscle_mask[muscle_mask == 7] = 0
    muscle_mask[muscle_mask != 0] = 2

    mask = np.maximum(body_outline,muscle_mask)
    imwrite(path_to_mask,mask)
    return mask

###################################
#Creation of the translation array
###################################

def get_translation_dim(path_to_tif,slice_thickness, x_dim):
    '''
    Get the height, width and area of the translation region

    Parameter
    ---------
    path_to_tif: string
        The pathstring to the translation array
    slice_thickness: float
        the thickness of a slice in the scaned array
    x_dim: the thickness of 

    Returns
    -------
    height, width, area: float rounded to second decimal
    
    '''
    img = imread(path_to_tif)
    zsh, _, _ = img.shape
    height_array = np.any(img >= 15, axis =(1,2))
    if np.any(height_array):
        height = (np.flatnonzero(height_array)[-1] - np.flatnonzero(height_array)[0]) * float(slice_thickness) * 0.1
    else: height = 0
    width_array = np.any(img >= 15, axis =(0,1))
    if np.any(width_array):
        width = (np.flatnonzero(width_array)[-1] - np.flatnonzero(width_array)[0]) * float(x_dim) * 0.1
    else: width = 0
    area_array = np.any(img >= 15, axis=1)
    if area_array.size != 0:
        area = np.count_nonzero(area_array)* float(x_dim) * float(slice_thickness) * 0.01
    else: area = 0
    return round(height,2), round(width,2), round(area,2)
            
            
def annotate_translation_image(observation_dict):
    '''
    Annotate the translation projection image.
    Adding name and dimensions of relevant area.
    
    Parameter
    ---------
    patient_dict: dictionary
        The dictionary containg the information of the data set
    '''
    img = Image.open(observation_dict['projection_png'])
    height, width, area = get_translation_dim(observation_dict['projection_tif'],observation_dict['slice_thickness'],observation_dict['x_dim'])
    #annotate the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text= (f'Instabile Bauchwand (Verschiebung > 1.5cm)\n'
                   f'Höhe: {height}cm,      Breite: {width}cm,      Fläche: {area}cm²'),
            fill=(0,0,0),
            anchor='ma',
            align = 'center',
            font=font,
            )
    #save the crosssection
    img.save(observation_dict['projection_png'],format='png')
        
def create_numpy_layer(path_to_data): 
    '''
    Create an array out of a textfile containg the x and y values of a 2 diamensional vectorfield.

    Parameter
    --------
    path_to_data: string
        the path to the txt file
    
    Returns
    -------
    data_array: nd.array
        the numpy array with the absolut values of the vectorfield
    '''
    
    file = open(path_to_data,mode='r')
    data_string = file.read()
    data_list = data_string.split("\n")
    vector_array = [data_list[i].split(',') for i in range(len(data_list)-1)]
    vector_array = np.array(vector_array, dtype=float)
    data_array = np.sqrt(vector_array[:,:vector_array.shape[1]//2]**2 + vector_array[:,vector_array.shape[1]//2:]**2)
    data_array = ndimage.zoom(input=data_array, zoom=(512/data_array.shape[0],512/data_array.shape[1]), order=3)
    return data_array

def create_translation_array(path_to_dir, number_of_slices, max_slice_id, path_to_save):
    '''
    Create the 3-dimensional translation array.

    Parameters
    ----------
    path_to_dir: string
        path to directory with array information
    number_of_slices: int
        total number of slices for the dataset
    max_slice_id: int
        maximum slice id. If 0 is included this is different from number_of_slices
    path_to_save: string
        path to the saving location of the final array
    '''

    Volume = np.zeros((number_of_slices,512,512),dtype=float)
    old_ind = number_of_slices
    for current_ind, slice_number in enumerate(range(max_slice_id+1-number_of_slices,max_slice_id + 1,1)):
        current_path = f'{path_to_dir}\\Verschiebung_{str(slice_number).zfill(3)}.csv'
        if os.path.exists(current_path):
            Volume[current_ind,...] = create_numpy_layer(current_path) 
            distance = current_ind - old_ind
            for step in range(1, distance, 1):
                Volume[old_ind + step,...] = (1 - step/distance)* Volume[old_ind,...] + (step/distance)* Volume[current_ind,...]
            old_ind = current_ind
    if Volume[-1,...].all() == 0: 
        distance = Volume.shape[0]-1 - old_ind
        for step in range(1, distance + 1, 1):
            Volume[old_ind + step,...] = (1 - step/distance)* Volume[old_ind,...] 
    imwrite(path_to_save,Volume)
             
def merge_tifs(path_to_label,path_to_translation_array,path_to_merged_tif):
    '''
    Replaces labels with the translation information.

    Parameters
    ----------
    path_to_label: string
        path to the labeld tif
    path_to_trnaslation_array: string
        path to the transwlation array
    path_to_merged_tif: string
        path to the destination of the merged arrays
    '''
    #Read both tif arrays
    label_array = imread(path_to_label)
    translation_array = imread(path_to_translation_array)
    #set all labels to 1
    label_array[label_array != 0] = 1
    #were label !=0 set it overwrite it with the translation value but at least 1
    label_array[label_array != 0] = np.maximum(translation_array[label_array !=0] , 1)
    #Treshold cutoff 60mm
    label_array[label_array >60] = 60
    #make the array intervalued for later use
    label_array = np.rint(label_array)
    label_array.astype(int)
    #save the merged tif
    imwrite(path_to_merged_tif,label_array)

#########################################
#helper functions for the CT-Crosssection
#########################################

def creat_crosssection(path_to_layer_txt,observation_dict):
    '''
    Create a png of the ct-slice with the largest translation.

    Parameters
    ----------
    path_to_layer_txt: string
        path to the .txt file containg the ct-slice id
    observation_dict: dict of string
        dict with the Patients Path information
    
    Returns
    --------
    layer: int
        slice id of the slice with max. translation
    
    '''
    layer_file = open(path_to_layer_txt,'r',encoding='utf8')
    layer = int(float(layer_file.readlines()[1]))
    layer_file.close()
    #get the dcm file containg that layer
    layer_path = f'{observation_dict["dcm_dir"]}\\{str(layer).zfill(6)}.dcm'
    #convert the dcm file into an PIL image
    ds = pydicom.filereader.dcmread(layer_path)
    img = ds.pixel_array
    plt.imsave(observation_dict['crosssection'],img,cmap='gray')
    return layer
    
    
def annotate_crosssection(observation_dict,layer):
    '''
    Annotate the ct_crosssection image.
    Adding name and dimensions of relevant area.
    
    Parameter
    ---------
    observation_dict: dict of string
        dict with the Patients Path information
    '''
    img = Image.open(observation_dict['crosssection'])
    #annotate the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text=f'Schicht der Maximalen Verschiebung: {layer}',
            fill=(255,255,255),
            anchor='ma',
            align='center',
            font=font,
            )
    #save the crosssection
    img.save(observation_dict['crosssection'],format='png')

###################################
#Anotation of the Images
###################################

def load_anotation_data(path_to_data):
    #get the dcm files
    slices = glob.glob(path_to_data+'/**/*', recursive=True)
    #set start values
    ds = pydicom.filereader.dcmread(slices[0])
    img = ds.pixel_array
    #initalize array for the data
    data = np.zeros((len(slices), ds.Rows, ds.Columns), dtype=img.dtype)
    header = [0] * len(slices)
    #loop over all files
    for file in slices:
        #get slice information
        ds = pydicom.filereader.dcmread(file)
        img = ds.pixel_array
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:,:,0]
        #set corresponding slice in the array
        data[ds.InstanceNumber-1] = img
        header[ds.InstanceNumber-1] = ds
    #convert to uint8 and make rgb
    data -= np.amin(data)
    data = data*255/np.amax(data)
    data = np.uint8(data)
    data = np.stack((data,)*3,axis=-1)
    return data,header

def get_hernia_length(data,mode):
    if mode == "length":
        model = load_model(f'{config.path_names["neuralnet"]}\\hernien_detector_z.h5',)
    elif mode == "width":
        model = load_model(f'{config.path_names["neuralnet"]}\\hernien_detector_x.h5',)
        
    prediction = model.predict(data, batch_size = 32,verbose = 0)

    hernia_interval = (prediction > 0.5).nonzero()[0]
    hernia_length = hernia_interval[-1] - hernia_interval[0]

    return hernia_length

def resize_array(data):
    resized_data = np.zeros((512,512,512,3),dtype=data.dtype)
    for i in range(data.shape[0]):
        img = Image.fromarray(data[i],mode = "RGB")
        img = img.resize((512,512))
        slice = np.array(img)
        resized_data[i] = slice
    return resized_data

def annotate_by_neural_net(path_to_dcm,x_res,z_res):
    #Get slice width 
    data, _ = load_anotation_data(path_to_dcm)
    #Get Height,width and area of the hernia
    hernia_height = get_hernia_length(data, mode = "length")
    hernia_height *= z_res*0.1
    
    width_data = resize_array(np.swapaxes(data,0,2))
    
    hernia_width = get_hernia_length(width_data, mode = "width")
    hernia_width *= x_res*0.1
    
    hernia_area = np.pi*hernia_height*hernia_width
    
    return hernia_width, hernia_height ,hernia_area

def annotate_by_label(path_to_tif,x_res,y_res,z_res):
    # load segmentation
    a = imread(path_to_tif)
    zsh, _, _ = a.shape

    # compute size of area between rectus left and right
    area = 0
    for k in range(zsh):
        y0,x0 = np.where(a[k]==1)
        y1,x1 = np.where(a[k]==2)
        if np.any(x0) and np.any(x1):
            argmax = np.argmax(x0)
            argmin = np.argmin(x1)
            x_max,y_max = x0[argmax],y0[argmax]
            x_min,y_min = x1[argmin],y1[argmin]
            area += np.sqrt((x_res*(x_max-x_min))**2 + (y_res*(y_max-y_min))**2) * z_res

    # compute size of hernia area
    hernia_area = 0
    for k in range(zsh):
        y,x = np.where(a[k]==7)
        if np.any(x):
            argmax = np.argmax(x)
            argmin = np.argmin(x)
            x_max,x_min = x[argmax],x[argmin]
            hernia_area += np.sqrt((x_res*(x_max-x_min))**2) * z_res
    return area, hernia_area

def annotate_image(observation,observation_paths):
    hernia_width_by_nn, hernia_height_by_nn,hernia_area_by_nn = annotate_by_neural_net(observation_paths['dcm_dir'],observation_paths['x_dim'],observation_paths['slice_thickness'])
    instable_area_by_label, hernia_area_by_label = annotate_by_label(observation_paths['tif'],observation_paths['x_dim'],observation_paths['y_dim'],observation_paths['slice_thickness'])
    
    #write hernia dimensions on the image
    to_annotate = Image.open(observation_paths['png'])
    draw = ImageDraw.Draw(to_annotate)
    font = ImageFont.truetype("arial.ttf", size=15)
    draw.text(xy=(to_annotate.width/2,0),
            text= (f'{observation}\n'
                   f'Detektiertes Bruchsackvolumen (rot)\n'
                   f'(Berechnete Größen) Breite: {round(hernia_width_by_nn,1)}cm,   Länge: {round(hernia_height_by_nn,1)}cm,    Bruchpforten Fläche: {round(hernia_area_by_nn,1)}cm²\n'
                   f'(Größen im Bild) Instabile_Fläche: {round(instable_area_by_label*0.01,1)}cm²,  Projezierte Fläche: {round(hernia_area_by_label*0.01,1)}cm²'),     
            fill=(0,0,0),
            anchor='ma',
            align = 'center',
            font = font,
            )
    to_annotate.save(observation_paths['png'],format='png')
