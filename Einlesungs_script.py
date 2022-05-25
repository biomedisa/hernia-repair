#!/usr/bin/env python
# coding: utf-8

#imports
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pydicom
import time,shutil,ssl
import requests
import urllib.request
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import glob
import tkinter as tk
import logging
from tkinter.filedialog import askdirectory
from subprocess import run
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dateutil import tz

#Update and interface functions
def ask_continue():
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

    destinations = [f'{os.environ["userprofile"]}\\git\\Netzwerke\\img_hernie.h5',
                    f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_x.h5',
                    f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_z.h5']

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

                
                
def load_directorys():
    ''' 
    Asks the user to select a Dicom Dataset and builds the required 
    directory structur by extracting the information from the Dataset.

    Returns
    -------
    string
        Path to the patients main directory
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

def create_patient_directory_auto(path_to_dir):
    '''
    Creates the main directory of a single Patient when programm is run in
    auto mode.

    Parameters
    ----------
    path_to_dir: string
        path to a patients dicom data
    '''
    
    #Console output
    os.system('cls')
    print('Loading Data...')

    #Get the raw dcm files
    files = os.listdir(path_to_dir)
    #Set patients directory 
    ds = pydicom.filereader.dcmread(f'{path_to_dir}\\{files[1]}')
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

def compare_slice_amount(nativ_dcm_dir, valsalva_dcm_dir):
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
        return ask_continue()
    elif nativ_slice_amount < valsalva_slice_amount:
        print(f'There are {valsalva_slice_amount - nativ_slice_amount} more valsalva than nativ({nativ_slice_amount}) scans! \n This will impact the result. \n\n')
        logging.warning(f'There are {valsalva_slice_amount - nativ_slice_amount} more valsalva than nativ({nativ_slice_amount}) scans! \n\n')        
        return ask_continue()

def creat_ct_crosssection(path_to_layer_txt,observation_path):
    '''
    Create a png of the ct-slice with the largest amount of translation.

    Parameters
    ----------
    path_to_layer_txt: string
        path to the .txt file containg the ct-slice id
    observation_path: dict of dict of string
        dict with the Patients Path information
    '''

    layer_file = open(path_to_layer_txt,'r',encoding='utf8')
    layer = int(float(layer_file.readlines()[1]))
    layer_file.close()
    for observation in ["Nativ","Valsalva"]:
        #get the dcm file containg that layer
        layer_path = f'{observation_path[observation]["dcm_dir"]}\\{str(layer).zfill(6)}.dcm'
        #convert the dcm file into an PIL image
        ds = pydicom.filereader.dcmread(layer_path)
        img = ds.pixel_array
        plt.imsave(observation_path[observation]['crosssection'],img,cmap='gray')
        img = Image.open(observation_path[observation]['crosssection'])
        #annotate the image
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf",size=20)
        draw.text(xy=(img.width/2,0),
                text= (f'{observation} \n'
                       f'Layer: {layer}'),
                fill=(255,255,255),
                anchor='ma',
                font=font,
                )
        #save the crosssection
        img.save(observation_path[observation]['crosssection'],format='png')

def get_distortion_dim(path_to_tif,slice_thickness, x_dim):
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
            
            
def annotate_distortion_image(patient_dict):
    '''
    Annotate the distortion projection image.
    Adding name and dimensions of relevant area.
    
    Parameter
    ---------
    patient_dict: dictionary
        The dictionary containg the information of the data set
    '''
    img = Image.open(patient_dict['projection_png'])
    height, width, area = get_distortion_dim(patient_dict['projection_tif'],patient_dict['slice_thickness'],patient_dict['x_dim'])
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
    img.save(patient_dict['projection_png'],format='png')
        
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
    data_array = zoom(data_array, (512/data_array.shape[0],512/data_array.shape[1]), order=3)
    return data_array

def create_distortion_array(path_to_dir, number_of_slices, max_slice_id, path_to_save):
    '''
    Create the 3-dimensional distortion array.

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
    old_ind = -1
    for current_ind, slice_number in enumerate(range(max_slice_id+1-number_of_slices,max_slice_id + 1,1)):
        current_path = f'{path_to_dir}\\Verschiebung_{str(slice_number).zfill(3)}.csv'
        if os.path.exists(current_path):
            Volume[current_ind,...] = create_numpy_layer(current_path) 
            if old_ind!=-1:
                distance = current_ind - old_ind
                for step in range(1, distance, 1):
                    Volume[old_ind + step,...] = ( (distance - step)*Volume[old_ind,...] + (step)*Volume[current_ind,...] ) / distance 
            old_ind = current_ind
    if Volume[-1,...].all() == 0: 
        distance = Volume.shape[0]-1 - old_ind
        for step in range(1, distance + 1, 1):
            Volume[old_ind + step,...] = Volume[old_ind,...]*(1-(step/distance))
    imwrite(path_to_save,Volume)
             
def merge_tifs(path_to_label,path_to_distortion_array,path_to_merged_tif):
    '''
    Replaces labels with the distortion information.

    Parameters
    ----------
    path_to_label: string
        path to the labeld tif
    path_to_distortion_array: string
        path to the distortion array
    path_to_merged_tif: string
        path to the destination of the merged arrays
    '''
    #Read both tif arrays
    label_array = imread(path_to_label)
    distortion_array = imread(path_to_distortion_array)
    #set all labels to 1
    label_array[label_array != 0] = 1
    #were label !=0 set it overwrite it with the distortionvalue but at least 1
    label_array[label_array != 0] = np.maximum(distortion_array[label_array !=0] , 1)
    #Treshold cutoff 60mm
    label_array[label_array >60] = 60
    #make the array intervalued for later use
    label_array = np.rint(label_array)
    label_array.astype(int)
    #save the merged tif
    imwrite(path_to_merged_tif,label_array)
             
def hernia_analysis(path_to_nativ=None, path_to_valsalva=None):
    #Set the paths for both observations
    Observations = ["Nativ","Valsalva"]
    observation_path = {observation:{"tif":"","vtk":"","png":"","crosssection":"","dcm_dir":"","slice_thickness":""}for observation in Observations}

    if path_to_nativ != None:
        first_level = create_patient_directory_auto(path_to_nativ)
        #Create and get the patients working directory
        observation_path['Nativ']['dcm_dir'] = path_to_nativ 
        observation_path['Valsalva']['dcm_dir'] = path_to_valsalva

    else: 
        #Create and get the patients working directory
        first_level = load_directorys()
        for observation in sorted(Observations):
            observation_path[observation]['dcm_dir'] = askdirectory(initialdir = first_level, title=f'Select {observation} Directory')

    for observation in sorted(Observations):            
        #Create Paths to the mesh and the img
        observation_path[observation]['tif'] = f'{first_level}\\final.{observation}.tif'
        observation_path[observation]['projection_tif'] = f'{first_level}\\{observation}_projection.tif'        
        observation_path[observation]['vtk'] = f'{first_level}\\{observation}_for_paraview.vtk'
        observation_path[observation]['projection_vtk'] = f'{first_level}\\{observation}_projection.vtk'        
        observation_path[observation]['png'] = f'{first_level}\\{observation}_front_view.png'
        observation_path[observation]['projection_png'] = f'{first_level}\\{observation}_projection.png'
        observation_path[observation]['crosssection'] = f'{first_level}\\{observation}_crosssection.png'
        observation_path[observation]['slice_thickness'],\
        observation_path[observation]['y_dim'],\
        observation_path[observation]['x_dim'] = get_slice_dims(observation_path[observation]['dcm_dir'])

    #selection Checks (Are the Dimensions of Nativ and Valsalva the same)
    if observation_path['Nativ']['slice_thickness'] != observation_path['Valsalva']['slice_thickness']:
        print(f'Nativ and Valsalva slice thickness missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva slice thickness (z-dim) missmatch! \n This will impact the result. \n\n')
        ask_continue()

    elif observation_path['Nativ']['y_dim'] != observation_path['Valsalva']['y_dim']:
        print(f'Nativ and Valsalva depth missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva depth (y-dim) missmatch! \n This will impact the result. \n\n')
        ask_continue()

    elif observation_path['Nativ']['x_dim'] != observation_path['Valsalva']['x_dim']:
        print(f'Nativ and Valsalva width missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva width (x-dim) missmatch! \n This will impact the result. \n\n')
        ask_continue()
    
    if compare_slice_amount(observation_path['Nativ']['dcm_dir'],observation_path['Valsalva']['dcm_dir']):
        return

    logging.debug('Starting Samuels script.')
    #Execute Samuels script automaticaly and combine results
    if mode == "Single":
        sam = run([f'{os.environ["userprofile"]}\\git\\hernia-repair\\Hernienauswertung_v0_13.exe',
                        observation_path['Nativ']['dcm_dir'], 
                        observation_path['Valsalva']['dcm_dir']
                    ])
        logging.debug('Finished Samuels Script.')
    elif mode == "Multi":
        sam = run([f'{os.environ["userprofile"]}\\git\\hernia-repair\\Hernienauswertung_v0_13batch.exe',
                observation_path['Nativ']['dcm_dir'], 
                observation_path['Valsalva']['dcm_dir']
            ])
        logging.debug('Finished Samuels Script.')
  
    #Set the saving paths for the optained data
    temp_paths = sorted(os.listdir(f'{os.environ["userprofile"]}\\git\\Temp')) 
    temp_path_to_archiv = temp_paths[0]
    temp_path_to_evaluation = temp_paths[1]

    #Set Time String for saving the data
    day_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_to_evaluation = f'{first_level}\\Auswertung_{day_string}'            
    path_to_archiv = f'{first_level}\\Archiv_zur_Fehlerdiagnose_{day_string}'    

    #Move the created directorys into the main directory
    shutil.move(temp_path_to_evaluation, path_to_evaluation)
    shutil.move(temp_path_to_archiv, path_to_archiv)    
    
    #Create the distortion array
    path_to_distortion_array = f'{path_to_archiv}\\distortion_array.tif'
    create_distortion_array(path_to_archiv, len(os.listdir(observation_path['Nativ']['dcm_dir'])),
                            int(sorted(os.listdir(observation_path['Nativ']['dcm_dir']))[-1].lstrip('0').rstrip('.dcm')),
                            path_to_distortion_array)
    
    #Run all subprocesses for both observations
    for observation in Observations:
            
        #console output
        os.system('cls')
        print(f'Processing {observation}:\n Computing Labels...')
        logging.debug(f'Processing {observation}:\n Computing Labels...')
        
        #Create the classification proposal, in form of a tif
        net = run([
                        'python',f'{os.environ["userprofile"]}\\git\\biomedisa\\demo\\biomedisa_deeplearning.py', 
                        observation_path[observation]["dcm_dir"], f'{os.environ["userprofile"]}\\git\\Netzwerke\\img_hernie.h5', "-p","-bs","6"
                  ])
        
        
        #Move the segmentiation propasal into the correct folder
        print(f'Moveing temporary files...')
        temp_path_to_tif = os.path.splitext(observation_path[observation]["dcm_dir"])[0]  
        shutil.move(f'{os.path.dirname(temp_path_to_tif)}\\final.{os.path.basename(temp_path_to_tif)}.tif',
                        observation_path[observation]["tif"])
        
        #Create the Distortion Array and the distortion projection tif

        merge_tifs(observation_path[observation]['tif'],path_to_distortion_array,observation_path[observation]['projection_tif'])     
   
        #Create meshes, in vtk format for Paraview
        print(f'Creating Meshes...')
        #Mesh of the Neural Network proposal
        mesh1 = run(["python",f'{os.environ["userprofile"]}\\git\\hernia-repair\\create_mesh.py', 
                    observation_path[observation]['tif'], observation_path[observation]['vtk'], observation_path[observation]['x_dim'],
                    observation_path[observation]['y_dim'], observation_path[observation]['slice_thickness'], 'labels'
                    ])
        
        #Mesh of the distortion projection
        mesh2 = run(["python",f'{os.environ["userprofile"]}\\git\\hernia-repair\\create_mesh.py', 
                    observation_path[observation]['projection_tif'], observation_path[observation]['projection_vtk'],observation_path[observation]['x_dim'],
                    observation_path[observation]['y_dim'],observation_path[observation]['slice_thickness'], 'distortion'
                    ])
        
        #Create images using Paraview
        print(f'Creating images...')
        #image of the neural network projection
        screenshot1 = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\paraview_screenshot.py',
                        observation_path[observation]['vtk'],
                        observation_path[observation]['png'],
                        "labels",
                        "7",
                        ])
        
        #image of the distortion projection
        screenshot2 = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\paraview_screenshot.py',
                        observation_path[observation]['projection_vtk'],
                        observation_path[observation]['projection_png'],
                        "distortion",
                        str(np.amax(imread(observation_path[observation]['projection_tif']))),
                        ])
        
        

        #Preprocess and Annotate the Paraview labeled images
        print(f'Scaling images...')
        #Read the image
        observation_img = plt.imread(observation_path[observation]['png'])
        projection_img  = plt.imread(observation_path[observation]['projection_png'])
        #Reshape to match size of sam_img and to fit annotation
        if observation == 'Nativ':
            observation_img = np.pad(observation_img, ((50,0),(0,1),(0,0)), mode='constant',constant_values=1)
            projection_img  = np.pad(projection_img, ((50,0),(0,1),(0,0)), mode='constant',constant_values=1)
        elif observation == 'Valsalva':
            observation_img = np.pad(observation_img, ((50,0),(0,0),(0,0)), mode='constant',constant_values=1)
            projection_img  = np.pad(projection_img, ((50,0),(0,0),(0,0)), mode='constant',constant_values=1)
        plt.imsave(observation_path[observation]['png'],observation_img)       
        plt.imsave(observation_path[observation]['projection_png'],projection_img)

        #Annotate the images
        print('Annotating images...')
        annotate = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\Prediction.py',
                        observation,
                        observation_path[observation]['dcm_dir'],
                        observation_path[observation]['tif'],
                        observation_path[observation]['png']
                        ])
        
        annotate_distortion_image(observation_path[observation])
    
    #Patching Images togther for presentation
    os.system('cls')
    print('Arranging Results...')

    #Get the crossection image as the layer with the biggest offset between nativ and valsalva
    creat_ct_crosssection(f'{path_to_archiv}\\sliceID and sliceName maxDisplacement.txt',observation_path)

    #Load all images
    sam_img = plt.imread(f'{path_to_evaluation}\\Verschiebung und Verzerrung.png')
    nat_img = plt.imread(observation_path['Nativ']['png'])[:,:,:3]
    nat_proj_img = plt.imread(observation_path['Nativ']['projection_png'])[:,:,:3]
    val_img = plt.imread(observation_path['Valsalva']['png'])[:,:,:3]
    val_proj_img = plt.imread(observation_path['Valsalva']['projection_png'])[:,:,:3]
    nativ_crosssection = plt.imread(observation_path['Nativ']['crosssection'])[:,:,:3]
    valsalva_crosssection = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]

    #Resize images to same width for stacking 
    nativ_crosssection = np.pad(nativ_crosssection,((0,0),(39,40),(0,0)), mode='constant')
    valsalva_crosssection = np.pad(valsalva_crosssection,((0,0),(39,39),(0,0)), mode='constant')
    
    #Stack the image pairs
    nat_and_val = np.hstack((nat_img,val_img))
    double_proj = np.hstack((nat_proj_img,val_proj_img))
    double_cross = np.hstack((nativ_crosssection,valsalva_crosssection))
    #Stack result, paraview images and crosssections
    combined_img = np.vstack((sam_img,nat_and_val,double_proj,double_cross))
    #Save image in evaluation directory
    plt.imsave(f'{path_to_evaluation}\\Finale_Auswertung.png',combined_img)

    #console Output
    print('Moveing used Data')


    #Move used Data into the archiv folder
    for file in (observation_path['Nativ']['png'],observation_path['Valsalva']['png'],
                observation_path['Nativ']['tif'],observation_path['Valsalva']['tif'],
                observation_path['Nativ']['vtk'],observation_path['Valsalva']['vtk'],
                observation_path['Nativ']['crosssection'],observation_path['Valsalva']['crosssection'],
                observation_path['Nativ']['projection_png'],observation_path['Valsalva']['projection_png'],
                observation_path['Nativ']['projection_tif'],observation_path['Valsalva']['projection_tif'],
                observation_path['Nativ']['projection_vtk'],observation_path['Valsalva']['projection_vtk']
                ):
        shutil.move(file, path_to_archiv)
    
    if mode == "Single":
        #show final result
        os.startfile(f'{path_to_evaluation}\\Finale_Auswertung.png')

#Try loop in case of error
try:
    if __name__ == "__main__":
        
        #Clear the consoloutput
        os.system('cls')
        print('Updating neuralnets')
        
        #Get time to measure execution time
        total_start_time = datetime.now()
        
        #Get the operation mode
        mode = sys.argv[1]

        #Set the main save directory    
        main_folder = f'{os.environ["userprofile"]}\\Hernien_Analyse_{mode}'
        if not os.path.exists(main_folder):
            os.mkdir(main_folder) 

        #Set the logging Config
        logging.basicConfig(filename= f'.\\Debug.log', level=logging.DEBUG)

        #Check for updates and update the neural nets
        try:
            network_folder = f'{os.environ["userprofile"]}\\git\\Netzwerke'
            if not os.path.exists(network_folder):
                os.mkdir(network_folder)
            update_neural_nets()
        except:
            print('Could not update neuralnets! Check your internet connection.\n', 'Start with old ones.')

        if mode == "Single":
            hernia_analysis()
        elif mode == "Multi":
            #open txt file with paths to the data
            txt_file = open(f'{os.environ["userprofile"]}\\git\\Pfade.txt','r',encoding='utf8')
            #read the first emptyline
            line = txt_file.readline()
            #loop over the file till EOF is reached
            while line:
                #get the starting time of the current iteration
                single_case_start_time = datetime.now()
                hernia_analysis(txt_file.readline()[1:-2],txt_file.readline()[1:-2])
                #get the end time of the current itteration
                single_case_end_time = datetime.now()
                #log the used time for the current itteration
                logging.info(f'Execution time: {single_case_end_time - single_case_start_time}')
                #skip the next empty line in the txt file
                line = txt_file.readline()

            #close the file
            txt_file.close()
        else: raise ValueError('Wrong operation Mode. Must be one of "Single" or "Multi"')

        total_end_time = datetime.now()
        print(f'Execution time: {total_end_time - total_start_time}')
        logging.info(f'Execution time: {total_end_time - total_start_time}')


# Catch the error and log it to a file in the main Directory
except: 
    #open the error txt file to write to
    logging.exception('Fehler bei der Ausführung!')
    raise
    
    
    
    
