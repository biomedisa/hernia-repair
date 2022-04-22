#!/usr/bin/env python
# coding: utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pydicom
import time,shutil,ssl
import requests
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import logging
from subprocess import run
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dateutil import tz

def ask_continue():
    print(f'Do You want to continue (y) or close the Application (n)?\n')
    user_input = input()
    if user_input in ['y','yes', 'ja' ]:
        print('Continuing...')
    elif user_input in ['n', 'no', 'nein']:
        print('Shuting Down.')
        exit()
    else: 
        print('Please enter only "y" or "n"!')
        ask_continue()

def update_neural_nets():
    sources = ['https://biomedisa.org/media/img_hernie.h5','https://biomedisa.org/media/Hernien_detector_x.h5','https://biomedisa.org/media/Hernien_detector_z.h5']
    destinations = [f'{os.environ["userprofile"]}\\git\\Netzwerke\\img_hernie.h5',f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_x.h5',f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_z.h5']
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

def create_patient_directory(path_to_dir):
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
    files = os.listdir(dcm_dir)
    #Set patients directory 
    ds = pydicom.filereader.dcmread(f'{dcm_dir}\\{files[1]}')
    z_res = ds.SliceThickness
    y_res, x_res = ds.PixelSpacing

    return str(z_res), str(y_res), str(x_res)

def compare_slice_amount(nativ_dcm_dir, valsalva_dcm_dir):
    nativ_slice_amount = len(os.listdir(nativ_dcm_dir))
    valsalva_slice_amount = len(os.listdir(valsalva_dcm_dir))
    if nativ_slice_amount > valsalva_slice_amount:
        print(f'There are {nativ_slice_amount - valsalva_slice_amount} more nativ than valsalva scans! \n This will impact the result. \n\n')
        ask_continue()
    elif nativ_slice_amount < valsalva_slice_amount:
        print(f'There are {valsalva_slice_amount - nativ_slice_amount} more nativ than valsalva scans! \n This will impact the result. \n\n')
        ask_continue()

def creat_ct_crosssection(path_to_layer_txt,observation_path):
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
                text= f'{observation} \n Layer: {layer}',
                fill=(255,255,255),
                anchor='ma',
                font=font,
                )
        #save the crosssection
        img.save(observation_path[observation]['crosssection'],format='png')

def hernia_analysis():
    #Set the paths for both observations
    Observations = ["Nativ","Valsalva"]
    observation_path = {observation:{"tif":"","vtk":"","png":"","crosssection":"","dcm_dir":"","length_dir":"","slice_thickness":""}for observation in Observations}
    
    #Create and get the patients working directory
    observation_path['Nativ']['dcm_dir'] = path_to_nativ 
    observation_path['Valsalva']['dcm_dir'] = path_to_valsalva

    first_level = create_patient_directory(observation_path['Nativ']['dcm_dir'])
    for observation in sorted(Observations):            
        #Create Paths to the mesh and the img
        observation_path[observation]['tif'] = f'{first_level}\\final.{observation}.tif'
        observation_path[observation]['vtk'] = f'{first_level}\\{observation}_for_paraview.vtk'
        observation_path[observation]['png'] = f'{first_level}\\{observation}_front_view.png'
        observation_path[observation]['length_dir'] = f'{first_level}\\{observation}_length'  
        observation_path[observation]['crosssection'] = f'{first_level}\\{observation}_crosssection.png'        
        observation_path[observation]['slice_thickness'],\
        observation_path[observation]['y_dim'],\
        observation_path[observation]['x_dim'] = get_slice_dims(observation_path[observation]['dcm_dir'])

        if not os.path.exists(observation_path[observation]['length_dir']):
            os.mkdir(observation_path[observation]['length_dir']) 

    #selection Checks (Are the Dimensions of Nativ and Valsalva the same)
    if observation_path['Nativ']['slice_thickness'] != observation_path['Valsalva']['slice_thickness']:
        print(f'Nativ and Valsalva slice thickness missmatch! \n This will impact the result. \n\n')
        ask_continue()

    elif observation_path['Nativ']['y_dim'] != observation_path['Valsalva']['y_dim']:
        print(f'Nativ and Valsalva depth missmatch! \n This will impact the result. \n\n')
        ask_continue()

    elif observation_path['Nativ']['x_dim'] != observation_path['Valsalva']['x_dim']:
        print(f'Nativ and Valsalva width missmatch! \n This will impact the result. \n\n')
        ask_continue()
    
    compare_slice_amount(observation_path['Nativ']['dcm_dir'],observation_path['Valsalva']['dcm_dir'])


    #Execute Samuels script automaticaly and combine results
    sam = run([f'{os.environ["userprofile"]}\\git\\hernia-repair\\Hernienauswertung_v0_13.exe',
                    observation_path['Nativ']['dcm_dir'], 
                    observation_path['Valsalva']['dcm_dir']
                ])
  
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
    
    
    #Run all subprocesses for both observations
    for observation in Observations:
            
        #console output
        os.system('cls')
        print(f'Processing {observation}:\n Computing Labels...')
        
        
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
        
        
        #Create nativ mesh, in vtk format for Paraview
        print(f'Creating Mesh...')
        mesh = run(["python",f'{os.environ["userprofile"]}\\git\\hernia-repair\\create_mesh.py', 
                    observation_path[observation]['tif'], observation_path[observation]['vtk'], observation_path[observation]['slice_thickness'] ]
                    )

        #Create image using Paraview
        print(f'Creating Image...')
        screenshot = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\paraview_screenshot.py',
                        observation_path[observation]['vtk'],
                        observation_path[observation]['png']
                        ])

        #Preprocess and Annotate the Paraview labeled images
        print(f'Scaling image...')
        #Read the image
        observation_img = plt.imread(observation_path[observation]['png'])
        #Reshape to match size of sam_img and to fit annotation
        if observation == 'Nativ':
            observation_img = np.pad(observation_img, ((50,0),(0,1),(0,0)), mode='constant',constant_values=1)
        elif observation == 'Valsalva':
            observation_img = np.pad(observation_img, ((50,0),(0,0),(0,0)), mode='constant',constant_values=1)
        plt.imsave(observation_path[observation]['png'],observation_img)       
        

        #Annotate the images
        print('Annotating image...')
        annotate = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\Prediction.py',
                        observation,
                        observation_path[observation]['dcm_dir'],
                        observation_path[observation]['length_dir'],
                        observation_path[observation]['tif'],
                        observation_path[observation]['png']
                        ])
    
    #Patching Images togther for presentation
    os.system('cls')
    print('Arranging Results...')

    #Get the crossection image as the layer with the biggest offset between nativ and valsalva
    creat_ct_crosssection(f'{path_to_archiv}\\sliceID and sliceName maxDisplacement.txt',observation_path)

    #Load all images
    sam_img = plt.imread(f'{path_to_evaluation}\\Verschiebung und Verzerrung.png')
    nat_img = plt.imread(observation_path['Nativ']['png'])[:,:,:3]
    val_img = plt.imread(observation_path['Valsalva']['png'])[:,:,:3]
    nativ_crosssection = plt.imread(observation_path['Nativ']['crosssection'])[:,:,:3]
    valsalva_crosssection = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]

    #Resize images to same width for stacking 
    nativ_crosssection = np.pad(nativ_crosssection,((0,0),(39,40),(0,0)), mode='constant')
    valsalva_crosssection = np.pad(valsalva_crosssection,((0,0),(39,39),(0,0)), mode='constant')
    
    #Stack the image pairs
    nat_and_val = np.hstack((nat_img,val_img))
    double_cross = np.hstack((nativ_crosssection,valsalva_crosssection))
    #Stack result, paraview images and crosssections
    combined_img = np.vstack((sam_img,nat_and_val,double_cross))
    #Save image in evaluation directory
    plt.imsave(f'{path_to_evaluation}\\Finale_Auswertung.png',combined_img)

    #console Output
    print('Moveing used Data')


    #Move used Data into the archiv folder
    for file in (observation_path['Nativ']['png'],observation_path['Valsalva']['png'],
                observation_path['Nativ']['tif'],observation_path['Valsalva']['tif'],
                observation_path['Nativ']['vtk'],observation_path['Valsalva']['vtk'],
                observation_path['Nativ']['crosssection'],observation_path['Valsalva']['crosssection'],
                ):
        shutil.move(file, path_to_archiv)
    
    #show final result
    os.system(f'start "{path_to_evaluation}\\Finale_Auswertung.png"')

#Try loop in case of error
try:
    if __name__ == "__main__":
        
        #Clear the consoloutput
        os.system('cls')
        print('Updating neuralnets')
        
        #Get time to measure execution time
        total_start_time = datetime.now()

        #Set the main save directory    
        main_folder = f'{os.environ["userprofile"]}\\Hernien_Analyse_{datetime.now().strftime("%Y-%m-%d")}'
        if not os.path.exists(main_folder):
            os.mkdir(main_folder) 

        #Set the logging Config
        logging.basicConfig(filename= f'{main_folder}\\time_per_patient.log', level=logging.DEBUG)

        #Check for updates and update the neural nets
        try:
            network_folder = f'{os.environ["userprofile"]}\\git\\Netzwerke'
            if not os.path.exists(network_folder):
                os.mkdir(network_folder)
            update_neural_nets()
        except:
            print('Could not update neuralnets! Check your internet connection.\n', 'Start with old ones.')
        
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
            logging.INFO(f'Execution time: {single_case_end_time - single_case_start_time}')
            #skip the next empty line in the txt file
            line = txt_file.readline()

        #close the file
        txt_file.close()

        total_end_time = datetime.now()
        print(f'Total execution time: {total_end_time - total_start_time}')
        logging.info(f'Total execution time: {total_end_time - total_start_time}')


# Catch the error and log it to a file in the main Directory
except Exception as Argument: 
    #open the error txt file to write to
    f = open('Error_file.txt', 'a')
    #Write into the error file
    f.write(str(Argument))
    #close the error file
    f.close()
    
    
    
    
