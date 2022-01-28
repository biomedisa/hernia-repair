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
import glob
import tkinter as tk
import logging
from tkinter.filedialog import askdirectory
from subprocess import call
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dateutil import tz

def update_neural_nets():
    sources = ['https://biomedisa.org/media/img_hernie.h5','https://biomedisa.org/media/Hernien_detector_x.h5','https://biomedisa.org/media/Hernien_detector_z.h5']
    destinations = [r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\img_hernie.h5",r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\hernien_detector_x.h5",r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\hernien_detector_z.h5"]
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
    #Ask the user for the Path to the Data via Tkinterface
    tk.Tk().withdraw()
    path_to_dir = askdirectory()


    #Console output
    print('Loading Data...')


    #Get the raw dcm files
    files = glob.glob(path_to_dir + '/**/*', recursive=True)

    #Set patients directory 
    for file in files:
        if os.path.isfile(file):
            ds = pydicom.filereader.dcmread(file)
            if not 'first_level' in locals():
                #name the directory containing all results after patient name + Birthdate     	      
                first_level = f'{main_path}\\{ds.PatientName}_{ds.PatientBirthDate}'
                first_level = first_level.replace('^','_')  
                first_level = first_level.replace(' ','_') 
                first_level = first_level.replace('ü','ue')
                first_level = first_level.replace('ä','ae')
                first_level = first_level.replace('ö','oe') 
            second_level = f'{first_level}\\{ds.StudyDate}_{ds.StudyDescription}'
            third_level = f'{second_level}\\{ds.SeriesNumber}_{ds.SeriesDescription}'

            if not os.path.exists(first_level):                       
                os.mkdir(first_level)
            if not os.path.exists(second_level):                       
                os.mkdir(second_level)
            if not os.path.exists(third_level):
                os.mkdir(third_level)

            path_to_dest = f'{third_level}\\{str(ds.InstanceNumber).zfill(6)}.dcm'
            if not os.path.exists(path_to_dest):			
                shutil.copy(file, path_to_dest)

    return first_level

def get_slice_thickness(observation_path):
    files = os.listdir(observation_path['Nativ']['dcm_dir'])
    #Set patients directory 
    ds = pydicom.filereader.dcmread(f'{observation_path["Nativ"]["dcm_dir"]}\\{files[1]}')
    nativ_slice_thickness = ds.SliceThickness
    return nativ_slice_thickness

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
    observation_path = {observation:{"tif":"","vtk":"","png":"","crosssection":"","dcm_dir":"","length_dir":""}for observation in Observations}
    
    #Create and get the patients working directory
    first_level = load_directorys()
    for observation in sorted(Observations):            
        #Create Paths to the mesh and the img
        observation_path[observation]['tif'] = f'{first_level}\\final.{observation}.tif'
        observation_path[observation]['vtk'] = f'{first_level}\\{observation}_for_paraview.vtk'
        observation_path[observation]['png'] = f'{first_level}\\{observation}_front_view.png'
        observation_path[observation]['length_dir'] = f'{first_level}\\{observation}_length'  
        observation_path[observation]['crosssection'] = f'{first_level}\\{observation}_crosssection.png'        
        observation_path[observation]['dcm_dir'] = askdirectory(initialdir = first_level)
        if not os.path.exists(observation_path[observation]['length_dir']):
            os.mkdir(observation_path[observation]['length_dir']) 

    slice_thickness = get_slice_thickness(observation_path)
    
    #Execute Samuels script automaticaly and combine results
    sam = call([r"C:\Users\Hernienforschung\Documents\Auswertungen\Hernienauswertung_v0_12.exe",
                    observation_path['Nativ']['dcm_dir'], 
                    observation_path['Valsalva']['dcm_dir']
                ])
  
    #Set the saving paths for the optained data
    temp_paths = sorted(os.listdir('C:\\Users\\Hernienforschung\\Documents\\Python_Scripts\\Temp')) 
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
        net = call(["python",r"C:\Users\Hernienforschung\git\biomedisa\demo\biomedisa_deeplearning.py", 
                    observation_path[observation]['dcm_dir'], r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\img_hernie.h5", "-p","-bs","6"]
                    )
        shutil.move(observation_path[observation]['dcm_dir'].replace(
                        os.path.basename(observation_path[observation]['dcm_dir']),
                        'final..tif'),
                        observation_path[observation]['tif'])

        #Create nativ mesh, in vtk format for Paraview
        mesh = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\create_mesh.py", 
                    observation_path[observation]['tif'], observation_path[observation]['vtk'], str(slice_thickness)]
                    )


        #console Output
        print('Done\n Creating Images...')


        #Create image using Paraview
        screenshot = call(["python",
                        r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\paraview_screenshot.py",
                        observation_path[observation]['vtk'],
                        observation_path[observation]['png']
                        ])


        #console Output
        os.system('cls')
        print(' Annotate images...')


        #Preprocess and Annotate the Paraview labeled images
        #Read both images
        observation_img = plt.imread(observation_path[observation]['png'])

        #Reshape to match size of sam_img and to fit annotation
        if observation == 'Nativ':
            observation_img = np.pad(observation_img, ((50,0),(0,1),(0,0)), mode='edge')
        elif observation == 'Valsalva':
            observation_img = np.pad(observation_img, ((50,0),(0,0),(0,0)), mode='edge')
        plt.imsave(observation_path[observation]['png'],observation_img)       
    
        #Annotate the images
        annotate = call(["python",
                        r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\Prediction.py",
                        observation,
                        observation_path[observation]['dcm_dir'],
                        observation_path[observation]['length_dir'],
                        observation_path[observation]['tif'],
                        observation_path[observation]['png']
                        ])
    
    #Get the crossection image as the layer with the biggest of set between observations
    creat_ct_crosssection(f'{path_to_archiv}\\sliceID and sliceName maxDisplacement.txt',observation_path)

    #Load all images
    sam_img = plt.imread(f'{path_to_evaluation}\\Verschiebung und Verzerrung.png')
    nat_img = plt.imread(observation_path['Nativ']['png'])[:,:,:3]
    val_img = plt.imread(observation_path['Valsalva']['png'])[:,:,:3]
    nativ_crosssection = plt.imread(observation_path['Nativ']['crosssection'])[:,:,:3]
    valsalva_crosssection = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]
    
    #Resize images to same width for stacking 
    nativ_crosssection = np.pad(nativ_crosssection,((0,0),(39,40),(0,0)), mode='edge')
    valsalva_crosssection = np.pad(valsalva_crosssection,((0,0),(39,39),(0,0)), mode='edge')
    
    #Stack the image pairs
    nat_and_val = np.hstack((nat_img,val_img))
    double_cross = np.hstack((nativ_crosssection,valsalva_crosssection))
    #Stack result, paraview images and crosssections
    combined_img = np.vstack((sam_img,nat_and_val,double_cross))
    #Save image in evaluation directory
    plt.imsave(f'{path_to_evaluation}\\Finale_Auswertung.png',combined_img)

    #console Output
    print('\nDone')

    #Move used Data into the archiv folder
    for file in (observation_path['Nativ']['png'],observation_path['Valsalva']['png'],
                observation_path['Nativ']['tif'],observation_path['Valsalva']['tif'],
                observation_path['Nativ']['vtk'],observation_path['Valsalva']['vtk'],
                observation_path['Nativ']['crosssection'],observation_path['Valsalva']['crosssection'],
                ):
        shutil.move(file, path_to_archiv)
    #show final result
    os.system(f'start {path_to_evaluation}\\Finale_Auswertung.png')

try:#Try loop in case of error
    if __name__ == "__main__":
        #Check for updates and update the neural nets
        try:
            update_neural_nets()
        except:
            print('Couldn´t update neuralnets! Check your internet connection.\n','Start with old ones.')
        #Get time to measure execution time
        start_time = datetime.now()
        #Set the main save directory    
        main_path = "D:\\Hernien_Analyse_Single"
        if not os.path.exists(main_path):
            os.mkdir(main_path)

        hernia_analysis()

        end_time = datetime.now()
        print(f'Execution time: {end_time - start_time}')
# Catch the error and log it to a file in the main Directory
except Exception as Argument: 
    #open the error txt file to write to
    f = open("D:\\Hernien_Analyse_Single\Error_file.txt", "a")
    #Write into the error file
    f.write(str(Argument))
    #close the error file
    f.close()
    
    
    
    
