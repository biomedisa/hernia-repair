#!/usr/bin/env python
# coding: utf-8

#imports
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import logging
from tkinter.filedialog import askdirectory
from subprocess import run
from datetime import datetime

import config
import hernia_helper
import create_mesh

             
def hernia_analysis(main_folder, path_to_nativ=None, path_to_valsalva=None, mode='Single'):
    #Set the paths for both observations
    Observations = ["Nativ","Valsalva"]
    observation_path = {observation:{"tif":"","vtk":"","png":"","crosssection":"","dcm_dir":"","slice_thickness":""}for observation in Observations}

    if path_to_nativ != None:
        first_level = hernia_helper.create_patient_directory_auto(path_to_nativ,main_folder)
        #Create and get the patients working directory
        observation_path['Nativ']['dcm_dir'] = path_to_nativ 
        observation_path['Valsalva']['dcm_dir'] = path_to_valsalva

    else: 
        #Create and get the patients working directory
        first_level = hernia_helper.load_directorys(main_folder=main_folder)
        for observation in sorted(Observations):
            observation_path[observation]['dcm_dir'] = askdirectory(initialdir = first_level, title=f'Select {observation} Directory')

    for observation in sorted(Observations):            
        #Create Paths to the mesh and the img
        observation_path[observation]['tif'] = f'{first_level}\\final.{observation}.tif'
        observation_path[observation]['mask'] = f'{first_level}\\{observation}_mask.tif'
        observation_path[observation]['projection_tif'] = f'{first_level}\\{observation}_projection.tif'        
        observation_path[observation]['vtk'] = f'{first_level}\\{observation}_for_paraview.vtk'
        observation_path[observation]['projection_vtk'] = f'{first_level}\\{observation}_projection.vtk'        
        observation_path[observation]['png'] = f'{first_level}\\{observation}_front_view.png'
        observation_path[observation]['projection_png'] = f'{first_level}\\{observation}_projection.png'
        observation_path[observation]['crosssection'] = f'{first_level}\\{observation}_crosssection.png'
        observation_path[observation]['slice_thickness'],\
        observation_path[observation]['y_dim'],\
        observation_path[observation]['x_dim'] = hernia_helper.get_slice_dims(observation_path[observation]['dcm_dir'])

    #selection Checks (Are the Dimensions of Nativ and Valsalva the same)
    if observation_path['Nativ']['slice_thickness'] != observation_path['Valsalva']['slice_thickness']:
        print(f'Nativ and Valsalva slice thickness missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva slice thickness (z-dim) missmatch! \n This will impact the result. \n\n')
        hernia_helper.ask_continue(mode)

    elif observation_path['Nativ']['y_dim'] != observation_path['Valsalva']['y_dim']:
        print(f'Nativ and Valsalva depth missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva depth (y-dim) missmatch! \n This will impact the result. \n\n')
        hernia_helper.ask_continue(mode)

    elif observation_path['Nativ']['x_dim'] != observation_path['Valsalva']['x_dim']:
        print(f'Nativ and Valsalva width missmatch! \n This will impact the result. \n\n')
        logging.warning(f'Nativ and Valsalva width (x-dim) missmatch! \n This will impact the result. \n\n')
        hernia_helper.ask_continue(mode)
    
    if hernia_helper.compare_slice_amount(observation_path['Nativ']['dcm_dir'],observation_path['Valsalva']['dcm_dir'],mode):
        return


    
    #Run all subprocesses for both observations
    for observation in Observations:
            
        #console output
        os.system('cls')
        print(f'Processing {observation}:\n Computing Labels...')
        logging.debug(f'Processing {observation}:\n Computing Labels...')
        
        #Create the classification proposal, in form of a tif
        net = run([
                        'python',f'{os.environ["userprofile"]}\\git\\biomedisa\\demo\\biomedisa_deeplearning.py', 
                        observation_path[observation]["dcm_dir"], f'{config.path_names["neuralnet"]}\\img_hernie.h5', "-p","-bs","6"
                  ])
        #Move the segmentiation propasal into the correct folder
        print(f'Moveing temporary files...')
        temp_path_to_tif = os.path.splitext(observation_path[observation]["dcm_dir"])[0]  
        shutil.move(f'{os.path.dirname(temp_path_to_tif)}\\final.{os.path.basename(temp_path_to_tif)}.tif',
                        observation_path[observation]["tif"])

        #create masks for Samuel
        hernia_helper.create_mask(observation_path[observation]["dcm_dir"],observation_path[observation]["tif"],observation_path[observation]["mask"])
        
    logging.debug('Starting Samuels script.')
    #Execute Samuels script automaticaly and combine results
    if mode == "Single":
        sam = run([f'{os.environ["userprofile"]}\\git\\hernia-repair\\Hernienauswertung_v0_13segm.exe',
                        observation_path['Nativ']['mask'], 
                        observation_path['Valsalva']['mask']
                    ])
        logging.debug('Finished Samuels Script.')
    elif mode == "Multi":
        sam = run([f'{os.environ["userprofile"]}\\git\\hernia-repair\\Hernienauswertung_v0_13segmbatch.exe',
                observation_path['Nativ']['mask'], 
                observation_path['Valsalva']['mask']
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
    
    #Create the translation array
    path_to_translation_array = f'{path_to_archiv}\\translation_array.tif'
    hernia_helper.create_translation_array(path_to_archiv, len(os.listdir(observation_path['Nativ']['dcm_dir'])),
                            int(sorted(os.listdir(observation_path['Nativ']['dcm_dir']))[-1].lstrip('0').rstrip('.dcm')),
                            path_to_translation_array)
    
    for observation in Observations:
        #Consol Output
        os.system('cls')
        print(f'Processing {observation}:\n Creating array of displacement...')
        logging.debug(f'Processing {observation}:\n Creating array of displayement...')
        #Create the Translation Array and the translation projection tif
        hernia_helper.merge_tifs(observation_path[observation]['tif'],path_to_translation_array,observation_path[observation]['projection_tif'])     
   
        #Consol output
        print(f'Creating Meshes...')
        logging.debug(f'Processing {observation}:\n Creating Meshes...')
        #Create meshes, in vtk format for Paraview
        create_mesh.CreateVTK(observation_path[observation]['tif'], observation_path[observation]['vtk'], observation_path[observation]['x_dim'],
                    observation_path[observation]['y_dim'], observation_path[observation]['slice_thickness'], 'labels')
        
        #Mesh of the translation projection
        create_mesh.CreateVTK(observation_path[observation]['projection_tif'], observation_path[observation]['projection_vtk'],observation_path[observation]['x_dim'],
                    observation_path[observation]['y_dim'],observation_path[observation]['slice_thickness'], 'translation')
        
        #Consol Output
        print(f'Creating images...')
        logging.debug(f'Processing {observation}:\n Creating images...')
        #Create images using Paraview
          #image of the neural network projection
        screenshot1 = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\paraview_screenshot.py',
                        observation_path[observation]['vtk'],
                        observation_path[observation]['png'],
                        "labels",
                        "7",
                        ])
        
          #image of the translation projection
        screenshot2 = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\paraview_screenshot.py',
                        observation_path[observation]['projection_vtk'],
                        observation_path[observation]['projection_png'],
                        "translation",
                        str(np.amax(imread(observation_path[observation]['projection_tif']))),
                        ])
        
        #Get the crossection image as the layer with the biggest offset between nativ and valsalva
        max_translation_layer = hernia_helper.creat_crosssection(f'{path_to_archiv}\\sliceID and sliceName maxDisplacement.txt',observation_path[observation])

        #Preprocess and Annotate the Paraview labeled images
        print(f'Scaling images...')
        #Read the images
        observation_img = plt.imread(observation_path[observation]['png'])
        projection_img  = plt.imread(observation_path[observation]['projection_png'])
        crosssection_img = plt.imread(observation_path[observation]['crosssection'])
        #Reshape to match size of sam_img and to fit annotation
        if observation == 'Nativ':
            observation_img = np.pad(observation_img, ((50,0),(0,1),(0,0)), mode='constant',constant_values=1)
            projection_img  = np.pad(projection_img, ((50,0),(0,1),(0,0)), mode='constant',constant_values=1)
            crosssection_img = np.pad(crosssection_img,((15,0),(39,40),(0,0)), mode='constant')
        elif observation == 'Valsalva':
            observation_img = np.pad(observation_img, ((50,0),(0,0),(0,0)), mode='constant',constant_values=1)
            projection_img  = np.pad(projection_img, ((50,0),(0,0),(0,0)), mode='constant',constant_values=1) 
            crosssection_img = np.pad(crosssection_img,((15,0),(39,39),(0,0)), mode='constant')
        plt.imsave(observation_path[observation]['png'],observation_img)       
        plt.imsave(observation_path[observation]['projection_png'],projection_img)
        plt.imsave(observation_path[observation]['crosssection'],crosssection_img)

        #Annotate the images
        print('Annotating images...')
        annotate = run(["python",
                        f'{os.environ["userprofile"]}\\git\\hernia-repair\\Prediction.py',
                        observation,
                        observation_path[observation]['dcm_dir'],
                        observation_path[observation]['tif'],
                        observation_path[observation]['png']
                        ])
        
        hernia_helper.annotate_translation_image(observation_path[observation])
        hernia_helper.annotate_crosssection(observation_path[observation],max_translation_layer)
    
    #Patching Images togther for presentation
    os.system('cls')
    print('Arranging Results...')

    #Load all images
    sam_img = plt.imread(f'{path_to_evaluation}\\Verschiebung und Verzerrung.png')
    nat_img = plt.imread(observation_path['Nativ']['png'])[:,:,:3]
    nat_proj_img = plt.imread(observation_path['Nativ']['projection_png'])[:,:,:3]
    val_img = plt.imread(observation_path['Valsalva']['png'])[:,:,:3]
    val_proj_img = plt.imread(observation_path['Valsalva']['projection_png'])[:,:,:3]
    nativ_crosssection = plt.imread(observation_path['Nativ']['crosssection'])[:,:,:3]
    valsalva_crosssection = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]
    
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
        main_folder = config.path_names['main']
        if not os.path.exists(main_folder):
            os.mkdir(main_folder) 

        #Set the logging Config
        logging.basicConfig(filename= f'.\\Debug.log', level=logging.DEBUG)

        #Check for updates and update the neural nets
        try:
            network_folder = config.path_names['neuralnet']
            if not os.path.exists(network_folder):
                os.mkdir(network_folder)
            hernia_helper.update_neural_nets()
        except:
            print('Could not update neuralnets! Check your internet connection.\n', 'Start with old ones.')

        if mode == "Single":
            hernia_analysis(main_folder=main_folder,mode=mode)
        elif mode == "Multi":
            #open txt file with paths to the data
            txt_file = open(config.path_names['multipath'],'r',encoding='utf8')
            #read the first emptyline
            line = txt_file.readline()
            #loop over the file till EOF is reached
            while line:
                #get the starting time of the current iteration
                single_case_start_time = datetime.now()
                hernia_analysis(txt_file.readline()[1:-2],txt_file.readline()[1:-2],mode)
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
    
    
    
    
