#!/usr/bin/env python
# coding: utf-8

#imports
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import shutil
from datetime import datetime
from subprocess import DEVNULL, PIPE, STDOUT, run, Popen
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askinteger
import subprocess

import matplotlib.pyplot as plt
import numpy as np

try:
    import config as config
except:
    import config_base as config
    
import create_mesh
import hernia_helper

consol_width = 120

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def config_logger(name='Timer'):
    # set the logging Config
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('%(message)s')

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)

    f_handler = logging.FileHandler(f'{config.BASE_DIR}/data/Debug.log')
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
    return logger

def hernia_analysis(main_folder, path_to_rest=None, path_to_valsalva=None, mode='Single', threshold=15):

    if sys.platform == "win32":
        python = 'python'
    else:
        python = 'python3'

    # set the timer 
    instance_start_time = datetime.now()

    # starting data loading
    logger.info(f'{" Loading Data ":-^{consol_width}}')

    # set the paths for both observations
    Observations = ["Rest","Valsalva"]
    observation_path = {observation:{} for observation in Observations}


    if path_to_rest != None:
        first_level = hernia_helper.create_patient_directory_auto(path_to_rest,main_folder)
        # create and get the patients working directory
        observation_path['Rest']['dcm_dir'] = path_to_rest 
        observation_path['Valsalva']['dcm_dir'] = path_to_valsalva

    else:
        # create and get the patients working directory
        first_level = hernia_helper.load_directorys(main_folder=main_folder)
        for observation in sorted(Observations):
            observation_path[observation]['dcm_dir'] = askdirectory(initialdir = first_level, title=f'Select {observation} Directory')
        threshold = askinteger(title='Instability threshold', prompt='Change threshold manually:', initialvalue=15)

    # console output
    logger.info(f'{f"Using Data from: {first_level}":^{consol_width}}')

    # set Time String for saving the data
    day_string = datetime.now().strftime("%Y-%m-%d_%H-%M")

    path_to_evaluation = f'{first_level}/Evaluation_{day_string}'
    os.mkdir(path_to_evaluation)

    path_to_archive    = f'{first_level}/Archive_for_Debugging_{day_string}'
    os.mkdir(path_to_archive)

    for observation in sorted(Observations):
        # create Paths to the mesh and the img
        observation_path[observation]['labels']  = f'{path_to_archive}/{observation}_labels.tif'
        observation_path[observation]['mask'] = f'{path_to_archive}/{observation}_mask.tif'
        observation_path[observation]['displacement_array'] = f'{path_to_archive}/{observation}_displacement_array.tif'
        observation_path[observation]['strain_array'] = f'{path_to_archive}/{observation}_strain_array.tif' 

        observation_path[observation]['labels_vtk'] = f'{path_to_archive}/{observation}_labels.vtk'
        observation_path[observation]['displacement_vtk'] = f'{path_to_archive}/{observation}_displacement.vtk'
        observation_path[observation]['strain_vtk'] = f'{path_to_archive}/{observation}_strain.vtk'

        observation_path[observation]['labels_png'] = f'{path_to_evaluation}/{observation}_labels.png'
        observation_path[observation]['displacement_png'] = f'{path_to_evaluation}/{observation}_displacement.png'
        observation_path[observation]['strain_png'] = f'{path_to_evaluation}/{observation}_strain.png'
        observation_path[observation]['crosssection'] = f'{path_to_evaluation}/{observation}_crosssection.png'

        observation_path[observation]['z_sh'],\
        observation_path[observation]['y_sh'],\
        observation_path[observation]['x_sh'] = hernia_helper.get_data_shape(observation_path[observation]['dcm_dir'])

        observation_path[observation]['z_spacing'],\
        observation_path[observation]['y_spacing'],\
        observation_path[observation]['x_spacing'] = hernia_helper.get_pixel_spacing(observation_path[observation]['dcm_dir'])

    # selection Checks (Are the Dimensions of Rest and Valsalva the same)
    if observation_path['Rest']['z_spacing'] != observation_path['Valsalva']['z_spacing']:
        logger.warning(f'{"Rest and Valsalva slice thickness missmatch!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(mode)

    elif observation_path['Rest']['y_spacing'] != observation_path['Valsalva']['y_spacing']:
        logger.warning(f'{"Rest and Valsalva depth missmatch (y-dim)!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(mode)

    elif observation_path['Rest']['x_spacing'] != observation_path['Valsalva']['x_spacing']:
        logger.warning(f'{"Rest and Valsalva width missmatch (x-dim)!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(mode)

    # test for same amount of slices
    hernia_helper.compare_slice_amount(observation_path['Rest']['dcm_dir'],observation_path['Valsalva']['dcm_dir'],mode)

    Info_file = open(f'{path_to_archive}/Info.txt','w')
    Info_file.writelines([  
                    f'Info for Rest Data:\n',
                    f'      Data used: \"{observation_path["Rest"]["dcm_dir"]}\"\n',
                    f'      Data shape (x,y,z): ({observation_path["Rest"]["x_sh"]},{observation_path["Rest"]["y_sh"]},{observation_path["Rest"]["z_sh"]})\n',
                    f'      Pixel spacing (x,y,z): ({observation_path["Rest"]["x_spacing"]},{observation_path["Rest"]["y_spacing"]},{observation_path["Rest"]["z_spacing"]})\n'
                    f'Info for Valsalva Data:\n',
                    f'      Data used: \"{observation_path["Valsalva"]["dcm_dir"]}\"\n',
                    f'      Data shape (x,y,z): ({observation_path["Valsalva"]["x_sh"]},{observation_path["Valsalva"]["y_sh"]},{observation_path["Valsalva"]["z_sh"]})\n', 
                    f'      Pixel spacing (x,y,z): ({observation_path["Valsalva"]["x_spacing"]},{observation_path["Valsalva"]["y_spacing"]},{observation_path["Valsalva"]["z_spacing"]})\n'
                    f'Threshold: {threshold}mm\n'
                        ])
    Info_file.close()


    step_time = datetime.now()
    logger.info(f'{f" Time for Setup: {step_time - instance_start_time} ":_^{consol_width}}\n\n')


    ###########################################################################
    # Computing Labels with Biomedisa
    ###########################################################################
    
    logger.info(f'{" Computing Labels ":=^{consol_width}}\n')

    # create the biomedisa labels and the mask for the registration for rest and valsalva
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # create the segmentation proposal, in form of a tif
        net = Popen([python,
                    f'{config.path_names["userprofile"]}/git/biomedisa/demo/biomedisa_deeplearning.py',
                    observation_path[observation]["dcm_dir"],
                    f'{config.path_names["neuralnet"]}/img_hernie.h5', "-p", "-bs", "6"],
                    stdin=DEVNULL, stderr=PIPE, stdout=PIPE)
        _, net_error = net.communicate()
        logger.debug(f'{net_error}')

        # move the segmentation propasal into the correct folder
        temp_path_to_tif = os.path.splitext(observation_path[observation]["dcm_dir"])[0]
        shutil.move(f'{os.path.dirname(temp_path_to_tif)}/final.{os.path.basename(temp_path_to_tif)}.tif',
                    observation_path[observation]["labels"])

        logger.info(f'{f" Time for {observation} labeling: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
        step_time = datetime.now()


    ###########################################################################
    # Creating masks of the body for registration
    ###########################################################################
    
    logger.info(f'{" Creating masks ":=^{consol_width}}\n')
    
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        hernia_helper.create_mask(observation_path[observation])

        logger.info(f'{f" Time for {observation} mask creation: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
        step_time = datetime.now()


    ###########################################################################
    # Creating the Displacement Array using DIPY
    ###########################################################################
    
    logger.info(f'{" Creating arrays of displacement and strain ":=^{consol_width}}\n')

    hernia_helper.create_displacement_array(observation_path)

    logger.info(f'{f" Time for registration: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
    step_time = datetime.now()


    ###########################################################################
    # Creating the vtk meshes for 3D visualization
    ###########################################################################
    
    logger.info(f'{" Creating Meshes ":=^{consol_width}}\n')

    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # create meshes, in vtk format for Paraview
            # mesh of the muscle labels
        create_mesh.CreateVTK(observation_path[observation], 'labels')

        logger.info(f'{f"Time for {observation} label mesh: {datetime.now() - step_time}":^{consol_width}}')
        step_time = datetime.now()

            # mesh of the displacement 
        observation_path[observation]['displacement_areas'] = create_mesh.CreateVTK(observation_path[observation], 'displacement')

        logger.info(f'{f"Time for {observation} displacement mesh: {datetime.now() - step_time}":^{consol_width}}')
        step_time = datetime.now()

            # mesh of the strain 
        create_mesh.CreateVTK(observation_path[observation], 'strain')

        logger.info(f'{f" Time for {observation} strain mesh: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
        step_time = datetime.now()

    ###########################################################################
    # Creating the images of the results
    ###########################################################################
    
    logger.info(f'{" Creating images ":=^{consol_width}}\n')

    #adjust the length of the threshold depending displacement lists to be the same for Rest and Valsalva
    displacement_dif = len(observation_path['Valsalva']['displacement_areas']) - len(observation_path['Rest']['displacement_areas'])
    if displacement_dif > 0:
        observation_path['Rest']['displacement_areas'] = np.pad(observation_path['Rest']['displacement_areas'], (0,displacement_dif))
    elif displacement_dif < 0:
        observation_path['Valsalva']['displacement_areas'] = np.pad(observation_path['Valsalva']['displacement_areas'], (0,-displacement_dif))
    
    # get patient-specific threshold for unstable abdominal wall
    threshold = hernia_helper.plot_individual_threshold(observation_path,path_to_archive,threshold)



    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        observation_path[observation]['z_centroid'],\
        observation_path[observation]['y_centroid'],\
        observation_path[observation]['x_centroid'] = hernia_helper.get_centroid(observation_path[observation]['mask'])

        # create images using Paraview
            # image of the neural network projection
        screenshot1 = Popen([python,
                            f'{config.BASE_DIR}/paraview_screenshot.py',
                            observation_path[observation]['displacement_vtk'],
                            observation_path[observation]['strain_vtk'],
                            observation_path[observation]['labels_vtk'],
                            f'{path_to_evaluation}/{observation}',
                            str(threshold)],
                            
                            stdin=DEVNULL, stderr=PIPE, stdout=PIPE)
        _,screenshot1_error = screenshot1.communicate()
        logger.debug(f'{screenshot1_error}')

            # image of the combination of labels and displacement projection
        screenshot2 = Popen([python,
                            f'{config.BASE_DIR}/x_ray.py',
                            observation_path[observation]['displacement_vtk'],
                            observation_path[observation]['labels_vtk'],
                            f'{path_to_evaluation}/{observation}_x_ray.png',
                            str(threshold)],

                            stdin=DEVNULL, stderr=PIPE, stdout=PIPE)
        _,screenshot2_error = screenshot2.communicate()
        logger.debug(f'{screenshot2_error}')

        logger.info(f'{f" Time for {observation} screenshots: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
        step_time = datetime.now()


    ###########################################################################
    # Preprocess and annotate all images
    ###########################################################################

    logger.info(f'{" Annotating images ":=^{consol_width}}\n')

    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # get the crossection image as the layer with the biggest offset between rest and valsalva
        max_displacement_layer = hernia_helper.create_crosssection(observation_path[observation])

        # annotate images
        hernia_helper.annotate_label_image(observation, observation_path[observation])
        hernia_helper.annotate_displacement_image(observation, observation_path[observation], observation_path[observation]['displacement_areas'][threshold], threshold)
        hernia_helper.annotate_crosssection(observation_path[observation], max_displacement_layer)
        hernia_helper.annotate_strain_image(observation, observation_path[observation])

        logger.info(f'{f" Time for {observation} annotations: {datetime.now() - step_time} ":_^{consol_width}}\n\n')
        step_time = datetime.now()


    ###########################################################################
    # Arangeing the results
    ###########################################################################

    # combine images for final result
    logger.info(f'{" Arranging Results ":=^{consol_width}}\n')

    # plot areas of displacement with respect to their magnitute
    hernia_helper.plot_displacement(observation_path, path_to_archive)
    hernia_helper.plot_displacement_lower(observation_path, path_to_archive)
    hernia_helper.plot_displacement_difference(observation_path, path_to_archive)

    # load all images
    rest_label_img        = plt.imread(observation_path['Rest']['labels_png'])[:,:,:3]
    rest_displacement_img = plt.imread(observation_path['Rest']['displacement_png'])[:,:,:3]
    rest_strain_img       = plt.imread(observation_path['Rest']['strain_png'])[:,:,:3]
    rest_crosssection     = plt.imread(observation_path['Rest']['crosssection'])[:,:,:3]
    val_label_img         = plt.imread(observation_path['Valsalva']['labels_png'])[:,:,:3]
    val_displacement_img  = plt.imread(observation_path['Valsalva']['displacement_png'])[:,:,:3] #2
    val_strain_img        = plt.imread(observation_path['Valsalva']['strain_png'])[:,:,:3]       #2
    val_crosssection      = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]

    # reshape to fit and add boarders
    rest_label_img        = np.pad(rest_label_img,       ((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1)))
    val_label_img         = np.pad(val_label_img,        ((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1)))
    rest_displacement_img = np.pad(rest_displacement_img,((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1)))
    val_displacement_img  = np.pad(val_displacement_img, ((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1))) #2
    rest_strain_img       = np.pad(rest_strain_img,      ((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1)))
    val_strain_img        = np.pad(val_strain_img,       ((5,5),(5,5),(0,0)), mode='constant', constant_values=((0,0),(0,0),(1,1))) #2
    rest_crosssection     = np.pad(rest_crosssection,    ((5,5),(0,0),(0,0)), mode='constant', constant_values=((0,1),(0,0),(1,1)))
    val_crosssection      = np.pad(val_crosssection,     ((5,5),(0,0),(0,0)), mode='constant', constant_values=((1,0),(0,0),(1,1)))


    # stack the image pairs
    #displacement_strain = np.vstack((rest_displacement_img,  rest_strain_img)) #new
    double_strain       = np.vstack((rest_strain_img,        val_strain_img)) #2
    double_displacement = np.vstack((rest_displacement_img,  val_displacement_img)) #2
    rest_and_val        = np.vstack((rest_label_img,         val_label_img))
    double_crosssection = np.vstack((rest_crosssection,      val_crosssection))
    # stack result, paraview images and crosssections 
    # combined_img  = np.hstack((displacement_strain, rest_and_val, double_crosssection)) #new
    first_part   = np.hstack((double_strain, double_displacement)) #2
    second_part  = np.hstack((first_part,    rest_and_val)) #2
    combined_img = np.hstack((second_part,   double_crosssection)) #2

    # save image in evaluation directory
    plt.imsave(f'{path_to_evaluation}/Combined_Results.png',combined_img)

    if mode == "Single":
        # show final result
        open_file(f'{path_to_evaluation}/Combined_Results.png')
    ###########################################################################




###############################################################################
#Main Loop 
###############################################################################

# try loop in case of error
try:
    if __name__ == "__main__":
        # define Logger for debugging
        logger = config_logger()

        # clear the consoloutput
        if sys.platform == "win32":
            os.system('cls')
        else:
            os.system('clear')
        print(f'{" H E D I ":=^{consol_width}}')
        print(f'{"Hernia Evaluation, Detection and Imaging": ^{consol_width}}')
        print(f'{"":=^{consol_width}}\n\n')

        logger.info(f'{" Starting Setup ":=^{consol_width}}')
        logger.info(f'{f"{datetime.now()}":^{consol_width}}\n\n')
        logger.info(f'{" Updating neuralnetwork ":-^{consol_width}}')


        # check for updates and update the neural nets
        try:
            network_folder = config.path_names['neuralnet']
            if not os.path.exists(network_folder):
                os.mkdir(network_folder)
            hernia_helper.update_neural_nets()
        except:
            print(f'{"Could not update neuralnetwork! Check your internet connection.":^{consol_width}}\n')
            print(f'{"Starting with the old one.":^{consol_width}}\n\n')

        #######################################################################

        # get time to measure execution time
        total_start_time = datetime.now()
        
        # get the operation mode
        mode = sys.argv[1]
        logger.debug(f'Starting in Mode: {mode}')

        # set the main save directory
        main_folder = config.path_names['main']
        if not os.path.exists(main_folder):
            os.mkdir(main_folder)

        # default threshold
        threshold = 15

        if mode == "Single":
            hernia_analysis(main_folder=main_folder,mode=mode,threshold=threshold)

        elif mode == "CMD":
            path_to_rest=sys.argv[2]
            path_to_valsalva=sys.argv[3]
            if len(sys.argv)==5:
                threshold=int(sys.argv[4])
            hernia_analysis(main_folder,path_to_rest,path_to_valsalva,"Single",threshold)

        elif mode == "Multi":
            # user defined threshold
            if len(sys.argv)==3:
                threshold=int(sys.argv[2])

            # open txt file with paths to the data
            txt_file = open(config.path_names['multipath'],'r',encoding='utf8')

            # read the first emptyline
            line = txt_file.readline()

            # loop over the file till EOF is reached
            while line:
                # get the starting time of the current iteration
                single_case_start_time = datetime.now()
                if sys.platform == "win32":
                    os.system('cls')
                else:
                    os.system('clear')
                print(f'{" H E D I ":=^{consol_width}}')
                print(f'{"Hernia Evaluation, Detection and Imaging": ^{consol_width}}')
                print(f'{"":=^{consol_width}}\n\n')
                
                logger.info(f'{f" Starting Time: {datetime.now()} ":=^{consol_width}}\n\n')

                path_to_rest = txt_file.readline()[1:-2]
                path_to_valsalva = txt_file.readline()[1:-2]
                if os.path.exists(path_to_rest) and os.path.exists(path_to_valsalva):
                    try:
                        hernia_analysis(main_folder,path_to_rest,path_to_valsalva,mode,threshold)
                    except Exception as e:
                        logger.info(f'Error: {e}\n\n')
                else:
                    logger.info(f'Error: data does not exist.\n\n')

                # get the end time of the current itteration
                single_case_end_time = datetime.now()

                # log the used time for the current itteration
                logger.info(f'Execution time: {single_case_end_time - single_case_start_time}\n\n')

                # skip the next empty line in the txt file
                line = txt_file.readline()

            # close the file
            txt_file.close()

        else: raise ValueError('Wrong operation mode. Must be one of "Single", "Multi" or "CMD"')

        total_end_time = datetime.now()
        logger.info(f'Execution time: {total_end_time - total_start_time}\n\n')

# catch the error and log it to a temp file
except: 
    # open the error txt file to write to
    logger.exception('Fehler bei der Ausführung!')

###############################################################################

