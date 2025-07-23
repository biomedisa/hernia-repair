#!/usr/bin/env python
# coding: utf-8

#imports
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import shutil
from datetime import datetime
from subprocess import DEVNULL, PIPE, Popen
import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askinteger
from tkinter import messagebox as mb
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tifffile import imread

try:
    import config as config
except:
    import config_base as config

import create_mesh
import hernia_helper

consol_width = 120

def update_neural_network():
    try:
        network_folder = config.path_names['neuralnet']
        if not os.path.exists(network_folder):
            os.mkdir(network_folder)
        hernia_helper.update_neural_nets()
    except:
        print(f'{"Could not update neural network! Check your internet connection.":^{consol_width}}\n')
        print(f'{"Starting with the old one.":^{consol_width}}\n\n')

def timedelta(start_time):
    delta = datetime.now() - start_time
    formatted_time = "{:02}:{:02}:{:02}".format(delta.seconds // 3600, (delta.seconds // 60) % 60, delta.seconds % 60)
    return formatted_time

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

def hernia_analysis(rest=None, valsalva=None, threshold=15,
        force=False, logger=None, save_level=1,
        displacement_field=None, scaling=3):
    '''
    Parameters
    ----------
    rest: string
        Location of Rest directory containing dicom files
    valsalva: string
        Location of Valsalva directory containing dicom files
    threshold: int
        Threshold indicating high displacement of the abdominal wall
    force: boolean
        Force registration. Ignore differences in dimensions and voxel spacing
    save_level: int
        Spcifies amount of data saved, 1=results, 2=all
    displacement_field: string
        Location of displacement field. Skips registration if specified
    scaling: int
        Scaling factor to reduce calculation time during registration
    '''
    # set the timer
    instance_start_time = datetime.now()

    # starting data loading
    if logger is None:
        logger = config_logger()
    logger.info(f'{" Loading Data ":-^{consol_width}}')

    # initialize observations
    Observations = ["Rest","Valsalva"]
    observation_dict = {observation:{} for observation in Observations}
    observation_path = {observation:{} for observation in Observations}
    observation_path['Rest']['dcm_dir'] = rest
    observation_path['Valsalva']['dcm_dir'] = valsalva

    # load image data
    observation_dict['Rest']['data'], observation_dict['Rest']['header'] = hernia_helper.load_mask_data(rest)
    observation_dict['Valsalva']['data'], observation_dict['Valsalva']['header'] = hernia_helper.load_mask_data(valsalva)

    # main directory
    main_folder = config.path_names['main']
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)

    # create result directories
    first_level = hernia_helper.create_patient_directory_auto(rest, main_folder)
    day_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_to_evaluation = f'{first_level}/Evaluation_{day_string}'
    os.mkdir(path_to_evaluation)
    path_to_archive = f'{first_level}/Archive_for_Debugging_{day_string}'
    os.mkdir(path_to_archive)

    # create paths to the data
    for observation in Observations:
        observation_path[observation]['labels'] = f'{path_to_archive}/{observation}_labels.tif'
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

        # get image dimensions
        observation_path[observation]['z_sh'],\
        observation_path[observation]['y_sh'],\
        observation_path[observation]['x_sh'] = hernia_helper.get_data_shape(observation_path[observation]['dcm_dir'])

        # get voxel spacing
        observation_path[observation]['z_spacing'],\
        observation_path[observation]['y_spacing'],\
        observation_path[observation]['x_spacing'] = hernia_helper.get_pixel_spacing(observation_path[observation]['dcm_dir'])

    # selection Checks (Are the Dimensions of Rest and Valsalva the same)
    if observation_path['Rest']['z_spacing'] != observation_path['Valsalva']['z_spacing']:
        logger.warning(f'{"Rest and Valsalva slice thickness missmatch!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(force)
    elif observation_path['Rest']['y_spacing'] != observation_path['Valsalva']['y_spacing']:
        logger.warning(f'{"Rest and Valsalva depth missmatch (y-dim)!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(force)
    elif observation_path['Rest']['x_spacing'] != observation_path['Valsalva']['x_spacing']:
        logger.warning(f'{"Rest and Valsalva width missmatch (x-dim)!":^{consol_width}} \n {"This will impact the result.":^{consol_width}} \n\n')
        hernia_helper.ask_continue(force)
    # test for same amount of slices
    hernia_helper.compare_slice_amount(observation_path['Rest']['dcm_dir'],observation_path['Valsalva']['dcm_dir'],force)

    # log data info
    logger.info(f'{f"Patient Directory: {first_level}":^{consol_width}}')
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

    # print setup time
    logger.info(f'{f" Time for Setup: {timedelta(instance_start_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Computing Labels with Biomedisa
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Segmenting Labels ":=^{consol_width}}\n')

    # create the biomedisa labels and the mask for the registration for rest and valsalva
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # create the segmentation
        from biomedisa.deeplearning import deep_learning
        from biomedisa.features.biomedisa_helper import save_data
        observation_dict[observation]['labels'] = deep_learning(observation_dict[observation]['data'],
                    path_to_model=f'{config.path_names["neuralnet"]}/img_hernie.h5',
                    predict=True)['regular']

        # save result
        if save_level==2:
            save_data(observation_path[observation]['labels'], observation_dict[observation]['labels'])

    logger.info(f'{f" Time for Labels: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Creating masks of the body for registration
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Creating Masks ":=^{consol_width}}\n')
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # create mask from image data
        observation_dict[observation]['mask'] = hernia_helper.create_mask(
            observation_dict[observation]['data'], observation_dict[observation]['header'])

        # save result
        save_data(observation_path[observation]['mask'], observation_dict[observation]['mask'])

    logger.info(f'{f" Time for Masks: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Creating the Displacement Array using DIPY
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Calculating Registration ":=^{consol_width}}\n')

    # calculate displacement field
    if displacement_field:
        displacement_field = imread(displacement_field)
    else:
        displacement_field = hernia_helper.create_displacement_array(
            observation_dict['Rest']['mask'], observation_dict['Valsalva']['mask'],
            observation_path['Rest']['z_spacing'], observation_path['Rest']['y_spacing'],
            observation_path['Rest']['x_spacing'], scaling)
        
    # magnitude of the displacement field
    outward_field, inward_field = displacement_field[0], displacement_field[1]
    observation_dict['Rest']['displacement_array'] = np.sqrt(outward_field[...,2]**2 + outward_field[...,1]**2 + outward_field[...,0]**2)
    observation_dict['Valsalva']['displacement_array'] = np.sqrt(inward_field[...,2]**2 + inward_field[...,1]**2 + inward_field[...,0]**2)

    # save results
    if save_level==2:
        save_data(f'{path_to_archive}/displacement_field.tif', displacement_field, compress=False)
        save_data(observation_path['Rest']['displacement_array'], observation_dict['Rest']['displacement_array'], compress=False)
        save_data(observation_path['Valsalva']['displacement_array'], observation_dict['Valsalva']['displacement_array'], compress=False)

    logger.info(f'{f" Time for Registration: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Creating the vtk meshes for 3D visualization
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Creating Meshes ":=^{consol_width}}\n')
    # create meshes in vtk format for Paraview
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # voxel spacing
        z_spacing=observation_path[observation]['z_spacing']
        y_spacing=observation_path[observation]['y_spacing']
        x_spacing=observation_path[observation]['x_spacing']

        # mesh of the labels
        create_mesh.CreateVTK(observation_path[observation]['labels_vtk'],
            mode='labels', image=observation_dict[observation]['labels'],
            z_spacing=z_spacing, y_spacing=y_spacing, x_spacing=x_spacing)

        # mesh of the displacement
        observation_path[observation]['displacement_areas'] = create_mesh.CreateVTK(
            observation_path[observation]['displacement_vtk'],
            mode='displacement', image=observation_dict[observation]['mask'],
            displacement_array=observation_dict[observation]['displacement_array'],
            z_spacing=z_spacing, y_spacing=y_spacing, x_spacing=x_spacing)

        # mesh of the strain
        create_mesh.CreateVTK(observation_path[observation]['strain_vtk'],
            mode='strain', image=observation_dict[observation]['mask'],
            outward_field=outward_field, inward_field=inward_field,
            z_spacing=z_spacing, y_spacing=y_spacing, x_spacing=x_spacing,
            observation=observation)

    logger.info(f'{f" Time for Meshes: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Creating the images of the results
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Creating Screenshots ":=^{consol_width}}\n')

    # adjust the length of the threshold depending displacement lists to be the same for Rest and Valsalva
    displacement_dif = len(observation_path['Valsalva']['displacement_areas']) - len(observation_path['Rest']['displacement_areas'])
    if displacement_dif > 0:
        observation_path['Rest']['displacement_areas'] = np.pad(observation_path['Rest']['displacement_areas'], (0,displacement_dif))
    elif displacement_dif < 0:
        observation_path['Valsalva']['displacement_areas'] = np.pad(observation_path['Valsalva']['displacement_areas'], (0,-displacement_dif))

    # get patient-specific threshold for unstable abdominal wall
    #threshold = hernia_helper.plot_individual_threshold(observation_path,path_to_archive,threshold)
    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        #observation_path[observation]['z_centroid'],\
        #observation_path[observation]['y_centroid'],\
        #observation_path[observation]['x_centroid'] = hernia_helper.get_centroid(observation_dict[observation]['mask'])

        # create images using Paraview
            # image of the neural network projection
        screenshot1 = Popen([sys.executable,
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
        screenshot2 = Popen([sys.executable,
                            f'{config.BASE_DIR}/x_ray.py',
                            observation_path[observation]['displacement_vtk'],
                            observation_path[observation]['labels_vtk'],
                            f'{path_to_evaluation}/{observation}_x_ray.png',
                            str(threshold)],
                            stdin=DEVNULL, stderr=PIPE, stdout=PIPE)
        _,screenshot2_error = screenshot2.communicate()
        logger.debug(f'{screenshot2_error}')

    logger.info(f'{f" Time for Screenshots: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Preprocess and annotate all images
    ###########################################################################
    step_time = datetime.now()
    logger.info(f'{" Annotating images ":=^{consol_width}}\n')

    for observation in Observations:
        logger.info(f'{f" Processing {observation} ":-^{consol_width}}')

        # get the crossection image as the layer with the biggest offset between rest and valsalva
        max_displacement_layer = hernia_helper.create_crosssection(
            observation_path[observation]['crosssection'], observation_dict[observation]['data'],
            observation_dict[observation]['labels'], observation_dict[observation]['displacement_array'])

        # annotate images
        hernia_helper.annotate_label_image(observation, observation_dict[observation], observation_path[observation])
        hernia_helper.annotate_displacement_image(observation, observation_dict[observation], observation_path[observation], threshold)
        hernia_helper.annotate_crosssection(observation, observation_path[observation], max_displacement_layer)
        hernia_helper.annotate_strain_image(observation, observation_path[observation])

    logger.info(f'{f" Time for Annotations: {timedelta(step_time)} ":_^{consol_width}}\n\n')

    ###########################################################################
    # Arranging results
    ###########################################################################
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
    path_to_result = f'{path_to_evaluation}/Combined_Results.png'
    plt.imsave(path_to_result, combined_img, dpi=300)
    open_file(path_to_result)

###############################################################################
# Main Loop
###############################################################################

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='HEDI: Hernia Evaluation, Detection, and Imaging',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # optional arguments
    parser.add_argument('-r','--rest', type=str, metavar='PATH', default=None,
                        help='Location of Rest directory containing dicom files')
    parser.add_argument('-v','--valsalva', type=str, metavar='PATH', default=None,
                        help='Location of Valsalva directory containing dicom files')
    parser.add_argument('-df','--displacement_field', type=str, metavar='PATH', default=None,
                        help='Location of displacement field. Skips registration if specified')
    parser.add_argument('-dd','--dicom_data', type=str, metavar='PATH', default=None,
                        help='Location of unstructured dicom data')
    parser.add_argument('-t', '--threshold', type=int, default=15,
                        help='Threshold indicating high displacement of the abdominal wall')
    parser.add_argument('-s', '--scaling', type=int, default=3,
                        help='Scaling factor to reduce calculation time during registration')
    parser.add_argument('-sl', '--save_level', type=int, default=1,
                        help='Spcifies amount of data saved, 1=results, 2=all')
    parser.add_argument('-f','--force', action='store_true', default=False,
                        help='Force registration. Ignore differences in dimensions and voxel spacing')
    args = parser.parse_args()

    # define Logger for debugging
    args.logger = config_logger()
    tk.Tk().withdraw()

    # console
    print(f'{" H E D I ":=^{consol_width}}')
    print(f'{"Hernia Evaluation, Detection, and Imaging": ^{consol_width}}')
    print(f'{"":=^{consol_width}}\n\n')

    # logger
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args.logger.info(f'{" Starting Setup ":=^{consol_width}}')
    args.logger.info(f'{f"{start_time}":^{consol_width}}\n\n')

    # initial download of neural network
    if not os.path.exists(f'{config.path_names["neuralnet"]}/img_hernie.h5'):
        args.logger.info(f'{" Downloading neural network ":-^{consol_width}}')
        update_neural_network()

    # update HEDI and Biomedisa
    #update = mb.askquestion(title='Update', message='Update HEDI?')
    update = 'no'
    if update=='yes':
        # update neural network
        args.logger.info(f'{" Updating neural network ":-^{consol_width}}')
        update_neural_network()
        # update biomedisa
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "biomedisa"])
        # update hedi
        subprocess.check_call(["git", "pull"], cwd=config.BASE_DIR)
        print('Update complete. Please restart HEDI.')

    else:
        # load dicom data and build patient directory
        if not (args.rest and args.valsalva):
            main_folder = config.path_names['main']
            if not os.path.exists(main_folder):
                os.mkdir(main_folder)
            first_level = hernia_helper.load_directorys(main_folder=main_folder, path_to_dir=args.dicom_data)
            args.rest = askdirectory(initialdir=first_level, title=f'Select Rest Directory')
            args.valsalva = askdirectory(initialdir=first_level, title=f'Select Valsalva Directory')
            args.threshold = askinteger(title='Instability threshold', prompt='Threshold:', initialvalue=15)

        # get time to measure execution time
        total_start_time = datetime.now()

        # hernia analysis
        del args.dicom_data
        hernia_analysis(**vars(args))
        args.logger.info(f'Execution time: {timedelta(total_start_time)}\n\n')

