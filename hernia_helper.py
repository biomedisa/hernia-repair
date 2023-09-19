#imports
import glob
import logging
import os
import shutil
import ssl
import time
import tkinter as tk
import urllib.request
from datetime import datetime
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import requests
from dateutil import tz
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.measure import label, find_contours
from tifffile import imread, imwrite
from tqdm import tqdm

try:
    import config as config
except:
    import config_base as config

###############################################################################
# Functions for startup, dataloading and assertion_check
###############################################################################

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

    print(f'{"Do You want to continue [y] or close the application [n]?":^120}')
    user_input = input()
    if user_input in ['y','yes', 'ja' ]:
        print(f'{"Continuing...":^120}\n')
        return True
    elif user_input in ['n', 'no', 'nein']:
        print(f'{"Shuting down!":^120}\n')
        exit()
    else: 
        print(f'{"Please enter only [y] or [n]!":^120}')
        ask_continue(mode)


def update_neural_nets():
    '''
    Updates the neuralnetworks by checking if there are any updates on the server.
    '''

    sources = ['https://biomedisa.org/media/img_hernie.h5']

    destinations = [f'{config.path_names["neuralnet"]}/img_hernie.h5']

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
            print(f'{f"Web:{t1}": ^120}')

            # destination
            timestamp2 = os.path.getmtime(dst)
            timestamp2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp2))
            t2 = datetime.strptime(timestamp2, '%Y-%m-%d %H:%M:%S')
            t2 = t2.astimezone(tz.tzlocal())
            print(f'{f"Local:{timestamp2}":_^120}\n\n')
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

    # ask the user for the Path to the Data via Tkinterface
    tk.Tk().withdraw()
    path_to_dir = askdirectory(title='Select Dataset')

    # get the raw dcm files
    files = glob.glob(path_to_dir + '/**/*', recursive=True)

    # set patients directory
    for file in files:
        if os.path.isfile(file):
            ds = pydicom.filereader.dcmread(file)
            PatientBirthDate = str(ds.PatientBirthDate)
            PatientName = str(ds.PatientName)
            PatientName = PatientName.replace('Ü','Ue')
            PatientName = PatientName.replace('Ä','Ae')
            PatientName = PatientName.replace('Ö','Oe') 
            PatientName = PatientName.replace('ü','ue')
            PatientName = PatientName.replace('ä','ae')
            PatientName = PatientName.replace('ö','oe')
            PatientName = PatientName.replace('ß','ss')

            StudyDate = str(ds.StudyDate)
            StudyDescription = str(ds.StudyDescription)

            # define Directories to store results and Dicom Data
            first_level = f'{main_folder}/{PatientName}'
            second_level = f'{first_level}/Dicom_Data_{StudyDate}'
            # check existence and create non existing directorys
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            if not os.path.exists(second_level):
                os.mkdir(second_level)
            break

    # loop over all files create subdirectories and copy the files
    for file in files:
        if os.path.isfile(file):
            ds = pydicom.filereader.dcmread(file)
            try:
                SeriesNumber = str(ds.SeriesNumber)
                SeriesDescription = str(ds.SeriesDescription)
            except:
                SeriesNumber      = "404"
                SeriesDescription = "No series description"
            # windows Name exceptions: /, \, :, *, ?, ", <, > and |
            SeriesDescription = SeriesDescription.replace('/',' ')
            SeriesDescription = SeriesDescription.replace('\\',' ')
            SeriesDescription = SeriesDescription.replace(':',' ')
            SeriesDescription = SeriesDescription.replace('*',' ')
            SeriesDescription = SeriesDescription.replace('?',' ')
            SeriesDescription = SeriesDescription.replace('"',' ')
            SeriesDescription = SeriesDescription.replace('<',' ')
            SeriesDescription = SeriesDescription.replace('>',' ')
            SeriesDescription = SeriesDescription.replace('|',' ')

            third_level = f'{second_level}/{SeriesNumber} {SeriesDescription}'

            if not os.path.exists(third_level):
                os.mkdir(third_level)

            # set the path to the current .dcm files and copy it 
            path_to_dest = f'{third_level}/{str(ds.InstanceNumber).zfill(6)}.dcm'
            if not os.path.exists(path_to_dest):
                shutil.copy(file, path_to_dest)

    logging.debug('Data Loaded successfully.')

    return first_level


def create_patient_directory_auto(dcm_dir,main_folder):
    '''
    Creates the main directory of a single Patient when programm is run in
    multi/auto mode.

    Parameters
    ----------
    dcm_dir: string
        Path to the directory of the .dcm datset
    main_folder: string
        Pathstring to the standard main directory

    Returns
    ----------
    string
        Pathstring to the patients directory
    '''

    # get the raw dcm files
    files = os.listdir(dcm_dir)
    # set patients directory 
    ds = pydicom.filereader.dcmread(f'{dcm_dir}/{files[1]}')
    PatientBirthDate = str(ds.PatientBirthDate)
    PatientName = str(ds.PatientName)
    PatientName = PatientName.replace('Ü','Ue')
    PatientName = PatientName.replace('Ä','Ae')
    PatientName = PatientName.replace('Ö','Oe') 
    PatientName = PatientName.replace('ü','ue')
    PatientName = PatientName.replace('ä','ae')
    PatientName = PatientName.replace('ö','oe')
    PatientName = PatientName.replace('ß','ss')

    # define Directories to store results and Dicom Data
    first_level = f'{main_folder}/{PatientName}'

    if not os.path.exists(first_level):
        os.mkdir(first_level)

    return first_level


def get_data_shape(dcm_dir):
    '''
    Returns the shape of the given tif data

    Parameter
    ---------
    dcm_dir: string
        Path to the directory of the .dcm datset

    Returns
    -------
    z_sh: string
        The shape in z-direction
    y_sh: string
        The shape in y-direction
    x_sh: string
        The shape in x-direction
    '''

    files = os.listdir(dcm_dir)
    # load a .dcm file of the datset and extract voxel side lengths
    ds = pydicom.filereader.dcmread(f'{dcm_dir}/{files[1]}')
    z_sh = len(files)
    y_sh = ds.Columns
    x_sh = ds.Rows

    return str(z_sh), str(y_sh), str(x_sh)


def get_pixel_spacing(dcm_dir):
    '''
    Returns the three voxel sidelengths of the given dataset.

    Parameter
    ---------
    dcm_dir: string
        Path to the directory of the .dcm dataset

    Returns
    -------
    z_res: string
        Voxel length in z-direction
    y_res: string
        Voxel length in y-direction
    x_res: string
        Voxel length in x-direction
    '''

    files = os.listdir(dcm_dir)
    # load a .dcm file of the datset and extract voxel side lengths
    ds = pydicom.filereader.dcmread(f'{dcm_dir}/{files[1]}')
    z_res = ds.SliceThickness
    y_res, x_res = ds.PixelSpacing

    return str(z_res), str(y_res), str(x_res)


def get_centroid(path_to_mask):
    '''
    Returns the coordinates of the centroid from a 3-dimensional binary array

    Parameters
    ---------
    path_to_mask: string
        Path to the binary mask in tifffile format

    Returns
    -------
    z: string
        z-coordinate of the centroid
    y: string
        y-coordinate of the centroid
    x: string
        x-coordinate of the centroid
    '''

    mask = imread(path_to_mask)
    z,y,x = ndimage.center_of_mass(mask)

    return str(z), str(y), str(x)


def compare_slice_amount(rest_dcm_dir,valsalva_dcm_dir,mode):
    '''
    Compare the amount of slices between the rest and valsalva Dataset.
    Prompts the user in case of a missmatch, when mode is Single.

    Parameters
    ----------
    rest_dcm_dir: string
        Path to the directory containing the .dcm data at rest
    valsalva_dcm_dir: string
        Path to the directory containing the .dcm data during valsalva
    mode: string
        The operation mode of HEDI one of 'Single' or 'Multi'

    Raises
    ------
    ask_continue()
        When the data has a missmatch
    '''

    rest_slice_amount = len(os.listdir(rest_dcm_dir))
    valsalva_slice_amount = len(os.listdir(valsalva_dcm_dir))
    if rest_slice_amount > valsalva_slice_amount:
        print(f'There are {rest_slice_amount - valsalva_slice_amount} more rest than valsalva({valsalva_slice_amount}) scans! \n This will impact the result. \n\n')
        logging.warning(f'There are {rest_slice_amount - valsalva_slice_amount} more rest than valsalva({valsalva_slice_amount}) scans! \n This will impact the result. \n\n')
        return ask_continue(mode)
    elif rest_slice_amount < valsalva_slice_amount:
        print(f'There are {valsalva_slice_amount - rest_slice_amount} more valsalva than rest({rest_slice_amount}) scans! \n This will impact the result. \n\n')
        logging.warning(f'There are {valsalva_slice_amount - rest_slice_amount} more valsalva than rest({rest_slice_amount}) scans! \n\n')
        return ask_continue(mode)


###############################################################################
# Creation of the mask for displacement detection
###############################################################################

def load_mask_data(dcm_dir):
    '''
    Retruns the pixel information stored in the .dcm files
    
    Parameters
    ----------
    dcm_dir: string
        Path to the directory of the .dcm dataset

    Returns 
        volume: nd.array
            Pixel information of the .dcm dataset
        header: list
            Header information of the .dcm dataset
    '''

    if os.path.isdir(dcm_dir):
        files = glob.glob(dcm_dir+'/**/*.dcm', recursive=True)
        for name in files:
            try:
                ds = pydicom.filereader.dcmread(name)
            except:
                files.remove(name)
        slice = ds.pixel_array
        volume = np.zeros((len(files), ds.Rows, ds.Columns), dtype=slice.dtype)
        header = [0] * len(files)
        MinInstanceNumber = np.inf
        for file_name in files:
            ds = pydicom.filereader.dcmread(file_name)
            if ds.InstanceNumber < MinInstanceNumber:
                MinInstanceNumber = ds.InstanceNumber
        for file_name in files:
            ds = pydicom.filereader.dcmread(file_name)
            slice = ds.pixel_array
            if len(slice.shape) == 3 and slice.shape[2] == 1:
                slice = slice[:,:,0]
            volume[ds.InstanceNumber-MinInstanceNumber] = slice
            header[ds.InstanceNumber-MinInstanceNumber] = ds
    else:
        volume, header = None, None
    return volume, header  


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
    # Image -> Adjust -> Threshold (-224 HU, -100==fat, 0==water, -1024==air)
    a = np.zeros_like(img)
    a[img>-224] = 1
    a[0] = 1
    a[-1] = 1
    a = fill(a, 0.9)
    a[0] = 0
    a[-1] = 0
    a = clean(a, 0.9)
    return a


def create_mask(observation_dict):
    '''
    Creates a mask of the abdominal region given by the dicom data.
    Pixels inside are labeled 1. Pixels outside 0.

    Parameters
    ----------
    observation_dict: dict of string
        A patients path dictionary for either the Rest or Valsalva data
        
    '''

    #load the dicom data
    data, header = load_mask_data(observation_dict['dcm_dir'])
    #adjust the data to the hounsfield range
    rescale_intercept = header[0].RescaleIntercept
    rescale_slope = header[0].RescaleSlope
    data = rescale_slope*data + rescale_intercept 
    #pad data with a blackpixels
    data = np.pad(data,((1,1),(256,256),(256,256)),constant_values=-1024)
    body_outline = threshold(data)
    #remove the padded area
    mask = body_outline[1:-1,256:-256,256:-256]
    #save the mask
    imwrite(observation_dict['mask'],mask,compression='zlib')


###############################################################################
# Creation of the displacement and strain arrays
###############################################################################

def get_strain_tensor(Ux,Uy):
    # initialize output matrix
    E = np.zeros([Ux.shape[0],Ux.shape[1],2,2])
    # displacement images gradients
    Uxy, Uxx = np.gradient(Ux)
    Uyy, Uyx = np.gradient(Uy)
    # the 2-D Eulerian strain tensor
    E[:,:,0,0] = 2*Uxx - (Uxx**2 + Uyx**2)
    E[:,:,0,1] = Uxy + Uyx - Uxx*Uxy - Uyx*Uyy
    E[:,:,1,0] = Uxy + Uyx - Uxy*Uxx - Uyy*Uyx
    E[:,:,1,1] = 2*Uyy - (Uxy**2 + Uyy**2)

    E = 0.5*E
    return E


def create_strain_layer(Ux, Uy):
    '''
    Create a layer of strain values corresponding to the displacment in direction 
    X and Y.

    Parameters
    ----------
        Ux: nd.array
            Array of the displacement in direction X
        Uy: nd.array
            Array of the displacement in direction Y

    Returns
    ----------
        strain_magnitude: nd.array
            Array of the strain values, same shape as Ux

    '''
    # calculate strain tensors
    E = get_strain_tensor(Ux,Uy)

    # compute a scalar comparison Value from the strain tensor
    # E_vgl = sqrt(E_xx^2 + E_yy^2 - E_xx*E_yy + 3*(E_xy^2))
    strain_magnitude = np.sqrt(
        E[:,:,0,0]**2 +
        E[:,:,1,1]**2 -
        E[:,:,0,0] * E[:,:,1,1] +
        3 * E[:,:,0,1]**2)

    return strain_magnitude


def symmetric_registration(static,moving):
    '''
    Create the displacement from moving to static and back,
    using symmetric diffemomorphic registration.
    The result is split into Y and X direction.

    Parameters
    ----------
    static: nd.array
        Array of the valsalva mask.
    moving: nd.array
        Array of the rest Mask

    Returns
    --------
    outward_field: nd.array of shape [Y|X]
    inward_field: nd.array of shape [Y|X]
    '''

    # initialize the symmetric diffeomoprphic registration
    metric = SSDMetric(dim = 2, smooth = 6, inner_iter = 10 )
    #metric = CCMetric(dim = 2,  sigma_diff = 2, radius= 4 )

    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=[64, 32, 16])

    mapping = sdr.optimize(static,moving)
    if mapping.is_inverse:
        inward_field = mapping.get_forward_field()
        outward_field = mapping.get_backward_field()
    else:
        inward_field = mapping.get_backward_field()
        outward_field = mapping.get_forward_field()

    return outward_field, inward_field


def create_displacement_layer(displacement_field, mode, y_shape=512, x_shape=512, centroid=[512,255.5]):
    '''
    Create the outward displacement for the outward and inward arrays.

    Parameters
    ----------
    displacement_field: nd.array
        Array of shape [Y|X] containing the components of displacement
    mode: string
        One of 'outward' or 'inward'
    y_shape: int
        Shape of the data in Y direction
    x_shape: int
        Shape of the data in X direction
    centroid: nd.array(2)
        2-D coordinates of the centroid from the displaced mask

    Returns
    --------
    absolute_displacement: nd.array
        Array of the displacement values of shape (y_shape,x_shape)
    '''

    absolute_displacement = np.sqrt(displacement_field[:,:,1]**2 + displacement_field[:,:,0]**2)

    x = np.arange(x_shape) - centroid[1]
    y = np.arange(y_shape) - centroid[0]
    X, Y = np.meshgrid(x,y)
    Normalization = np.sqrt(X**2 + Y**2)
    Normalization[Normalization==0] = 1
    X = X/Normalization
    Y = Y/Normalization

    # compute the Component of the vector field pointing away from the centroid
    if mode == 'outward':
        absolute_displacement[(displacement_field[:,:,1]*X + displacement_field[:,:,0]*Y) < 0] = 0
    elif mode == 'inward':
        absolute_displacement[(displacement_field[:,:,1]*X + displacement_field[:,:,0]*Y) > 0] = 0

    return absolute_displacement


def create_displacement_array(path_dict):
    '''
    Create the 3-D displacement and strain arrays.
    Saves them to the given paths.

    Parameters
    ----------
    path_dict: dict of dict of string
        A patients path dictionary
    '''
    # load both masks
    rest = imread(path_dict['Rest']['mask'])
    valsalva = imread(path_dict['Valsalva']['mask'])
    # get shapes of the masks
    num_slices, y_shape, x_shape = rest.shape
    num_slices = min(num_slices, valsalva.shape[0])
    # initilazie the arrays
    outward_inward = np.zeros((num_slices,y_shape,x_shape,2),dtype=float)
    outward_inward_strain = np.zeros((num_slices,y_shape,x_shape,2),dtype=float)
    # initialize the step size as 1cm per evaluation
    step_size = int(10 // float(path_dict['Rest']['z_spacing']))

    # loop over one layer every cm
    for layer in tqdm(range(0,num_slices,step_size)):
        if np.any(valsalva[layer]) and np.any(rest[layer]):
            outward, inward = symmetric_registration(valsalva[layer],rest[layer])

            # Get the centroid of the mask
            rest_centroid = ndimage.center_of_mass(rest[layer])
            valsalva_centroid = ndimage.center_of_mass(valsalva[layer])

            # create the displacement and strain values for this layer
            outward_inward[layer,:,:,0] = create_displacement_layer(outward,'outward',y_shape,x_shape,rest_centroid)
            outward_inward[layer,:,:,1] = create_displacement_layer(inward,'inward',y_shape,x_shape,valsalva_centroid)
            outward_inward_strain[layer,:,:,0] = create_strain_layer(outward[:,:,1],outward[:,:,0])
            outward_inward_strain[layer,:,:,1] = create_strain_layer(inward[:,:,1],inward[:,:,0])

        if layer>=step_size:
            for step in range(1, step_size, 1):
                outward_inward[layer-step_size + step,...] = (1 - step/step_size)* outward_inward[layer-step_size,...] + (step/step_size)* outward_inward[layer,...]
                outward_inward_strain[layer-step_size + step,...] = (1 - step/step_size)* outward_inward_strain[layer-step_size,...] + (step/step_size)* outward_inward_strain[layer,...]

    if layer < num_slices-1 and np.any(valsalva[num_slices-1]) and np.any(rest[num_slices-1]):
        outward_last, inward_last = symmetric_registration(valsalva[num_slices-1],rest[num_slices-1])
        # get the centroid of the mask
        rest_centroid = ndimage.center_of_mass(rest[num_slices-1])
        valsalva_centroid = ndimage.center_of_mass(valsalva[num_slices-1])
        outward_inward[-1,:,:,0] = create_displacement_layer(outward_last,'outward',y_shape,x_shape,rest_centroid,)
        outward_inward[-1,:,:,1] = create_displacement_layer(inward_last,'inward',y_shape,x_shape,valsalva_centroid)
        outward_inward_strain[-1,:,:,0] = create_strain_layer(outward_last[:,:,1],outward_last[:,:,0]) 
        outward_inward_strain[-1,:,:,1] = create_strain_layer(inward_last[:,:,1],inward_last[:,:,0])

    for step in range(1,num_slices-layer,1):
        outward_inward[layer+step,...] = (1 - step/(num_slices-layer))*outward_inward[layer,...] + (step/(num_slices-layer))*outward_inward[-1,...]
        outward_inward_strain[layer+step,...] = (1 - step/(num_slices-layer))*outward_inward_strain[layer,...] + (step/(num_slices-layer))*outward_inward_strain[-1,...]

    # set negativ/imward displacement to 0
    outward_inward[outward_inward < 0] = 0
    # remove strain outliers
    strain_threshold = np.quantile(outward_inward_strain, 0.95)
    outward_inward_strain[outward_inward_strain > strain_threshold] = strain_threshold

    #outward_inward_strain[...,0] = ndimage.gaussian_filter(outward_inward_strain[...,0],6)
    #outward_inward_strain[...,1] = ndimage.gaussian_filter(outward_inward_strain[...,1],6)
    
    # save to a given location
    imwrite(path_dict['Rest']['displacement_array'],outward_inward[...,0],compression='zlib')
    imwrite(path_dict['Valsalva']['displacement_array'],outward_inward[...,1],compression='zlib')
    imwrite(path_dict['Rest']['strain_array'],outward_inward_strain[...,0],compression='zlib')
    imwrite(path_dict['Valsalva']['strain_array'],outward_inward_strain[...,1],compression='zlib')


def get_displacement_dims(path_dict,threshold):
    '''
    Computes the length and width of the largest connected region
    where the displacment is larger than threshold.

    Parameters
    ----------
    path_dict: dict of dict of string
        A patients path dictionary
    threshold: float
        Threshold value defining a displacment as to large

    Returns
    ----------
    length: float
        length of the largest displacment area 
    width: float
        width of the largest displacemnt area
    '''

    # get the largest CC 
    mask = imread(path_dict['mask'])
    displacement = imread(path_dict['displacement_array'])
    mask = np.copy(mask[:displacement.shape[0]])
    displacement = displacement >= threshold
    displacement[mask==0] = 0
    components = label(displacement)
    # assume at least 1 CC
    if (components.max() != 0): 
        largestCC = components == np.argmax(np.bincount(components.flat)[1:])+1
        projection = largestCC.sum(axis=1)
        length = round(np.amax(np.count_nonzero(projection, axis=0))*float(path_dict['z_spacing'])*0.1)
        width  = round(np.amax(np.count_nonzero(projection, axis=1))*float(path_dict['x_spacing'])*0.1)
    else:
        length = 0
        width  = 0
        
    return length, width


def annotate_displacement_image(observation,observation_dict,area,individual_threshold):
    '''
    Annotate the displacement image.
    Adding name and dimensions of relevant area.

    Parameter
    ---------
    observation: string
        One of 'Rest' or 'Valsalva'
    observation_dict: dict of string
        A patients path dictionary for either the Rest or Valsalva data
    area: float
        Area of the displacment larger than threshold
    individual_threshold: int
        The patient specific threshold for the displacement. 
    '''
    img = Image.open(observation_dict['displacement_png'])
    length, width = get_displacement_dims(observation_dict,individual_threshold)
    # annotate the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text= (f'{observation}\n'
                   f'Unstable Abdominal Wall (Displacement > {individual_threshold}mm)\n'
                   f'Width: {width}cm  |  Length: {length}cm  |  Area: {round(area)}cm²'
                   ),
            fill=(0,0,0),
            anchor='ma',
            align = 'center',
            font=font,
            )
    # save the image
    img.save(observation_dict['displacement_png'],format='png')


def annotate_strain_image(observation,observation_dict):
    '''
    Annotate the strain image.
    Adding name and dimensions of relevant area.

    Parameter
    ---------
    observation: string
        One of 'Rest' or 'Valsalva'
    obseravtion_dict: dict of string
        A patients dictionary containing the paths to either the rest or valsalva data
    '''
    # loaf the strain image
    img = Image.open(observation_dict['strain_png'])
    # annotate the imge
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text= (f'{observation}\n'
                   f'Strain\n'),
            fill=(0,0,0),
            anchor='ma',
            align = 'center',
            font=font,
            )
    # save the image
    img.save(observation_dict['strain_png'],format='png')


def plot_displacement(path_dict,path_to_save):
    font = {'family': 'serif', 'color':  'red'}
    # plot areas of displacement with respect to their magnitute
    plt.plot(np.arange(10,len(path_dict['Rest']['displacement_areas'])),
            path_dict['Rest']['displacement_areas'][10:],
            color='blue', label='Rest')
    plt.plot([15],path_dict['Rest']['displacement_areas'][15],
            marker='v', color='g',
            label=f'{np.round(path_dict["Rest"]["displacement_areas"][15])}cm²')
    plt.plot(np.arange(10,len(path_dict['Valsalva']['displacement_areas'])),
            path_dict['Valsalva']['displacement_areas'][10:],
            color='red',label='Valsalva')
    plt.plot([15],path_dict['Valsalva']['displacement_areas'][15],
            marker='^', color='orange',
            label=f'{np.round(path_dict["Valsalva"]["displacement_areas"][15])}cm²')
    plt.title('Work in Progress. NOT for Clinical Use!',fontdict=font)
    plt.ylabel('Defectarea in cm²')
    plt.xlabel('Magnitude of displacement in mm')
    plt.legend(title = 'Modus')
    plt.savefig(f'{path_to_save}/DefectareaDiagramm.png')
    plt.close()


def plot_displacement_lower(path_dict,path_to_save):
    font = {'family': 'serif', 'color':  'red'}

    total_area_rest = path_dict['Rest']['displacement_areas'][0]
    total_area_val = path_dict['Valsalva']['displacement_areas'][0]

    # plot areas of displacement with respect to their magnitute
    plt.plot(np.arange(0, len(path_dict['Rest']['displacement_areas'])),
            total_area_rest - path_dict['Rest']['displacement_areas'],
            color='blue', label='Rest')
    plt.plot([15],(total_area_rest - path_dict['Rest']['displacement_areas'][15]),
            marker='v', color='g',
            label=f'{np.round(total_area_rest - path_dict["Rest"]["displacement_areas"][15])}cm²')
    plt.plot(np.arange(0,len(path_dict['Valsalva']['displacement_areas'])),
            total_area_val - path_dict['Valsalva']['displacement_areas'],
            color='red',label='Valsalva')
    plt.plot([15],(total_area_val - path_dict['Valsalva']['displacement_areas'][15]),
            marker='^', color='orange',
            label=f'{np.round(total_area_val - path_dict["Valsalva"]["displacement_areas"][15])}cm²')
    plt.title('Work in Progress. NOT for Clinical Use!',fontdict=font)
    plt.ylabel('Stable Area in cm²')
    plt.xlabel('Magnitude of displacement in mm')
    plt.legend(title = 'Modus')
    plt.savefig(f'{path_to_save}/Area_lower_than_treshhold.png')
    plt.close()


def plot_displacement_difference(path_dict,path_to_save):
    font = {'family': 'serif', 'color':  'red'}

    total_area_rest = path_dict['Rest']['displacement_areas'][0]
    total_area_valsalva = path_dict['Valsalva']['displacement_areas'][0]
    area_difference = total_area_valsalva - total_area_rest


    # plot areas of displacement with respect to their magnitute
    plt.plot(np.arange(0, len(path_dict['Rest']['displacement_areas'])),
            area_difference + path_dict['Rest']['displacement_areas'] - path_dict['Valsalva']['displacement_areas'],
            color='blue')

    plt.title('Work in Progress. NOT for Clinical Use!',fontdict=font)
    plt.ylabel('Area difference lower than treshhold in cm²')
    plt.xlabel('Magnitude of displacement in mm')
    # plt.legend(title = 'Modus')
    plt.savefig(f'{path_to_save}/DefectAreaDifference.png')
    plt.close()


def plot_individual_threshold(path_dict,path_to_save,threshold):

    area_rest_unstable = np.copy(path_dict['Rest']['displacement_areas'])
    area_valsalva_unstable = np.copy(path_dict['Valsalva']['displacement_areas'])
    area_rest_stable = np.copy(area_rest_unstable[0] - area_rest_unstable)
    area_valsalva_stable = np.copy(area_valsalva_unstable[0] - area_valsalva_unstable)


    min_area = 20

    relative_threshold = np.ones_like(area_rest_unstable)
    for k in range(len(relative_threshold)):
        if area_rest_stable[k]>0 and area_valsalva_unstable[k]>0 and area_rest_unstable[k]>min_area:
            relative_threshold[k] = (1 - (area_valsalva_stable[k]) / (area_rest_stable[k]))**2 + ((area_rest_unstable[k]) / (area_valsalva_unstable[k]))**2

    # x-achse
    plt.plot(np.arange(0, len(relative_threshold)),
            relative_threshold,
            color='blue')

    # threshold
    if not threshold:
        threshold = max(np.argmin(relative_threshold), 10)

    # marker
    plt.plot([threshold],[relative_threshold[threshold]],
                marker='^', color='orange',
                label=f'{threshold}mm')

    plt.ylabel('Instability ratio')
    plt.xlabel('Magnitude of translation in mm')
    plt.legend()
    plt.savefig(f'{path_to_save}/RelativeThreshold.png')
    plt.close()

    return threshold


###############################################################################
# Helper functions for the CT-Crosssection
###############################################################################

def create_crosssection(observation_dict):
    '''
    Create a png of the CT slice with the largest displacement.

    Parameters
    ----------
    obseravtion_dict: dict of string
        A patients dictionary containing the paths to either the rest or valsalva data

    Returns
    --------
    layer: int
        Slice id of the slice with max. displacement

    '''
    # load the labels
    label_array = imread(observation_dict['labels'])
    # load the displacement array
    displacement_array = imread(observation_dict['displacement_array'])
    # get the layer of the largest hernia sac or displacment
    if np.any(label_array==7):
        layer = np.argmax(np.sum(label_array==7,axis=(1,2)))
    else:
        layer = np.argmax(np.amax(displacement_array,axis=(1,2)))
    # get the dcm file containg that layer (they start at 1)
    layer_path = f'{observation_dict["dcm_dir"]}/{str(layer+1).zfill(6)}.dcm'
    # convert the dcm file into an 8bit rgb array and adjust the size to fit with other data
    ds = pydicom.filereader.dcmread(layer_path)
    img = ds.pixel_array
    img = np.pad(img, ((44,44),(5,5)), mode='constant',constant_values=0)
    # get corresponding slice of label data
    label_array = label_array[layer]
    label_array = np.pad(label_array, ((44,44),(5,5)), mode='constant',constant_values=0)
    resolution = 600
    if resolution != 600:
        import cv2
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        label_a = np.zeros_like(img)
        for k in np.unique(label_array):
            tmp = np.zeros_like(label_array)
            tmp[label_array==k]=1
            tmp = cv2.resize(tmp, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
            label_a[tmp==1]=k
        label_array = np.copy(label_a)
    label_color = ['black','darkblue','royalblue','lightsteelblue','beige','sandybrown',None,'red']
    # add the contours of every label over the img
    fig = plt.figure(frameon=False, figsize=(.87, 1))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto', cmap = plt.cm.gray)
    for label_value in np.unique(label_array):
        label = label_array == label_value
        contours = find_contours(label)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color= label_color[label_value], linewidth = .2,)

    plt.savefig(observation_dict['crosssection'],dpi=resolution)
    plt.close()
    return layer


def annotate_crosssection(observation_dict,layer):
    '''
    Annotate the ct_crosssection image.
    Adding name and dimensions of relevant area.

    Parameter
    ---------
    obseravtion_dict: dict of string
        A patients dictionary containing the paths to either the rest or valsalva data
    layer: int 
        Slice id of the slice with max. displacement
    '''
    img = Image.open(observation_dict['crosssection'])
    # annotate the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text=f'Hernia sac largest at Y = {(int(observation_dict["z_sh"]) - layer)*float(observation_dict["z_spacing"])}mm \n'
                 f'(slice {int(observation_dict["z_sh"]) - layer})',
            fill=(255,255,255),
            anchor='ma',
            align='center',
            font=font,
            )
    # save the crosssection
    img.save(observation_dict['crosssection'],format='png')


###############################################################################
# Anotation of the label image
###############################################################################

def get_label_sizes(path_to_tif,x_res,y_res,z_res):
    '''
    Computes the length, width, area and volume of the labeled hernia.
    as well as the volume of the abdominal cavity.

    Parameters
    ----------
        path_to_tif: string
            Pathstring to the .tiff file containing the labels.
        x_res: float
            The real length represented by one pixel in X direction.
        y_res: float
            The real length represented by one pixel in Y direction.
        z_res: float
            The real length represented by one pixel in Z direction.
    
    Returns
    ----------
        hernia_width: float
            Width of the Hernia Sac
        hernia_length: float
            Length of the Hernia Sac
        hernia_area: float
            Area of the Hernia Sac
        hernia_volume: float
            Volume of the Hernia Sac
        abdomen_volume: float
            Volume of the abdominal cavity
    '''
    # load segmentation
    label_array = imread(path_to_tif)
    zsh, _, _ = label_array.shape

    hernia_only = label_array==7

    # compute Volume of herniasack
    hernia_volume = round(np.sum(hernia_only)*x_res*y_res*z_res*0.001)

    # compute Volume of Abdomen 
    abdomen_volume= round(np.sum(label_array==5)*x_res*y_res*z_res*0.001)

    # compute largest CC
    components = label(hernia_only)
    if (components.max() != 0): # assume at least 1 CC
        largestCC = components == np.argmax(np.bincount(components.flat)[1:])+1

        # compute area of herniasack
        hernia_area   = round(np.sum(np.amax(largestCC,axis=1))*x_res*z_res*0.01)

        # compute width and length of herniasack
        hernia_width  = round(np.sum(np.amax(largestCC,axis=(0,1)))*x_res*0.1)

        # compute width and length of herniasack
        hernia_length = round(np.sum(np.amax(largestCC,axis=(1,2)))*z_res*0.1)

    else:
        hernia_area   = 0
        hernia_width  = 0
        hernia_length = 0


    return hernia_width, hernia_length, hernia_area, hernia_volume, abdomen_volume


def annotate_label_image(observation,observation_dict):
    '''
    Annotate the image of the labels.
    Adding names and dimensions of relevant areas and volumes.

    Parameters
    ----------
    observation: string
        One of 'Rest' or 'Valsalva'
    obseravtion_dict: dict of string
        A patients dictionary containing the paths to either the rest or valsalva data
    '''
    hernia_width, hernia_length, hernia_area, hernia_volume, abdominal_volume = get_label_sizes(
            observation_dict['labels'],float(observation_dict['x_spacing']),
            float(observation_dict['y_spacing']),float(observation_dict['z_spacing']))

    # write hernia dimensions on the image
    to_annotate = Image.open(observation_dict['labels_png'])
    draw = ImageDraw.Draw(to_annotate)
    font = ImageFont.truetype("arial.ttf", size=18)
    draw.text(xy=(to_annotate.width/2,0),
            text= (f'{observation}\n'
                   f'Hernia Width: {hernia_width}cm  |  Hernia Length: {hernia_length}cm  |  Hernia Area {hernia_area}cm²\n'
                   f'Hernia Volume: {hernia_volume}cm³  |  Abd. Cavity Volume: {abdominal_volume}cm³\n'
                   f'Loss of Domain: {round(hernia_volume/(hernia_volume+abdominal_volume),2)} (Sabbagh)  |  {round(hernia_volume/abdominal_volume,2)} (Tanaka)'
                   ),     
            fill=(0,0,0),
            anchor='ma',
            align = 'center',
            font = font,
            )
    to_annotate.save(observation_dict['labels_png'],format='png')

