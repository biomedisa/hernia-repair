import numpy as np
import pydicom, os, glob, sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageFont
from tifffile import imread

def fill_length_dir(path_to_data,save_path):
    #get the dcm files
    slices = glob.glob(path_to_data+'/**/*', recursive=True)
    #set start values
    ds = pydicom.filereader.dcmread(slices[0])
    img = ds.pixel_array
    #initalize array for the data
    data = np.zeros((len(slices), ds.Rows, ds.Columns), dtype=img.dtype)
    #loop over all files
    for file in slices:
        #get slice information
        ds = pydicom.filereader.dcmread(file)
        img = ds.pixel_array
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:,:,0]
        #set corresponding slice in the array
        data[ds.InstanceNumber-1] = img
    #convert to uint8
    data = np.uint8(data*255./np.amax(data))
    #creat subfolders
    if not os.path.exists(save_path+'\\Länge'):
        os.mkdir(save_path+'\\Länge')
        os.mkdir(save_path+'\\Länge\\Daten')
    if not os.path.exists(save_path+'\\Breite'):
        os.mkdir(save_path+'\\Breite')
        os.mkdir(save_path+'\\Breite\\Daten')
    #save every layer in z and x direction
        for z in range(data.shape[0]):
            ## save data as tif
            img = Image.fromarray(data[z,:,:], mode='L')
            img.resize((512,512))
            img.save(save_path +'\\Länge\\Daten\\'+ str(z).zfill(6) +'.tif')

        for x in range(data.shape[2]):
            ## save data as tif
            img = Image.fromarray(data[:,:,x], mode='L')
            img.resize((512,512))
            img.save(save_path +'\\Breite\\Daten\\'+ str(x).zfill(6) +'.tif')

def get_hernia_length(path_to_model,path_to_data):

    model = tf.keras.models.load_model(path_to_model)

    datagen = ImageDataGenerator( 
    )

    hernia_data = datagen.flow_from_directory(
        path_to_data,
        class_mode= None,
        color_mode='rgb',
        batch_size= 32,
        shuffle= False,
        target_size=(512,512),
    )

    prediction = model.predict(hernia_data, verbose=0)

    hernia_interval = np.argwhere(prediction>=0.5)
    try:
        hernia_length = hernia_interval[-1][0] - hernia_interval[0][0]
    except:
        hernia_length = 0

    return hernia_length

def get_hernia_area(height,width):
    #compute the area of an elipse with hernia height and width
    area = np.pi*height*width
    return area

def annotate_by_label(path_to_tif,path_to_dcm):
    # load segmentation
    a = imread(path_to_tif)
    zsh, _, _ = a.shape

    # load tomographic data & get pixel spacing
    file = path_to_dcm + '\\' + str(1).zfill(6) + '.dcm'

    # get slice thickness
    ds = pydicom.filereader.dcmread(file)
    z_res = float(ds.SliceThickness)
    y_res, x_res = ds.PixelSpacing

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

def annotate_by_neural_net(path_to_dcm,path_to_length_dir):
    #fill the length directory with images
    fill_length_dir(path_to_dcm,path_to_length_dir)
    #Create Paths to both subdirectories
    height_dir = path_to_length_dir + '\\Länge'
    width_dir  = path_to_length_dir + '\\Breite'
    #Get slice width 
    file = path_to_dcm + '\\' + str(1).zfill(6) + '.dcm'
    ds = pydicom.filereader.dcmread(file)
    slice_thickness = ds.SliceThickness
    _ , slice_width = ds.PixelSpacing
    #Get Height,width and area of the hernia
    hernia_width = get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_x.h5",width_dir)
    hernia_width *= slice_width*0.1
    hernia_height = get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_z.h5",height_dir)
    hernia_height *= slice_thickness*0.1
    hernia_area = get_hernia_area(hernia_height,hernia_width)
    return hernia_width, hernia_height ,hernia_area

def annotate_image(observation,path_to_dcm,path_to_length_dir,path_to_tif,path_to_png):
    hernia_width_by_nn, hernia_height_by_nn,hernia_area_by_nn = annotate_by_neural_net(path_to_dcm,path_to_length_dir)
    instabel_area_by_label, hernia_area_by_label = annotate_by_label(path_to_tif,path_to_dcm)
    
    #write hernia dimensions on the image
    to_annotate = Image.open(path_to_png)
    draw = ImageDraw.Draw(to_annotate)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(to_annotate.width/2,0),
            text= observation + 
            '\nBerechnete Größen:\n' + 'Breite:' + str(round(hernia_width_by_nn,1)) + 'cm     Länge:'+ str(round(hernia_height_by_nn,1)) + 'cm    Bruchpforten Fläche:' + str(round(hernia_area_by_nn,1)) + 'cm²' +
            '\nGrößen im Bild:\n' + 'Instabile_Fläche:'+ str(round(instabel_area_by_label*0.01,1)) + 'cm²    Projezierte Fläche:' + str(round(hernia_area_by_label*0.01,1)) + 'cm²',     
            fill=(0,0,0),
            anchor="ma",
            font=font,
            )
    to_annotate.save(path_to_png,format='png')


observation = sys.argv[1]
path_to_dcm = sys.argv[2]
path_to_length_dir = sys.argv[3]
path_to_tif = sys.argv[4]
path_to_png = sys.argv[5]
annotate_image(observation,
                path_to_dcm,
                path_to_length_dir,
                path_to_tif,
                path_to_png
                )