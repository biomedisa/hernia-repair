import numpy as np
import pydicom, glob, sys, os
from tensorflow.python.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from tifffile import imread

def load_data(path_to_data):
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
    data = np.uint8(data*255./np.amax(data))
    data = np.stack((data,)*3,axis=-1)

    #for width detection
    '''
    for x in range(data.shape[2]):
        ## save data as tif
        img = Image.fromarray(data[:,:,x], mode='L')
        img.resize((512,512))
        img.save(save_path +'\\Breite\\Daten\\'+ str(x).zfill(6) +'.tif')
    '''
    return data,header

def get_hernia_length(data,mode):
    if mode == "length":
        model = load_model(f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_z.h5')
    elif mode == "width":
        model = load_model(f'{os.environ["userprofile"]}\\git\\Netzwerke\\hernien_detector_x.h5')
        
    prediction = model.predict(data, batch_size = 32,verbose = 0)

    hernia_interval = (prediction > 0.5).nonzero()[0]
    hernia_length = hernia_interval[-1] - hernia_interval[0]

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
    file = f'{path_to_dcm}/{str(1).zfill(6)}.dcm'

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

def resize_array(data):
    resized_data = np.zeros((512,512,512,3),dtype=data.dtype)
    for i in range(data.shape[0]):
        img = Image.fromarray(data[i],mode = "RGB")
        img = img.resize((512,512))
        slice = np.array(img)
        resized_data[i] = slice
    return(resized_data)

def annotate_by_neural_net(path_to_dcm):
    #Get slice width 
    data, header = load_data(path_to_dcm)
    slice_thickness = header[0].SliceThickness
    _ , slice_width = header[0].PixelSpacing
    #Get Height,width and area of the hernia
    hernia_height = get_hernia_length(data, mode = "length")
    hernia_height *= slice_thickness*0.1
    
    width_data = resize_array(np.swapaxes(data,0,2))
    
    hernia_width = get_hernia_length(width_data,mode = "width")
    hernia_width *= slice_width*0.1
    
    hernia_area = get_hernia_area(hernia_height,hernia_width)
    
    return hernia_width, hernia_height ,hernia_area

def annotate_image(observation,path_to_dcm,path_to_tif,path_to_png):
    hernia_width_by_nn, hernia_height_by_nn,hernia_area_by_nn = annotate_by_neural_net(path_to_dcm)
    instable_area_by_label, hernia_area_by_label = annotate_by_label(path_to_tif,path_to_dcm)
    
    #write hernia dimensions on the image
    to_annotate = Image.open(path_to_png)
    draw = ImageDraw.Draw(to_annotate)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text(xy=(0,0),
            text= (f'{observation}\n'
                   f'Detektiertes Bruchsackvolumen (rot)\n'
                   f'(Berechnete Größen) Breite: {str(round(hernia_width_by_nn,1))}cm   Länge: {str(round(hernia_height_by_nn,1))}cm    Bruchpforten Fläche: {str(round(hernia_area_by_nn,1))}cm²\n'
                   f'(Größen im Bild) Instabile_Fläche: {str(round(instable_area_by_label*0.01,1))}cm²   Projezierte Fläche: {str(round(hernia_area_by_label*0.01,1))}cm²'),     
            fill=(0,0,0),
            align = "center"
            font = font
            )
    to_annotate.save(path_to_png,format='png')


observation = sys.argv[1]
path_to_dcm = sys.argv[2]
path_to_tif = sys.argv[3]
path_to_png = sys.argv[4]
annotate_image(observation,
                path_to_dcm,
                path_to_tif,
                path_to_png
                )
