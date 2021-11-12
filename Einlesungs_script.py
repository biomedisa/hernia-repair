#!/usr/bin/env python
# coding: utf-8
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pydicom
import glob
import time,shutil,ssl
import requests
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from math import pi
from tkinter.filedialog import askdirectory
from tifffile import imread
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
    files = glob.glob(path_to_dir+'/**/*', recursive=True)
    
    #Set patients directory 
    for file in files:
        if os.path.isfile(file):
            ds = pydicom.filereader.dcmread(file)
            if not 'first_level' in locals():
                #name the directory containing all results after patient name + Birthdate     	      
                first_level = main_path +'\\'+str(ds.PatientName)+'_'+str(ds.PatientBirthDate)
                first_level = first_level.replace('^','_')  
                first_level = first_level.replace(' ','_') 
                first_level = first_level.replace('ü','ue')
                first_level = first_level.replace('ä','ae')
                first_level = first_level.replace('ö','oe') 
            second_level = first_level +'\\' +str(ds.StudyDate)+'_'+str(ds.StudyDescription)
            third_level = second_level + '\\' + str(ds.SeriesNumber) + '_' + str(ds.SeriesDescription)

            if not os.path.exists(first_level):                       
                os.mkdir(first_level)
            if not os.path.exists(second_level):                       
                os.mkdir(second_level)
            if not os.path.exists(third_level):
                os.mkdir(third_level)
            
            path_to_dest = third_level + '\\' + str(ds.InstanceNumber).zfill(6) + '.dcm'
            if not os.path.exists(path_to_dest):			
                shutil.copy(file, path_to_dest)
    return first_level

def get_slice_thickness():
    nativ_slice_thickness = pydicom.filereader.dcmread(
                                    observation_path['Nativ']['dcm_dir'] + '\\000001.dcm'
                                    ).SliceThickness
    valsalva_slice_thickness = pydicom.filereader.dcmread(
                                    observation_path['Valsalva']['dcm_dir'] + '\\000001.dcm'
                                    ).SliceThickness
    if nativ_slice_thickness == valsalva_slice_thickness: 
        return nativ_slice_thickness

    else: print('Nativ and Valsalva slice thickness do not match! \n',
                'Please load Data with matching thickness.')

def Creat_CT_crosssection(observation,path_to_tif,path_to_dcm):
    #get tif with labels
    label = imread(path_to_tif)
    #set all labels that aren´t a hernia to 0
    label[label!=7] = 0
    #sum the amount of pixels containing the hernia in each horizontal slice
    label_sum = np.sum(label, axis=(1,2))
    #get the layer that contains the largest amount of hernia pixels
    layer = np.argmax(label_sum) + 1
    #get the dcm file containg that layer
    layer_path = path_to_dcm + '\\' + str(layer).zfill(6) + '.dcm'
    #convert the dcm file into an PIL image
    path_to_png = path_to_tif.replace(os.path.basename(path_to_tif), observation + '_crosssection.png')
    ds = pydicom.filereader.dcmread(layer_path)
    img = ds.pixel_array
    plt.imsave(path_to_png,img,cmap='gray')
    img = Image.open(path_to_png)
    #annotate the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf",size=20)
    draw.text(xy=(img.width/2,0),
            text= observation +' \n' + 'Layer: ' + str(layer),
            fill=(255,255,255),
            anchor='ma',
            font=font,
            )
    #save the crosssection        

    img.save(path_to_png,format='png')
    
    
    return path_to_png

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

def get_hernia_area(height,width):
    #compute the area of an elipse with hernia height and width
    area = pi*height*width
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
    sys.path.insert(1,r'C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair')
    import Prediction
    #fill the length directory with images
    fill_length_dir(path_to_dcm,path_to_length_dir)
    #Create Paths to both subdirectories
    height_dir = path_to_length_dir + '\\Länge'
    width_dir  = path_to_length_dir + '\\Breite'
    #Get slice width 
    file = path_to_dcm + '\\' + str(1).zfill(6) + '.dcm'
    ds = pydicom.filereader.dcmread(file)
    _ , slice_width = ds.PixelSpacing
    #Get Height,width and area of the hernia
    hernia_width = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_x.h5",width_dir)
    hernia_width *= slice_width*0.1
    hernia_height = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_z.h5",height_dir)
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

if __name__ == "__main__":
    #Check for updates and update the neural nets
    try:
        update_neural_nets()
    except:
        print('Couldn´t update neuralnets! Check your internet connection.\n','Start with old ones.')
    #Get time to measure execution time
    start_time = datetime.now()
    #Set the main save directory    
    main_path = "D:\\Hernien_Analyse"
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    #Set the patients directorys and fill with data
    first_level = load_directorys()

    #Set the paths for both observations
    Observations = ["Nativ","Valsalva"] 
    observation_exists = {observation:False for observation in Observations}
    observation_path = {observation:{"tif":"","vtk":"","png":"","crosssection":"","dcm_dir":"","length_dir":""}for observation in Observations}
    
    for observation in sorted(Observations):            
        #Create Paths to the mesh and the img
        observation_path[observation]['tif'] = first_level + '\\final.' + observation + '.tif'
        observation_path[observation]['vtk'] = first_level + '\\' + observation + '_for_paraview.vtk'
        observation_path[observation]['png'] = first_level + '\\' + observation + '_front_view.png'
        observation_path[observation]['dcm_dir'] = askdirectory(initialdir = first_level)
        observation_exists[observation] = True
        observation_path[observation]['length_dir'] = first_level+'\\' + observation + '_length'          
        if not os.path.exists(observation_path[observation]['length_dir']):
            os.mkdir(observation_path[observation]['length_dir']) 

    slice_thickness = get_slice_thickness()
    
    #Run all subprocesses for both observations
    for observation in Observations:
        if observation_exists[observation]:
            
            #console output
            os.system('cls')
            print('Processing ' + observation + ':\n Computing Labels...')


            #Create the classification proposal in form of a tif
            net = call(["python",r"C:\Users\Hernienforschung\git\biomedisa\demo\biomedisa_deeplearning.py", 
                        observation_path[observation]['dcm_dir'], r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\img_hernie.h5", "-p","-bs","6"]
                        )
            shutil.move(observation_path[observation]['dcm_dir'].replace(
                            os.path.basename(observation_path[observation]['dcm_dir']),
                            'final.' + '.tif'),
                            observation_path[observation]['tif'])
 
            #Create nativ mesh in vtk format for Paraview
            mesh = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\create_mesh.py", 
                        observation_path[observation]['tif'], observation_path[observation]['vtk'], str(slice_thickness)]
                        )


            #console Output
            print('Done\n Creating Images...')


            #Create image using Paraview
            screenshot = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\paraview_screenshot.py",observation_path[observation]['vtk'],observation_path[observation]['png']])
            
            #Create CT crosssection
            observation_path[observation]['crosssection'] = Creat_CT_crosssection(observation,observation_path[observation]['tif'],observation_path[observation]['dcm_dir'])

    #Execute Samuels script automaticaly and combine results
    if  all(observation_exists.values()): #Check if Data is complete

        #get time and date
        day = datetime.now()
        #Set Time String for saving the data
        day_string = day.strftime("%Y-%m-%d_%H-%M")
        #Execute Samuels Script
        sam = call([r"C:\Users\Hernienforschung\Documents\Auswertungen\Hernienauswertung_v0_11.exe", observation_path['Nativ']['dcm_dir'], observation_path['Valsalva']['dcm_dir']])
        #Set the saving paths for the optained data
        temp_paths = sorted(os.listdir('C:\\Users\\Hernienforschung\\Documents\\Python_Scripts\\Temp')) 
        temp_path_to_archiv = temp_paths[0]
        temp_path_to_evaluation = temp_paths[1]

        
        #console Output
        os.system('cls')
        print(' Annotate images...')
        

        #Preprocess and Annotate the Paraview labeled images
        #Read both images
        nat_img = plt.imread(observation_path['Nativ']['png'])
        val_img = plt.imread(observation_path['Valsalva']['png'])  
        
        #Reshape to match size of sam_img and to fit annotation
        nat_img = np.pad(nat_img, ((50,0),(0,1),(0,0)), mode='edge')
        val_img = np.pad(val_img, ((50,0),(0,0),(0,0)), mode='edge')
        plt.imsave(observation_path['Nativ']['png'],nat_img)
        plt.imsave(observation_path['Valsalva']['png'],val_img)        
    
        for observation in Observations:
            #Annotate the images
            annotate_image(observation,observation_path[observation]['dcm_dir'],
                            observation_path[observation]['length_dir'],
                            observation_path[observation]['tif'],
                            observation_path[observation]['png'])
        
        #Load all images
        nat_img = plt.imread(observation_path['Nativ']['png'])[:,:,:3]
        val_img = plt.imread(observation_path['Valsalva']['png'])[:,:,:3]
        
        #Check if directory was created a minute later and rename if neccesary
        sam_img = plt.imread(temp_path_to_evaluation + '\\Verschiebung und Verzerrung.png')

        nativ_crosssection = plt.imread(observation_path['Nativ']['crosssection'])[:,:,:3]
        valsalva_crosssection = plt.imread(observation_path['Valsalva']['crosssection'])[:,:,:3]
        
        #Resize images to same width for stacking 
        nativ_crosssection = np.pad(nativ_crosssection,((0,0),(39,40),(0,0)), mode='edge')
        valsalva_crosssection = np.pad(valsalva_crosssection,((0,0),(39,39),(0,0)), mode='edge')
        
        #Stack the image pairs
        nat_and_val = np.hstack((nat_img,val_img))
        double_cross = np.hstack((nativ_crosssection,valsalva_crosssection))
        #Stack result, paraview images and crosssections
        sam_and_labels = np.vstack((sam_img,nat_and_val))
        combined_img = np.vstack((sam_and_labels,double_cross))
        

        #console Output
        print('\nDone')


        #Move everything into one directory
        path_to_evaluation = first_level + '\\Auswertung_' + day_string
        shutil.move(temp_path_to_evaluation, path_to_evaluation)
        plt.imsave(path_to_evaluation + '\\Finale_Auswertung.png',combined_img)
        #Move used Data into the archiv folder
        archiv = first_level + '\\Archiv_zur_Fehlerdiagnose_' + day_string
        for file in (temp_path_to_archiv,observation_path['Nativ']['png'],observation_path['Valsalva']['png'],
                    observation_path['Nativ']['crosssection'],observation_path['Valsalva']['crosssection'],
                    observation_path['Nativ']['tif'],observation_path['Valsalva']['tif'],
                    observation_path['Nativ']['vtk'],observation_path['Valsalva']['vtk'],
                    ):
            shutil.move(file, archiv)
        #show final result
        os.system('start ' + path_to_evaluation + '\\Finale_Auswertung.png')

    #if either scan is missing do nothing
    else:
        print('Error: Nativ or Valsalva Data missing')

    end_time = datetime.now()
    print('Execution time: {}'.format(end_time - start_time))