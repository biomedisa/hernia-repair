#!/usr/bin/env python
# coding: utf-8
import os
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
from PIL import Image, ImageDraw
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

def Creat_CT_crosssection(path_to_tif,path_to_dcm):
    #get tif with labels
    label = imread(path_to_tif)
    #set all labels that aren´t a hernia to 0
    label[label!=7] = 0
    #sum the amount of pixels containing the hernia in each horizontal slice
    label_sum = np.sum(np.sum(label, axis=2),axis=1)
    #get the layer that contains the largest amount of hernia pixels
    layer = np.argmax(label_sum) + 1
    #get the dcm file containg that layer
    layer_path = path_to_dcm + '\\' + str(layer).zfill(6) + '.dcm'
    #convert the dcm file into an PIL image
    path_to_png = path_to_dcm + '_crosssection.png'    
    ds = pydicom.filereader.dcmread(layer_path)
    img = ds.pixel_array
    plt.imsave(path_to_png,img,cmap='gray')
    img = Image.open(path_to_png)
    #annotate the image
    draw = ImageDraw.Draw(img)
    draw.text(xy=(200,0),
            text='Nativ \n Layer: ' + str(layer),
            fill=(255,255,255),
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

def annotate_nativimage():
    import Prediction
    #fill the length directory with images
    fill_length_dir(nativ_dir,nativ_length_dir)
    #Create Paths to both subdirectories
    nativ_height_dir = nativ_length_dir + '\\Länge'
    nativ_width_dir  = nativ_length_dir + '\\Breite'
    #Get Height,width and area of the hernia
    nativ_hernia_height = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_z.h5",nativ_height_dir)
    nativ_hernia_height *= slice_thickness*0.1 
    nativ_hernia_width = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_x.h5",nativ_width_dir)
    nativ_hernia_width *= slice_width*0.1
    nativ_hernia_area = get_hernia_area(nativ_hernia_height,nativ_hernia_width)
    #write hernia dimensions on the image
    img = Image.open(nativ_png)
    draw = ImageDraw.Draw(img)
    draw.text(xy=(200,0),
            text= 'Nativ \n' + 'Breite:' + str(round(nativ_hernia_width,1)) + 'cm Länge:'+ str(round(nativ_hernia_height,1)) + 'cm Fläche:' + str(round(nativ_hernia_area,1)) + 'cm²',
            fill=(0,0,0),
            )
    img.save(nativ_png,format='png')

def annotate_valsalvaimage():
    import Prediction
    #fill the length directory with images
    fill_length_dir(valsalva_dir,valsalva_length_dir)
    #Create Paths to both subdirectories
    valsalva_height_dir = valsalva_length_dir + '\\Länge'
    valsalva_width_dir  = valsalva_length_dir + '\\Breite'
    #Get Hernia height,width and area
    valsalva_hernia_height = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_z.h5",valsalva_height_dir)
    valsalva_hernia_height *= slice_thickness*0.1
    valsalva_hernia_width = Prediction.get_hernia_length(r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\Hernien_detector_x.h5",valsalva_width_dir)
    valsalva_hernia_width *= slice_width*0.1
    valsalva_hernia_area = get_hernia_area(valsalva_hernia_height,valsalva_hernia_width)
    #write dimensions on the image
    img = Image.open(valsalva_png)
    draw = ImageDraw.Draw(img)
    draw.text(xy=(200,0), 
            text= 'Valsalva \n' + 'Breite:'+ str(round(valsalva_hernia_width,1)) + 'cm Länge:' + str(round(valsalva_hernia_height,1)) + 'cm Fläche:' + str(round(valsalva_hernia_area,1)) +'cm²',
            fill= (0,0,0),
            )
    img.save(valsalva_png,format='png')

if __name__ == "__main__":
    #Check for and update the neural nets
    update_neural_nets()
    #Get the Path to the Data
    tk.Tk().withdraw()
    path_to_dir = askdirectory()

    #Get the raw dcm files
    files = glob.glob(path_to_dir+'/**/*', recursive=True)
    #Set the main save directory
    main_path = "C:\\Users\\Hernienforschung\\Desktop\\Hernien_Analyse" 
    nat_exists = False   #Existence booleans
    val_exists = False
    
    #create main directory if not yet existing
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    #Loop over all dcm files set directory names and search for Bvalue
    for file in files:      
        if os.path.isfile(file):
            #read dicom properties        
            ds = pydicom.filereader.dcmread(file)
            #name the directory containing all results after patient name + Birthdate     	      
            first_level = main_path +'\\'+str(ds.PatientName)+'_'+str(ds.PatientBirthDate)
            first_level = first_level.replace('^','_')  
            first_level = first_level.replace(' ','_') 
            first_level = first_level.replace('ü','ue')
            first_level = first_level.replace('ä','ae')
            first_level = first_level.replace('ö','oe')                        
            #Set the subdirectories
            nativ_dir = first_level+'\\nativ'
            nativ_length_dir = first_level+'\\nativ_length'
            valsalva_dir = first_level+'\\valsalva'
            valsalva_length_dir = first_level+'\\valsalva_length'
            #Create the directorys if not existing            
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            if not os.path.exists(nativ_dir):
                os.mkdir(nativ_dir)
            if not os.path.exists(nativ_length_dir):
                os.mkdir(nativ_length_dir)
            if not os.path.exists(valsalva_dir):					
                os.mkdir(valsalva_dir)
            if not os.path.exists(valsalva_length_dir):
                os.mkdir(valsalva_length_dir)             
            #Store String of the Series Name 
            series = str(ds.SeriesDescription)
            #Check for the Bvalue
            if 'B20s' in series:
                Bvalue = 'B20s'                    
                break
            elif 'B31s' in series:
                Bvalue = 'B31s'
    
    #fill the subdirectories         
    for file in files:
        if os.path.isfile(file):
            #read dicom properties  
            ds = pydicom.filereader.dcmread(file)
            #Store String of the Series Name 
            series = str(ds.SeriesDescription)
            #Extracte the horizontal nativ series coresponting to detectet Bvalue
            if ('o' in series or 'nativ' in series) and (Bvalue in series) and not ('SPO' in series or 'pressen' in series or 'm' in series):
                nat_exists = True
                #Set slice thickness if given
                try:
                    slice_thickness = ds.SliceThickness
                    _, slice_width = ds.PixelSpacing     
                except:
                    slice_thickness = '1'
                #Set the final path for the dcm files
                path_to_dest = nativ_dir + '\\' + str(ds.InstanceNumber).zfill(6) + '.dcm' 
                #Link the file if not yet linked
                if not os.path.exists(path_to_dest):			
                    shutil.copy(file, path_to_dest)
            
            #Extracte the horizontal valsalva series
            elif ('m' in series or 'pressen' in series) and (Bvalue in series) and not ('SPO' in series):  
                val_exists = True
                #Set the final path for the dcm files
                path_to_dest = valsalva_dir + '\\' + str(ds.InstanceNumber).zfill(6) + '.dcm'
                #Link the file if not yet linked
                if not os.path.exists(path_to_dest):
                    shutil.copy(file, path_to_dest)
            else:   
                continue


    #Create results for nativ data
    if nat_exists:
        
        #Create the classification proposal in for of a tif
        net1 = call(["python",r"C:\Users\Hernienforschung\git\biomedisa\demo\biomedisa_deeplearning.py", 
                    nativ_dir, r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\img_hernie.h5", "-p","-bs","6"]
                    )
        #Create Paths to the mesh and the img
        nativ_tif = first_level+'\\final.nativ.tif'
        nativ_vtk = nativ_tif.replace('.tif','.vtk')
        nativ_png = nativ_tif.replace('.tif','.png')
        
        #Create nativ mesh in vtk format for Paraview
        mesh1 = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\create_mesh.py", 
                    nativ_tif, str(slice_thickness)]
                    )
        
        #Create image using Paraview
        screen1 = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\paraview_screenshot.py",nativ_vtk])
        
        #Create CT crosssection
        nativ_cross_path = Creat_CT_crosssection(nativ_tif,nativ_dir)


    #Create results for valsalva data
    if val_exists:
        


        #Create the classification proposal in for of a tif
        net2 = call(["python",r"C:\Users\Hernienforschung\git\biomedisa\demo\biomedisa_deeplearning.py",
                    valsalva_dir, r"C:\Users\Hernienforschung\Documents\Python_Scripts\Netzwerke\img_hernie.h5", "-p","-bs","6"]
                    )
        
        #Create Paths to the mesh and the img                    
        valsalva_tif = first_level+'\\final.valsalva.tif'
        valsalva_vtk = valsalva_tif.replace('.tif','.vtk')
        valsalva_png = valsalva_tif.replace('.tif','.png')   

        #Create Valsalva mesh in vtk format for Paraview
        mesh2 = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\create_mesh.py", 
                    valsalva_tif, str(slice_thickness)]
                    )

        #Create img using Paraview
        screen2 = call(["python",r"C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\paraview_screenshot.py",valsalva_vtk])
       
        #Create CT crosssection images
        valsalva_cross_path = Creat_CT_crosssection(valsalva_tif,valsalva_dir)


    #Execute Samuels script automaticaly and combine results
    if nat_exists and val_exists: #Check if Data is complete
        
        #annotate bothimages
        annotate_nativimage()
        annotate_valsalvaimage()
        
        #get time and date
        day = datetime.now()
        #Set Time String for saving the data
        day_string = day.strftime("%Y-%m-%d_%H-%M")
        #Execute Samuels Script
        sam = call([r"C:\Users\Hernienforschung\Documents\Auswertungen\Hernienauswertung_v0_11.exe", nativ_dir, valsalva_dir])
        #Set the saving paths for the optained data
        sam_path = 'Auswertung_' + day_string
        sam_path_two = 'Archiv_zur_Fehlerdiagnose_' + day_string

        #Combine all images
        #Read all images
        nat_img = plt.imread(nativ_png)
        val_img = plt.imread(valsalva_png)
        sam_img = plt.imread(sam_path + '\\Verschiebung und Verzerrung.png')
        nativ_crosssection = plt.imread(nativ_cross_path)[:,:,:3]
        valsalva_crosssection = plt.imread(valsalva_cross_path)[:,:,:3]
        #Resize images to same width for stacking 
        nat_img = np.pad(nat_img, ((0,0),(0,1),(0,0)), mode='edge')
        nativ_crosssection = np.pad(nativ_crosssection,((0,0),(39,40),(0,0)), mode='edge')
        valsalva_crosssection = np.pad(valsalva_crosssection,((0,0),(39,39),(0,0)), mode='edge')
        #Stack the image pairs
        nat_and_val = np.hstack((nat_img,val_img))
        double_cross = np.hstack((nativ_crosssection,valsalva_crosssection))
        #Stack result, paraview images and crosssections
        sam_and_para = np.vstack((sam_img,nat_and_val))
        combined_img = np.vstack((sam_and_para,double_cross))
        

        #Move everything into one directory
        auswertung = first_level + '\\Auswertung_' + day_string
        shutil.move(sam_path, auswertung)
        plt.imsave(auswertung + '\\Finale_Auswertung.png',combined_img)
        #Move used Data into the archiv folder
        archiv = first_level + '\\Archiv_zur_Fehlerdiagnose_' + day_string
        for file in (sam_path_two,nativ_png,valsalva_png,nativ_cross_path,valsalva_cross_path,nativ_tif,valsalva_tif,nativ_vtk,valsalva_vtk):
            shutil.move(file, archiv)
        #show final result
        os.system('start ' + auswertung + '\\Finale_Auswertung.png')

    #if either scan is missing do nothing
    else:
        print('Error: Nativ or Valsalva Data missing')