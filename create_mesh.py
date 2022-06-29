from tifffile import imread
import numpy as np
import sys
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def MarchingCubes(image,scale,scale_min,scale_max):

    # marching cubes
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.GenerateValues(scale,scale_min,scale_max)
    mc.Update()

    # To remain largest region
    #confilter = vtk.vtkPolyDataConnectivityFilter()
    #confilter.SetInputData(mc.GetOutput())
    #confilter.SetExtractionModeToLargestRegion()
    #confilter.Update()

    # reduce poly data
    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(mc.GetOutput())


    # smooth surface
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(inputPoly)
    smoothFilter.SetNumberOfIterations(50)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    finalPoly = vtk.vtkPolyData()
    finalPoly.ShallowCopy(smoothFilter.GetOutput())
    
    return finalPoly#confilter.GetOutput()


def CreateVTK(path_to_data,path_to_save,x_thickness,y_thickness,slice_thickness,mode='labels'):
    
    image = imread(path_to_data)
    
    if mode == 'translation':
        scale, scale_min, scale_max = int(np.amax(image)),1,int(np.amax(image))
    elif mode == 'labels':
        image[0,0,0] = 7
        scale, scale_min, scale_max = 7,1,7
        
    else: raise ValueError('mode must be one of "labels", "translation".')

    #flip image
    image = np.flip(image, axis=(0))

    # get image dims
    zsh,ysh,xsh=image.shape

    # numpy to vtk
    sc = numpy_to_vtk(num_array=image.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    imageData = vtk.vtkImageData()
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(x_thickness, y_thickness, slice_thickness)
    #imageData.SetDimensions(zsh, ysh, xsh)
    imageData.SetExtent(0,xsh-1,0,ysh-1,0,zsh-1)
    imageData.GetPointData().SetScalars(sc)

    # get poly data
    poly = MarchingCubes(imageData,scale,scale_min,scale_max)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(path_to_save)
    writer.Write()

