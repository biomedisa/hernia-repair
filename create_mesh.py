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


def CreateVTK(image,path_to_save,x_thickness,y_thickness,slice_thickness,mode='labels'):
    
    if mode == 'distortion':
        if np.amax(image) > 15:
            image[image>50] = 50
            image = np.rint(image)
            image.astype(int)
            scale, scale_min, scale_max = np.amax(image),1,np.amax(image)
            
        elif np.amax(image) > 5:
            image[0,0,0] = 15
            image = np.rint(2*image)/2
            scale, scale_min, scale_max = 30,0.5,15
            
        else:
            image[0,0,0] = 5
            image = np.rint(10*image)/10
            scale, scale_min, scale_max = 50,0.1,5
    
    elif mode == 'labels':
        image[0,0,0] = 7
        scale, scale_min, scale_max = 7,1,7
        
    else: raise ValueError('mode must be one of "labels", "distortion".')

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

if __name__ == "__main__":

    # path to data
    path_to_data = sys.argv[1]
    path_to_save = sys.argv[2]

    # load data
    data = imread(path_to_data)

    # create vtk file
    CreateVTK(data, path_to_save,float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),sys.argv[6])

