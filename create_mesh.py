# imports
import numpy as np
import vtk
from tifffile import imread
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def MarchingCubes(image,threshold):
    '''
    Applies the marchingCubes algorithm to the data
    to optain a mesh of the data.
    
    Parameters
    ----------
    image: vtk image data
    threshold: The amount of contour levels
    '''

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.GenerateValues(threshold,1,threshold)
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
    smoothFilter.SetNumberOfIterations(30)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    finalPoly = vtk.vtkPolyData()
    finalPoly.ShallowCopy(smoothFilter.GetOutput())
    
    return finalPoly#confilter.GetOutput()


def CreateVTK(observation_dict=None,mode='labels',image=None,displacement_array=None,path_to_save=None,
    z_spacing=None, y_spacing=None, x_spacing=None):
    '''
    Creates a VTK file from a given tif multilayer image,
    useing the marchingcubes Algorithm.
    
    Parameters
    ----------
    obseravtion_dict: dict of string
        A patients dictionary containing the paths to either the rest or valsalva data
    mode: string
        one of labels, displacement or strain
    -------
    Creat a VTK mesh of either the muscel labels, the displacment of 
    the torso between Rest and Valsalva or the strain of the displacment.
    '''

    if mode == 'labels':
        if image is None:
            path_to_data = observation_dict['labels']
        if path_to_save is None:
            path_to_save = observation_dict['labels_vtk']
        threshold = 7

    elif mode == 'displacement':
        if image is None:
            path_to_data = observation_dict['mask']
        if displacement_array is None:
            displacement_array = imread(observation_dict['displacement_array'])
        if path_to_save is None:
            path_to_save = observation_dict['displacement_vtk']
        threshold = 1

    elif mode == 'strain':
        if image is None:
            path_to_data = observation_dict['mask']
        if displacement_array is None:
            displacement_array = imread(observation_dict['strain_array'])
        if path_to_save is None:
            path_to_save = observation_dict['strain_vtk']
        threshold = 1

    else:
        raise ValueError('mode must be one of "labels", "displacement" or "strain".')

    if x_spacing is None:
        x_spacing  = float(observation_dict['x_spacing'])
    if y_spacing is None:
        y_spacing  = float(observation_dict['y_spacing'])
    if z_spacing is None:
        z_spacing = float(observation_dict['z_spacing'])

    # load data
    if image is None:
        image = imread(path_to_data)

    # reduce image size to displacement array
    if mode in ['displacement','strain']:
        image = np.copy(image[:displacement_array.shape[0]])

    # get image dims
    zsh,ysh,xsh=image.shape

    # flip image
    image = np.flip(image, axis=(0))
    if mode in ['displacement','strain']:
        displacement_array = np.flip(displacement_array, axis=(0))
    # numpy to vtk
    sc = numpy_to_vtk(num_array=image.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    imageData = vtk.vtkImageData()
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(x_spacing, y_spacing, z_spacing)
    #imageData.SetDimensions(zsh, ysh, xsh)
    imageData.SetExtent(0,xsh-1,0,ysh-1,0,zsh-1)
    imageData.GetPointData().SetScalars(sc)

    # get poly data
    poly = MarchingCubes(imageData,threshold)

    if mode in ['displacement','strain']:

        # get points
        points = poly.GetPoints()
        array = points.GetData()
        numpy_points = vtk_to_numpy(array)

        # get cells
        cells = poly.GetPolys()
        array = cells.GetData()
        numpy_cells = vtk_to_numpy(array)
        numpy_cells = numpy_cells.reshape(-1,4)

        # get surface area and set scalar to cell
        array = vtk.vtkFloatArray()
        array.SetNumberOfComponents(1) # this is 3 for a vector
        array.SetNumberOfTuples(cells.GetNumberOfCells())
        nCells, nCols = numpy_cells.shape
        tmp = np.empty((nCells,3,3))
        if mode == 'displacement':
            # get maximum displacement
            max_displacement = int(np.amax(displacement_array))
            surface = np.zeros(max(16,max_displacement+1))
        for k in range(nCells):
            centroid = np.zeros(3)
            for l in range(1,4):
                id = numpy_cells[k,l]
                tmp[k,l-1] = numpy_points[id]   # x,y,z
                centroid += numpy_points[id]
            centroid /= 3
            centroid[0] /= x_spacing
            centroid[1] /= y_spacing
            centroid[2] /= z_spacing
            centroid = np.round(centroid).astype(int)
            trans_val = displacement_array[min(centroid[2],zsh-1),centroid[1],centroid[0]]
            array.SetValue(k, trans_val)
            if mode == 'displacement':
                # compute the surface areas depending on the magnitute
                A = tmp[k,0]
                B = tmp[k,1]
                C = tmp[k,2]
                a = B-A
                b = C-A
                magnitude = int(trans_val)
                surface[:magnitude+1] += 0.5*np.sqrt((a[1]*b[2]-a[2]*b[1])**2 + (a[2]*b[0]-a[0]*b[2])**2 + (a[0]*b[1]-a[1]*b[0])**2)
        poly.GetCellData().SetScalars(array)
        if mode == 'displacement':
            array.SetName("displacement")
        elif mode == 'strain':
            array.SetName("strain")

    # save data with vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(path_to_save)
    writer.Write()

    if mode == 'displacement':
        return np.round(surface*0.01,2)

