import sys

try:
    import config as config
except:
    import config_base as config

if sys.platform == "win32":
    sys.path.insert(0,f'{config.path_names["paraview"]}\\bin\\Lib')
    sys.path.insert(0,f'{config.path_names["paraview"]}\\bin\\Lib\\site-packages')
else:
    sys.path.insert(0,f'{config.path_names["paraview"]}/lib')
    sys.path.insert(0,f'{config.path_names["paraview"]}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')

from paraview.simple import *

def get_displacment_colormap(renderView,threshold,MaxDisplacement):
    ColorMap = GetColorTransferFunction('displacement')

    # Set Color Tables [Label_value_1,R,G,B,
    #                   Label_value_2,R,G,B,
    #                   ...
    #                  ]
    
    ColorMap.ApplyPreset('Cold and Hot', True)
    if MaxDisplacement <= threshold:
        ColorMap.RGBPoints =    [0.0, 0.0, 1.0, 1.0,
                                 MaxDisplacement, 0.0, 0.0, MaxDisplacement/threshold
                                ]

    elif (MaxDisplacement - threshold) <= 30:
        ColorMap.RGBPoints =    [0.0, 0.0, 1.0, 1.0,
                                 threshold, 0.0, 0.0, 1.0,
                                 threshold, 1.0, 0.0, 0.0,
                                 MaxDisplacement, 1.0, (MaxDisplacement-threshold)/30, 0.0
                                ]
    
    elif (MaxDisplacement - threshold) <= 60:
        ColorMap.RGBPoints =    [0.0, 0.0, 1.0, 1.0,
                                 threshold, 0.0, 0.0, 1.0,
                                 threshold, 1.0, 0.0, 0.0,
                                 (threshold +30), 1.0, 1.0, 0.0,
                                 MaxDisplacement, 1.0, 1.0, (MaxDisplacement-threshold-30)/30
                                ]

    else:
        ColorMap.RGBPoints =    [0.0, 0.0, 1.0, 1.0,
                                 threshold, 0.0, 0.0, 1.0,
                                 threshold, 1.0, 0.0, 0.0,
                                 (threshold + 30), 1.0, 1.0, 0.0,
                                 (threshold + 60), 1.0, 1.0, 1.0,
                                 MaxDisplacement, 1.0, 0.0, 1.0
                                ]

    # get color legend/bar 
    ColorBar = GetScalarBar(ColorMap, renderView)
    ColorBar.HorizontalTitle = 1
    ColorBar.LabelFontSize = 10

    return ColorMap, ColorBar


def get_strain_colormap(renderView):
    ColorMap = GetColorTransferFunction('strain')
    ColorMap.ApplyPreset('Cool to Warm', True)
    ColorMap.Discretize = 1
    ColorMap.NumberOfTableValues = 10

    ColorBar = GetScalarBar(ColorMap, renderView)
    ColorBar.LabelFontSize = 10

    

    return ColorMap, ColorBar


def get_label_colormap(renderView):
    ColorMap = GetColorTransferFunction('scalars')
    ColorMap.InterpretValuesAsCategories = 1
    ColorMap.AnnotationsInitialized = 1

    ColorMap.Annotations      = ['1', 'abd. rect. (r)',
                                 '2', 'abd. rect. (l)', 
                                 '3', 'abd. obl. (r)', 
                                 '4', 'abd. obl. (l)', 
                                 '5', 'abd. Cavity volume', 
                                 '7', 'Hernia volume']

    ColorMap.IndexedColors    = [0.231, 0.298, 0.753, 
                                 0.435, 0.569, 0.953, 
                                 0.675, 0.784, 0.992, 
                                 0.865, 0.865, 0.865, 
                                 0.969, 0.72, 0.612, 
                                 0.706, 0.0157, 0.149]

    ColorMap.IndexedOpacities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    ColorBar = GetScalarBar(ColorMap, renderView)
    ColorBar.HorizontalTitle = 1
    ColorBar.LabelFontSize = 10

    return ColorMap, ColorBar


def get_grid(renderView,DataBounds):
    # Hide orientation axes
    renderView.OrientationAxesVisibility = 0

    # Show the length axes
    renderView.AxesGrid.Visibility = 1

    # Adjust the Grids properties
    renderView.AxesGrid.ShowGrid = 1
    renderView.AxesGrid.GridColor = [0.0, 0.0, 0.0]

    renderView.AxesGrid.ShowEdges = 0
    renderView.AxesGrid.ShowTicks = 1

    renderView.AxesGrid.CullBackface = 1
    renderView.AxesGrid.CullFrontface = 0
    renderView.AxesGrid.AxesToLabel = 7

    # Set The X-axis properties
    renderView.AxesGrid.XTitle = 'X Axis\n (mm)'
    renderView.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
    renderView.AxesGrid.XTitleBold = 1
    renderView.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]

    # Set The Z-axis properties
    renderView.AxesGrid.ZTitle = 'Z Axis\n (mm)'
    renderView.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
    renderView.AxesGrid.ZTitleBold = 1
    renderView.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

    #Set Custom Bounds
    renderView.AxesGrid.UseCustomBounds = 1
    renderView.AxesGrid.CustomBounds = DataBounds


def make_screenshot(DataBounds,renderView,path_to_save):

    # get layout
    layout = GetLayout()

    # layout/tab size in pixels
    layout.SetSize(600, 600)

    # current camera placement for renderView1
    renderView1.CameraPosition = [0.5*(DataBounds[0]+DataBounds[1]), -1000, 0.5*(DataBounds[4]+DataBounds[5])]
    renderView1.CameraFocalPoint = [0.5*(DataBounds[0]+DataBounds[1]), 0., 0.5*(DataBounds[4]+DataBounds[5])]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 300.

    # Activate camera parallel projection
    renderView1.CameraParallelProjection = 1
    
    # Reset the camera to fit all data
    #renderView1.ResetCamera()

    # save screenshot
    SaveScreenshot(path_to_save, renderView, ImageResolution=[600, 600],
        OverrideColorPalette='WhiteBackground')


displacement_mesh = sys.argv[1]
strain_mesh       = sys.argv[2]
label_mesh        = sys.argv[3]
path_to_save = sys.argv[4]
threshold = float(sys.argv[5])

if displacement_mesh != 'None':
    # create a new 'Legacy VTK Reader'
    DisplacementMesh = LegacyVTKReader(registrationName='displacement_mesh', FileNames=[displacement_mesh])
    # get the maximum Value of the Data Range and their bounds
    MaxDisplacement = DisplacementMesh.CellData['displacement'].GetRange()[1]
    BodyBounds = DisplacementMesh.GetDataInformation().DataInformation.GetBounds()
    # set active source
    SetActiveSource(DisplacementMesh)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    DisplacementDisplay = Show(DisplacementMesh, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'mode'
    DisplacementColorMap, DisplacementColorBar = get_displacment_colormap(renderView1,threshold,MaxDisplacement)

    # trace defaults for the display properties.
    DisplacementDisplay.Representation = 'Surface'
    DisplacementDisplay.ColorArrayName = ['CELLS', 'displacement']
    DisplacementDisplay.LookupTable = DisplacementColorMap
    DisplacementDisplay.SelectTCoordArray = 'None'
    DisplacementDisplay.SelectNormalArray = 'None'
    DisplacementDisplay.SelectTangentArray = 'None'
    DisplacementDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    DisplacementDisplay.SelectOrientationVectors = 'None'
    DisplacementDisplay.ScaleFactor = 42.
    DisplacementDisplay.SelectScaleArray = 'displacement'
    DisplacementDisplay.GlyphType = 'Arrow'
    DisplacementDisplay.GlyphTableIndexArray = 'displacement'
    DisplacementDisplay.GaussianRadius = 2.
    DisplacementDisplay.SetScaleArray = [None, '']
    DisplacementDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    DisplacementDisplay.OpacityArray = [None, '']
    DisplacementDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    DisplacementDisplay.DataAxesGrid = 'GridAxesRepresentation'
    DisplacementDisplay.PolarAxes = 'PolarAxesRepresentation'

    # show color bar/color legend
    DisplacementDisplay.SetScalarBarVisibility(renderView1, True)
    # hide the scalar bar for this color map if no visible data is colored by it.
    DisplacementDisplay.RescaleTransferFunctionToDataRange(False,True)
    # get the material library
    materialLibrary1 = GetMaterialLibrary()
    # get the datagrid
    get_grid(renderView1,BodyBounds)
    # make a screenshot of the curent Data
    make_screenshot(BodyBounds,renderView1,f'{path_to_save}_displacement.png')
    # hide data in view
    Hide(DisplacementMesh, renderView1)

if strain_mesh != 'None':
    StrainMesh = LegacyVTKReader(registrationName='strain_mesh', FileNames=[strain_mesh])

    if not BodyBounds:
        BodyBounds = StrainMesh.GetDataInformation().DataInformation.GetBounds()
    # set active source
    SetActiveSource(StrainMesh)

    # show data in view
    StrainDisplay = Show(StrainMesh, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'mode'
    StrainColorMap, StrainColorBar = get_strain_colormap(renderView1)

    StrainDisplay.Representation = 'Surface'
    StrainDisplay.ColorArrayName = ['CELLS', 'strain']
    StrainDisplay.LookupTable = StrainColorMap
    StrainDisplay.SelectTCoordArray = 'None'
    StrainDisplay.SelectNormalArray = 'None'
    StrainDisplay.SelectTangentArray = 'None'
    StrainDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    StrainDisplay.SelectOrientationVectors = 'None'
    StrainDisplay.ScaleFactor = 42.
    StrainDisplay.SelectScaleArray = 'strain'
    StrainDisplay.GlyphType = 'Arrow'
    StrainDisplay.GlyphTableIndexArray = 'strain'
    StrainDisplay.GaussianRadius = 2.
    StrainDisplay.SetScaleArray = [None, '']
    StrainDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    StrainDisplay.OpacityArray = [None, '']
    StrainDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    StrainDisplay.DataAxesGrid = 'GridAxesRepresentation'
    StrainDisplay.PolarAxes = 'PolarAxesRepresentation'

    # Show color bar/color legend
    StrainDisplay.SetScalarBarVisibility(renderView1, True)
    # get the datagrid
    get_grid(renderView1,BodyBounds)
    # make a screenshot of the curent Data
    make_screenshot(BodyBounds,renderView1,f'{path_to_save}_strain.png')
    # hide data in view
    Hide(StrainMesh, renderView1)

if label_mesh != 'None':

    LabelMesh = LegacyVTKReader(registrationName='label_mesh', FileNames=[label_mesh])
    LabelBounds = LabelMesh.GetDataInformation().DataInformation.GetBounds()
    # set active source
    SetActiveSource(LabelMesh)

    # show data in view
    LabelDisplay = Show(LabelMesh, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'mode'
    LabelColorMap, LabelColorBar = get_label_colormap(renderView1)

    LabelDisplay.Representation = 'Surface'
    LabelDisplay.ColorArrayName = ['CELLS', 'scalars']
    LabelDisplay.LookupTable = LabelColorMap
    LabelDisplay.SelectTCoordArray = 'None'
    LabelDisplay.SelectNormalArray = 'None'
    LabelDisplay.SelectTangentArray = 'None'
    LabelDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    LabelDisplay.SelectOrientationVectors = 'None'
    LabelDisplay.ScaleFactor = 42.
    LabelDisplay.SelectScaleArray = 'scalars'
    LabelDisplay.GlyphType = 'Arrow'
    LabelDisplay.GlyphTableIndexArray = 'scalars'
    LabelDisplay.GaussianRadius = 2.
    LabelDisplay.SetScaleArray = [None, '']
    LabelDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    LabelDisplay.OpacityArray = [None, '']
    LabelDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    LabelDisplay.DataAxesGrid = 'GridAxesRepresentation'
    LabelDisplay.PolarAxes = 'PolarAxesRepresentation'

    # show color bar/color legend
    LabelDisplay.SetScalarBarVisibility(renderView1, True)
    # set color bar name
    LabelColorBar.Title = 'Labels'
    # make a screenshot of the curent Data
    if BodyBounds:
        make_screenshot(BodyBounds,renderView1,f'{path_to_save}_labels.png')
    else:
        # get the datagrid
        get_grid(renderView1,LabelBounds)
        make_screenshot(LabelBounds,renderView1,f'{path_to_save}_labels.png')

    # hide data in view
    Hide(LabelMesh, renderView1)

