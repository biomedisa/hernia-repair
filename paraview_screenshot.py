import sys
import os

sys.path.insert(0,f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64\\bin\\Lib')
sys.path.insert(0,f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64\\bin\\Lib\\site-packages')

from paraview.simple import *

mesh = sys.argv[1]
path_to_save = sys.argv[2]
mode = sys.argv[3]
max_range = float(sys.argv[4])

Connect()
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
Paraview = LegacyVTKReader(registrationName='first', FileNames=[mesh])

# set active source
SetActiveSource(Paraview)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
ParaviewDisplay = Show(Paraview, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'scalars'
scalarsLUT = GetColorTransferFunction('scalars')

# trace defaults for the display properties.

ParaviewDisplay.Representation = 'Surface'
ParaviewDisplay.ColorArrayName = ['CELLS', 'scalars']
ParaviewDisplay.LookupTable = scalarsLUT
ParaviewDisplay.SelectTCoordArray = 'None'
ParaviewDisplay.SelectNormalArray = 'None'
ParaviewDisplay.SelectTangentArray = 'None'
ParaviewDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ParaviewDisplay.SelectOrientationVectors = 'None'
ParaviewDisplay.ScaleFactor = 42.19332799911499
ParaviewDisplay.SelectScaleArray = 'scalars'
ParaviewDisplay.GlyphType = 'Arrow'
ParaviewDisplay.GlyphTableIndexArray = 'scalars'
ParaviewDisplay.GaussianRadius = 2.10966639999557497
ParaviewDisplay.SetScaleArray = [None, '']
ParaviewDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ParaviewDisplay.OpacityArray = [None, '']
ParaviewDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ParaviewDisplay.DataAxesGrid = 'GridAxesRepresentation'
ParaviewDisplay.PolarAxes = 'PolarAxesRepresentation'

# show color bar/color legend
ParaviewDisplay.SetScalarBarVisibility(renderView1, True)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Hide the scalar bar for this color map if no visible data is colored by it.
ParaviewDisplay.RescaleTransferFunctionToDataRange(False,True)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# reset view to fit data
renderView1.ResetCamera()
if mode == "translation":
    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    scalarsLUT.ApplyPreset('Cold and Hot', True)
    scalarsLUT.RGBPoints = [1.0, 0.0, 1.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 0.5, 15.0, 1.0, 0.0, 0.0, max_range, 1.0, 1.0, 0.0]


# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(590, 590)

# current camera placement for renderView1
renderView1.CameraPosition = [263.2996940612793, -945.4659242482333, 214.03336000442505]
renderView1.CameraFocalPoint = [263.2996940612793, 255.01700592041016, 214.03336000442505]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = 310.70784564812436

# save screenshot
SaveScreenshot(path_to_save, renderView1, ImageResolution=[590, 590],
    OverrideColorPalette='WhiteBackground')
#Reset the session
Disconnect()
