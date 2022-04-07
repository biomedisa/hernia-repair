import sys
import os

sys.path.append(f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64\\bin\\Lib\\site-packages\\paraview')
sys.path.append(f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64\\bin\\Lib\\site-packages\\win32\\libs')
sys.path.append(f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64\\bin\\Lib\\site-packages\\win32com\\libs')
print(sys.path)

from paraview.simple import *

mesh = sys.argv[1]
path_to_save = sys.argv[2]

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
scalarsLUT.RescaleTransferFunction(1.0, 7.0)

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

# hide color bar/color legend
ParaviewDisplay.SetScalarBarVisibility(renderView1, False)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# reset view to fit data
renderView1.ResetCamera()

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
