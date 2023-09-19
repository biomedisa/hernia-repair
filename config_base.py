import os,sys
'''
main     :   Main directory for result storage
neuralnet:   Directory containing the neuralnetworks 
paraview :   Directory containing the Paraview files
multipath:   Txt file containing the File paths for multi automatic evaluation
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    # Windows
    path_names = {
        'userprofile': os.environ["userprofile"],
        'main': f'{os.environ["userprofile"]}\\Hernia_Analysis_Results',
        'neuralnet': BASE_DIR+'/data/neuralnet',
        'paraview' : f'{os.environ["userprofile"]}\\Paraview\\ParaView 5.10.1-Windows-Python3.9-msvc2017-AMD64',
        'multipath': BASE_DIR+'/data/Paths/Paths.txt',
    }
else:
    # Linux
    path_names = {
        'userprofile': os.path.expanduser("~"),
        'main': BASE_DIR+'/data/Hernia_Analysis_Results',
        'neuralnet': BASE_DIR+'/data/neuralnet',
        'paraview' : '/opt/ParaView-5.10.1',
        'multipath': BASE_DIR+'/data/Paths/Paths.txt',
    }
