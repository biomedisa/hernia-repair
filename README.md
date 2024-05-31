# Hernia-Repair
## Necessary Setup Steps
### __Install Git__
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).


### __Install Anaconda3__
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).<br />
**Important Note!!!**: Anaconda must be installed within the *%Userprofile%* directory!

### __Install Paraview (Version 5.10.1 for Python 3.9)__
Install [Paraview](https://www.paraview.org/download/) in the same directory as Anaconda *(%Userprofile%)*.

### __Install NVIDIA Driver__
Download and install the correct Nvidia driver [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

### __Clone HEDI repository__
The repository should be in the directory: *%Userprofile%\git*

Open the cmd consol and enter:
```
mkdir git
cd git
git clone https://github.com/biomedisa/hernia-repair.git 
``` 

## __Automatic Setup and Execution__
This section explains how to setup and run HEDI 
automatically with the included batch files.
Experienced users can also enter the commands manually
as explained in the next section.

Within the cloned repository lies the batchfile `setup.bat`
Running this file will automaticly create an enviornment 
with all dependencies needed to run HEDI.

## __Manual Setup and Execution (optional)__
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`)
or activate conda within the cmd shell via: 
```
call "%Userprofile%\anaconda3\Scripts\activate.bat"
```
Create and activate the conda environment:
```
conda env create -f environment.yml
```
The .yml file contains all infos about the required packages
needed to run HEDI.

You can check if the enviornment was created by:
```
conda activate biomedisa
```
To Run the application use the command:
```
python "%Userprofile%\git\hernia-repair\HEDI_main.py" Single
```
### __Configure the config file__

Check all paths in the `config_base.py` file, if any paths in this file need a name change also rename the file to `config.py`.
Any further changes should then be restricted to the file named `config.py`.

After a successfull setup HEDI can be executed by running
the batchfile `HEDI.bat`.

### __How to use the application__
  When running the application the user will be asked by a popup window to select the dataset containing the Dicomdata of one patient.
  ![Alt Text](res/SelectDataset.png)
  
  The data is then sorted and stored in the directory *%Userprofile%\Hernia_Analysis_Results\Patient_Name\Dicom_Data*<br />
  Next the user will be asked by a popup window to select the directory within the above directory that contains the data at rest in axial direction.
   ![Alt Text](res/SelectRest.png)
  
  Afterwards the corresponding data during valslva needs to be selected in the same way.
   ![Alt Text](res/SelectValsalva.png)
  
  Lastly the user is asked to select a threshold value for the area of instability. (Default = 15mm)
   ![Alt Text](res/SelectThreshold.png)
  
  Deppending on your machine the application now runs for 4-20 min. <br />
  All files are stored in the above mentioned directory *%Userprofile%\Hernia_Analysis_Results\Patient_Name* . <br />
  When finished the endresult is presented as an image.
  
# License

This project is covered under the **EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL)**.
  
  
  
  
  

