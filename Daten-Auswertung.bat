git checkout 21.03
git pull

cd ..

cd biomedisa
git checkout master
git pull
cd ..


call %Userprofile%\anaconda3\Scripts\activate.bat 
call activate biomedisa

rmdir /s /Q Temp
mkdir Temp

call python %Userprofile%\git\hernia-repair\Einlesungs_script.py

pause
