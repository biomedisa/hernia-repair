git checkout main
git pull

cd ..

cd biomedisa
git checkout master
git pull
cd ..


call "%Userprofile%\anaconda3\Scripts\activate.bat" 
call activate biomedisa

rmdir /s /Q Temp
mkdir Temp

cd Temp

call python "%Userprofile%\git\hernia-repair\Einlesungs_script.py" Single

pause
