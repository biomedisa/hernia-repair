git checkout 21.03
git pull

cd ..

cd biomedisa
git checkout main
git pull
cd ..
cd ..


call anaconda3\Scripts\activate.bat \anaconda3
call activate biomedisa

cd git

rmdir /s /Q Temp
mkdir Temp

call python \hernia-repair\Einlesungs_script.py

rmdir /s /Q Temp

pause
