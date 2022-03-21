git checkout main
git pull

cd ..

call \anaconda3\Scripts\activate.bat \anaconda3
call activate biomedisa

rmdir /s /Q Temp
mkdir Temp

call python \hernia-repair\Einlesungs_script.py

pause
