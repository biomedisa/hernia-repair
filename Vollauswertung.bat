git checkout main
git pull

cd ..
cd biomedisa
git checkout master
git pull
cd..

rmdir /s /Q Temp
mkdir Temp
cd Temp

call %userprofile%\anaconda3\Scripts\activate.bat
call activate biomedisa

call python C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\vollauswertung_automatisch.py

pause
