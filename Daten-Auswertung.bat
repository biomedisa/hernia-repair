cd ..
cd Documents
cd Python_Scripts
call C:\Users\Hernienforschung\anaconda3\Scripts\activate.bat C:\Users\Hernienforschung\anaconda3
call activate biomedisa
cd temporäre_Dateien

git checkout main
git pull git@github.com:biomedisa/hernia-repair.git

call python C:\Users\Hernienforschung\Documents\Python_Scripts\hernia-repair\Einlesungs_script.py

Del "A*"

pause