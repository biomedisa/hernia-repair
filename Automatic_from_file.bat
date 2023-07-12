git checkout main
git pull
cd ..


cd biomedisa
git checkout master
git pull
cd ..


cd hernia-repair
cd data


call "%Userprofile%\anaconda3\Scripts\activate.bat"
call activate biomedisa
call python "%Userprofile%\git\hernia-repair\HEDI_main.py" Multi

pause
