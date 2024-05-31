git checkout main
git pull

call "%Userprofile%\anaconda3\Scripts\activate.bat"
call activate biomedisa
call python "%Userprofile%\git\hernia-repair\HEDI_main.py" Single

pause
