call "%Userprofile%\anaconda3\Scripts\activate.bat"
call conda env create -f "%Userprofile%\git\hernia-repair\environment.yml"
call activate biomedisa
pause

