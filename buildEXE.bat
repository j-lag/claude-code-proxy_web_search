call ../../Python312_x64/python_env.bat
rem python -m nuitka --mode=standalone --assume-yes-for-downloads start_proxy.py
python -m nuitka --mode=onefile --assume-yes-for-downloads start_proxy.py

pause