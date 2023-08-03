@echo off

REM Activate the virtual environment
call cclstm\Scripts\activate

REM Install required packages
pip install -r requirements.txt
