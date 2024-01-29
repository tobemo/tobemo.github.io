@ECHO OFF
ECHO Setting up..

py -m venv .venv
CALL .venv\Scripts\activate
pip install -r requirements.txt

PAUSE

DEL setup.bat
