@ECHO OFF
ECHO Setting up..

py -m venv .venv
CALL .venv\Scripts\activate
pip install -r requirements.txt

git init
git status

git stage cli.py train.py train_and_shutdown.py requirements.txt
git commit -m "initial commit"

git stage configs/*_reference.yaml
git commit -m "reference configs"

git status

PAUSE

DEL setup.bat
