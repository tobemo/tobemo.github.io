import os

from cli import get_cli

if __name__ == "__main__":
    get_cli()
    os.system("shutdown -s -t 360") # shutdown in 60s
