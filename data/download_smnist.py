"""Download sequential MNIST data from https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data."""
import requests
from pathlib import Path
import tarfile


URL = r"https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
fp = Path('data')


if __name__ == "__main__":
    print("Fetching tar..")
    response = requests.get(URL, stream=True)
    print(" status:", response.status_code)
    
    print("Extracting tar..")
    tar = tarfile.open(fileobj=response.raw, mode='r|gz')
    tar.extractall(fp)
    tar.close()
