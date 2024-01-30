"""https://gist.github.com/clvcooke/c050fddeedda3451fd4ccc6829cbff11"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(files: list[Path], labels: list[int], n: int = -1) -> pd.DataFrame:
    """Txt file to dataframe.
    Args:
        files (list[Path]): Data file paths.
        labels (list[int]): Label file paths.
        n (int, optional): How much points to load. Useful for debugging. Defaults to -1.

    Returns:
        pd.DataFrame: Dataframe with rows holding both x and y coordinates for each sample and columns holding sequence data.
    """
    data = []
    for fp in files[:n]:
        sample = pd.read_csv(fp, sep=' ', header=None, names=['dx', 'dy', 'eos', 'eod'])
        sample = sample.drop(['eos', 'eod'], axis=1)
        sample = sample.T
        sample = sample.iloc[ :, 1:-1 ] # first col holds starting coordinate; last is always 0
        data.append(
            sample
        )

    data = pd.concat(data, axis=0)
    data.columns = pd.RangeIndex(0, data.shape[1])
    new_index = pd.MultiIndex.from_product([labels[:n], ['dx', 'dy']], names=['Number', 'Direction'])
    new_index = pd.MultiIndex.from_tuples([(a, *b) for a, b in zip(np.arange(len(labels[:n]) * 2)//2, new_index)], names=['id', 'number', 'direction'])
    data.index = new_index
    
    return data


if __name__ == '__main__':
    fp_base = Path('data')
    
    train_files = (Path('data') / "sequences").glob("trainimg-*-inputdata.txt")
    test_files = (Path('data') / "sequences").glob("testimg-*-inputdata.txt")

    train_files = sorted(train_files, key=lambda s: int(s.stem.split('-')[1]))
    test_files = sorted(test_files, key=lambda s: int(s.stem.split('-')[1]))
    
    train_labels = np.loadtxt(Path('data') / "sequences" / "trainlabels.txt", dtype=np.int32)
    test_labels = np.loadtxt(Path('data') / "sequences" / "testlabels.txt", dtype=np.int32)
    
    train_data = load_data(train_files, train_labels)
    test_data = load_data(test_files, test_labels)
    
    train_data.to_parquet(fp_base / "train.parquet")
    test_data.to_parquet(fp_base / "test.parquet")
