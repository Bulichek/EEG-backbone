import argparse
from pathlib import Path

import numpy as np
import pickle
from loguru import logger


def split_data(data: np.ndarray, path_to_save: Path, train_frac: float = 0.7) -> None:
    """
    Split data on train and validation
    """
    size = data.shape[0]
    all_ids = np.random.permutation(size)
    train_ids, val_ids = np.split(all_ids, [int(size * train_frac)])
    data_split = {"train": train_ids, "val": val_ids}
    with open(path_to_save, "wb") as file:
        pickle.dump(data_split, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--path-to-save", type=Path, required=True)
    args = parser.parse_args()

    logger.info('Splittig data.')
    data = np.load(args.data_path, allow_pickle=True)
    split_data(data, args.path_to_save, args.train_frac)
    logger.info('Data successfully splitted.')

main()