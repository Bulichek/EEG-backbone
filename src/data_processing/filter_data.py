import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
from loguru import logger


def filter_data(
    data: np.ndarray, requiered_labels: List[Union[str, int]], path_to_save: Path
) -> None:
    """
    Remove samples with redundant labels
    """
    filtered = [
        sample for sample in data if sample["label"][-1].lower() in requiered_labels
    ]
    with open(path_to_save, "wb") as file:
        np.save(file, np.array(filtered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--path-to-save", type=Path, required=True)
    parser.add_argument("--required-labels", type=str, required=True)
    args = parser.parse_args()

    logger.info("Filtering data.")
    data = np.load(args.data_path, allow_pickle=True)
    required_labels = [*args.required_labels]
    filter_data(data, required_labels, args.path_to_save)
    logger.info("Data successfully filtered.")


main()
