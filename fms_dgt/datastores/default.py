# Standard
from typing import List, TypeVar
import json
import os

# Third Party
import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_datastore

T = TypeVar("T")


@register_datastore("default")
class DefaultDatastore(BaseDatastore):
    """
    The default datastore is a data store that takes a name of a `.jsonl`,
    `.json`, `.yaml`, or `parquet` file, loads it, and returns an iterator
    over it.

    If none of the filename extension matches, it attempts to give the first
    argument to huggingface's `datasets.load_dataset` function to load the data, and returns an iterator over it.

    It also has an ability to dump the data into a file.

    @param output_dir  : The destination directory for writing the data.
    @param data_format : The data format for writing the data. It is not used for loading; The format is guessed from data_path.
    @param data        : A list. Data loaded from the files are concatenated with this list.
    @param data_path   : The path to the file to load. If it is a list, it is interpreted as the arguments to huggingface's `datasets.load_dataset` .

    """

    def __init__(
        self,
        output_dir: str = None,
        data_format: str = "jsonl",
        data: List[T] = None,
        data_path: str = None,
        data_split: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._output_path = (
            os.path.join(output_dir, self.store_name + "." + data_format)
            if output_dir
            else None
        )
        self._data_path = data_path
        self._data_split = data_split
        self._data = data or []
        if self._restart and os.path.exists(self._output_path):
            os.remove(self._output_path)

        os.makedirs(output_dir, exist_ok=True)

    def save_data(self, new_data: DATASET_TYPE) -> None:

        data_format = os.path.splitext(self._output_path)[-1]

        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.to_dict("records")

        if data_format == ".jsonl":
            _write_json(new_data, self._output_path)
        elif data_format == ".yaml":
            raise NotImplementedError
        elif data_format == ".parquet":
            _write_parquet(new_data, self._output_path)
        else:
            raise ValueError(f"Unhandled data format: {data_format}")

    def load_data(self) -> List[T]:

        data = self._data
        loaded_data = []
        data_path = self._data_path if self._data_path else self._output_path

        if type(data_path) == list:
            loaded_data = _read_huggingface(data_path, self._data_split)
        elif os.path.exists(data_path):
            output_format = os.path.splitext(data_path)[-1]
            if output_format == ".jsonl":
                loaded_data = _read_jsonl(data_path)
            elif output_format == ".json":
                loaded_data = _read_json(data_path)
            elif output_format == ".yaml":
                loaded_data = _read_yaml(data_path)
            elif output_format == ".parquet":
                loaded_data = _read_parquet(data_path)
            elif os.path.isdir(data_path):
                loaded_data = _read_huggingface([data_path], self._data_split)
            else:
                raise ValueError(f"Unhandled output format: {output_format}")

        data = _join_data(loaded_data, data)

        return data


###
# Utilities
###


def _write_json(new_data: List[T], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")


def _write_parquet(new_data: List[T], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(new_data).to_parquet(
        output_path,
        engine="fastparquet",
        append=os.path.isfile(output_path),
    )


def _read_jsonl(output_path: str):
    with open(output_path, "r") as f:
        try:
            machine_data = [json.loads(l.strip()) for l in f.readlines()]
        except ValueError:
            machine_data = []
    return machine_data


def _read_json(output_path: str):
    with open(output_path, "r") as f:
        try:
            machine_data = json.load(f)
        except ValueError:
            machine_data = []
    return machine_data


def _read_parquet(output_path: str):
    machine_data = (
        pd.read_parquet(output_path, engine="fastparquet").apply(dict, axis=1).to_list()
    )
    return machine_data


def _read_yaml(output_path: str):
    with open(output_path, "r") as f:
        machine_data = list(yaml.safe_load(f))
    return machine_data


def _read_huggingface(dataset_args: List[str], split: str):
    data = datasets.load_dataset(*dataset_args)
    if split:
        data = data[split]
    return data


def _join_data(dataset: DATASET_TYPE, seed_data: List):
    if seed_data:
        if type(dataset) == list:
            dataset = dataset + seed_data
        elif type(dataset) == datasets.Dataset:
            seed_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=seed_data))
            dataset = datasets.concatenate_datasets([dataset, seed_dataset])
        else:
            raise ValueError(
                f"Data used for default 'load_dataset' method must be one of {DATASET_TYPE}!"
            )
    return dataset
