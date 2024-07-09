# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TypeVar, Union
import abc
import json
import os

# Local
from fms_dgt.base.registry import get_dataloader
from fms_dgt.dataloaders.default import DefaultDataloader
from fms_dgt.utils import group_data_by_attribute
import fms_dgt.dataloaders

DEFAULT_OUTPUT_DIR = "output"
DATALOADER_TYPE_KEY = "type"


@dataclass
class SdgData(abc.ABC):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_name: str

    def to_output_dict(self):
        return asdict(self)


class SdgTask:
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = SdgData
    OUTPUT_DATA_TYPE = (
        INPUT_DATA_TYPE  # default output data type is the main type of the task
    )

    def __init__(
        self,
        name: str,
        task_description: str,
        created_by: str,
        data_builder: str,
        output_dir: Optional[str] = None,
        dataloader: Optional[Dict] = None,
        seed_data: Optional[List[Any]] = None,
        num_outputs_to_generate: Optional[int] = None,
    ):
        self._name = name
        self._task_description = task_description
        self._created_by = created_by
        self._data_builder = data_builder
        self._seed_data = seed_data

        self._num_outputs_to_generate = num_outputs_to_generate
        self.machine_data = []

        self._output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        self._output_path = self._get_default_output_path()

        if dataloader is None:
            self._dataloader = DefaultDataloader(data=seed_data)
        else:
            assert (
                DATALOADER_TYPE_KEY in dataloader
            ), f"Must specify data loader type with '{DATALOADER_TYPE_KEY}' key"
            self._dataloader = get_dataloader(dataloader.pop(DATALOADER_TYPE_KEY))(
                **dataloader
            )

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=kwargs.pop("task_name", self._name), **kwargs
        )

    def instantiate_output_example(self, **kwargs: Any):
        return self.OUTPUT_DATA_TYPE(**kwargs)

    def get_example(self) -> SdgData:
        try:
            return self.instantiate_input_example(**next(self._dataloader))
        except StopIteration:
            return None

    def get_all_examples(self) -> List[SdgData]:
        outputs = []
        next_output = self.get_example()
        while next_output is not None:
            outputs.append(next_output)
            next_output = self.get_example()
        return outputs

    @property
    def name(self):
        return self._name

    @property
    def seed_data(self):
        return self._seed_data

    @property
    def task_description(self):
        return self._task_description

    @property
    def data_builder(self):
        return self._data_builder

    @property
    def output_path(self) -> str:
        return self._output_path

    @property
    def num_outputs_to_generate(self):
        return self._num_outputs_to_generate

    def is_complete(self):
        return len(self.machine_data) > self.num_outputs_to_generate

    def _get_default_output_path(self):
        path_components = []
        path_components.append(self._output_dir)
        path_components.append(self._name)
        path_components.append("generated_instructions.jsonl")
        return os.path.join(*path_components)

    def save_data(
        self, new_data: Union[SdgData, List[SdgData]], output_path: str = None
    ) -> None:
        if type(new_data) != list:
            new_data = [new_data]

        output_path = self._output_path if output_path is None else output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            for d in new_data:
                f.write(json.dumps(d.to_output_dict()) + "\n")

    def load_data(self, output_path: str = None) -> List[SdgData]:
        output_path = self._output_path if output_path is None else output_path
        with open(output_path, "r") as f:
            try:
                machine_data = [
                    self.instantiate_output_example(**json.loads(l.strip()))
                    for l in f.readlines()
                ]
            except ValueError:
                machine_data = []

        self.machine_data = machine_data

    def clear_data(self, output_path: str = None) -> List[SdgData]:
        output_path = self._output_path if output_path is None else output_path
        if os.path.exists(output_path):
            os.remove(output_path)


T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    return group_data_by_attribute(data_list, "task_name")
