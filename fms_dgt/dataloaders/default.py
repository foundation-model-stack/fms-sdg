# Standard
from typing import Any

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.dataloader import BaseDataloader
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_dataloader


@register_dataloader("default")
class DefaultDataloader(BaseDataloader):
    """
    The default dataloader takes a datastore and iterates over it.

    @param state_datastore : A data store.
    @param loop_over_data  : If true, it wraps around when it reaches the end of the iterator.

    """

    def __init__(
        self,
        data: DATASET_TYPE,
        state_datastore: BaseDatastore = None,
        loop_over_data: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._data = data
        self._state_datastore = state_datastore
        self._i = 0
        self._loop_over_data = loop_over_data

    def get_state(self) -> Any:
        return self._i

    def set_state(self, state: Any) -> None:
        self._i = state

    def __next__(self) -> Any:
        try:
            value = self._data[self._i]
            self._i += 1
            return value
        except IndexError:
            # reset cycle
            if self._loop_over_data:
                self._i = 0
            raise StopIteration
