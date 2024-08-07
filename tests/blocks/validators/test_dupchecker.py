# Standard
from typing import List
import json

# Third Party
import pytest

# Local
from fms_dgt.blocks.validators.dupchecker import DupCheckerValidator


class TestDupCheckerValidator:
    def test_matches(self):
        validator = DupCheckerValidator(name="test_dupchecker_validator")
        all_data = [
            "I went to the store",
            "I went to the store yesterday",
            "blue red yellow green",
        ]

        data_entry = "I went to the store"
        inputs = [{"a": data_entry, "b": all_data}]
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert not inputs[0]["result"]

        data_entry = "one two three"
        inputs = [{"a": data_entry, "b": all_data}]
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert inputs[0]["result"]
