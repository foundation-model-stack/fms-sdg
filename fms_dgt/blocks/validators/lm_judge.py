# Standard
from typing import Any, Callable, Dict, List, Optional

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseValidatorBlock
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.blocks.generators.llm import LMGenerator

TYPE_KEY = "type"


@register_block("llm_judge")
class LMJudgeValidator(BaseValidatorBlock):
    """LLM-based Validator"""

    def __init__(self, lm: Dict = None, **kwargs: Any):
        super().__init__(**kwargs)
        assert (
            TYPE_KEY in lm
        ), f"Must specify {TYPE_KEY} in 'lm' field of {self.name} block"

        self._llm_generator: LMGenerator = get_block(lm.pop(TYPE_KEY))(**lm)
        self.blocks = [self._llm_generator]

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        success_func: Callable = None,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs,
    ):

        # simplify generation here
        llm_outputs = self._llm_generator.generate(
            inputs,
            arg_fields=arg_fields,
            kwarg_fields=kwarg_fields,
            result_field=result_field,
            **kwargs,
        )

        for llm_output in llm_outputs:
            lm_res = self.get_result(llm_output, result_field)
            new_result = success_func(lm_res)
            self.write_result(llm_output, new_result, result_field=result_field)

        return llm_outputs

    def _validate(self, lm_output: str, success_func: Callable) -> bool:
        return success_func(lm_output)
