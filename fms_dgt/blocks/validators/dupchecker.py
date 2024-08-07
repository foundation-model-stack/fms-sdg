# Standard
from typing import Any, Dict, List, Optional, Union

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block

@register_block("dupchecker")
class DupCheckerValidator(BaseValidatorBlock):
    """Check for duplicates in the input"""

    def tokenize(self, inp: Union[List, str]):
        # This doesn't have to tokenize,
        # the input may not even be strings
        return inp

    def _validate(self, item: Any, check_against: List[Any]) -> bool:
        return item not in check_against
