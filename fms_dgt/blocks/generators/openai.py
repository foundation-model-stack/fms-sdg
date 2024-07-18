"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.util import find_spec
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import copy

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.blocks.generators.llm import MODEL_ID_OR_PATH, LMGenerator
from fms_dgt.resources.openai import OpenAIKeyResource
from fms_dgt.utils import sdg_logger
import fms_dgt.blocks.generators.utils as generator_utils
import fms_dgt.utils as utils

try:
    # Third Party
    from openai import OpenAI
except ModuleNotFoundError:
    pass


def oa_completion(client, chat: bool = False, **kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install fms_dgt[openai]`"
        )
    else:
        # Third Party
        import openai

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        # Standard
        import traceback

        traceback.print_exc()

    @generator_utils.retry_on_specific_exceptions(
        on_exceptions=[openai.OpenAIError],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        if chat:
            return client.chat.completions.create(**kwargs)
        else:
            return client.completions.create(**kwargs)

    return completion()


@register_block("openai", "vllm-remote")
class OpenaiCompletionsLM(LMGenerator):
    def __init__(
        self,
        base_url: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        try:
            # Third Party
            import openai  # noqa: E401
            import tiktoken
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install .[openai]`",
            )

        self.base_url = base_url
        self._chat = False

        if self.batch_size is None:
            self._batch_size = 1

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        if self.base_url:
            self.client = openai.OpenAI(api_key="EMPTY", base_url=self.base_url)
        else:
            self._openai_resource: OpenAIKeyResource = get_resource(
                "openai", "OPENAI_API_KEY"
            )
            self.client = OpenAI(api_key=self._openai_resource.key)

    def _prepare_input(self, prompt: str):
        return prompt

    def _extract_output(self, resp) -> str:
        return resp.text

    def generate_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        # we group requests by their generation_kwargs,
        grouper = generator_utils.Grouper(requests, lambda x: str(x.kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )
        for key, reqs in grouper.get_grouped().items():
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self.batch_size
            )

            for chunk in chunks:
                inputs = [self._prepare_input(instance.args[0]) for instance in chunk]
                # all kwargs are identical
                gen_kwargs = next(iter(chunk)).kwargs
                kwargs = self.modify_gen_kwargs(gen_kwargs)
                kwargs[("messages" if self._chat else "prompt")] = inputs

                model_id = kwargs.pop("model_id_or_path", self.model_id_or_path)
                until = kwargs.get("stop", None)

                response = oa_completion(
                    client=self.client,
                    chat=self._chat,
                    model=model_id,
                    **kwargs,
                )

                for resp, instance in zip(response.choices, chunk):
                    s = self._extract_output(resp)
                    self.update_instance_with_result(
                        "generate_batch", s, instance, until
                    )
                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(self, *args, **kwargs):
        raise NotImplementedError("No support for logits.")

    def modify_gen_kwargs(self, gen_kwargs: dict) -> dict:
        # sampling_params
        if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
            # start with default params then overwrite with kwargs
            kwargs = {**self._base_kwargs, **kwargs}
            until = None
            if "stop_sequences" in kwargs:
                until = kwargs.pop("stop_sequences")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `kwargs['stop_sequences']` to be of type Union[str,list] but got {until}"
                    )
            kwargs["stop"] = until
            if "max_new_tokens" in kwargs.keys():
                kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
            if "min_new_tokens" in kwargs:
                kwargs.pop("min_new_tokens")
            if "decoding_method" in kwargs:
                kwargs.pop("decoding_method")

            kwargs.pop("random_seed", None)

        else:
            raise ValueError(
                f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
            )
        return kwargs


@register_block("openai-chat", "vllm-remote-chat")
class OpenaiChatCompletionsLM(OpenaiCompletionsLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chat = True
        if self.batch_size is None:
            self._batch_size = 1
        if self.block_type == "openai-chat":
            sdg_logger.warn(f"OpenAI Chat models only support batch size of 1")
            self._batch_size = 1

    def _prepare_input(self, prompt: str):
        return {"role": "user", "content": prompt}

    def _extract_output(self, resp) -> str:
        return resp.message.content

    def generate_batch(self, *args: Any, **kwargs: Any) -> None:
        return super().generate_batch(*args, **kwargs)

    def loglikelihood_batch(self, *args, **kwargs):
        raise NotImplementedError("No support for logits.")
