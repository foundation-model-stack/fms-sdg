# Standard
from typing import List
import json

# Third Party
import pytest

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.blocks.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation

single_intent_kwargs = {
    "intent_only": True,
    "min_ct": 1,
    "max_ct": 1,
}

multi_intent_kwargs = {
    "intent_only": True,
    "min_ct": 2,
    "max_ct": 2,
}

parallel_kwargs = {
    "check_arg_question_overlap": True,
    "intent_only": False,
    "min_ct": 2,
    "max_ct": 2,
}

parallel_nested_kwargs = {
    "check_arg_question_overlap": True,
    "intent_only": False,
    "require_nested": True,
    "min_ct": 2,
    "max_ct": 2,
}


def get_args(func_calls):
    return {
        k: v for k, v in TEST_APIS.items() if k in [fc["name"] for fc in func_calls]
    }

def run_single_intent(cval,desired_result=True):
    validator = APIGenSpecValidator(name="test_single_intent")

    # single intent
    func_calls = [{"name": "add"}]
    question = "add 3 with 4"
    api_info = get_args(func_calls)
    # api_info = get_args([{"name": "add"}])

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": cval,
            **single_intent_kwargs,
        }
    ]
    validator.generate(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(single_intent_kwargs.keys()),
        result_field="result",
    )
    print("TESTAPI1", test_instance)
    assert test_instance[0]["result"] == desired_result

def run_multi_intent(cval,desired_result=True):
    validator = APIGenSpecValidator(name="test_multi_intent")
    # multiple intent
    func_calls = [
        {"name": "add"},
        {"name": "add_event"},
    ]
    question = "add 3 with 4"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            # "c": json.dumps(func_calls),
            # "c": json.dumps([{"name": "add", "args": ()}, {"name": "add_event", "args": ()}, {"args": (), "name": "foo"}]),  # crashes
            # "c": json.dumps([{"name": "add", "arguments": ()}, {"name": "add_event", "arguments": ()}]),
            #"c": json.dumps([{'name': "add", "arguments": ()}, {"name": "add_event", "arguments": ()}, {"arguments": (),"name": "add"}]),
            "c": cval,
            **multi_intent_kwargs,
        }
    ]
    validator.generate(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(multi_intent_kwargs.keys()),
        result_field="result",
    )
    print("TESTMI", test_instance)
    assert test_instance[0]["result"] == desired_result

def run_parallel(kwargs,func_calls,desired_result=True):
        validator = APIGenSpecValidator(name="test_parallel_single")

        question = "add 33 with 4 then add 4 with 5"
        api_info = get_args(func_calls)

        test_instance = [
            {
                "a": api_info,
                "b": question,
                "c": json.dumps(func_calls),
                **kwargs,
            }
        ]
        validator.generate(
            test_instance,
            arg_fields=["a", "b", "c"],
            kwarg_fields=list(kwargs.keys()),
            result_field="result",
        )
        print("TESTPS1", test_instance)
        assert test_instance[0]["result"] == desired_result

class TestApiValidator:
    def test_single_intent_crash1(self):
        run_single_intent(json.dumps(["name"])) # crash

        # same issue
    def test_single_intent_crash2(self):
        run_single_intent(json.dumps([1])) # crash

    # def test_single_intent(self):
    #     run_single_intent(json.dumps(['add', 'add']), False) # correctly fails
    #     run_single_intent(json.dumps([{'add': 'foo'}]), False)  # correctly fails
    #
    # def test_multi_intent(self):
    #     run_multi_intent(json.dumps([{"name": "add", "arguments": ()},
    #                                  {"name": "add_event", "arguments": ()}]),
    #                      False) # correctly fails because intent only allows "name" field

    def test_parallel_crash1(self):
        run_parallel(parallel_kwargs,
                     [{"name": "add", "arguments": None},
                      {"name": "add", "arguments": []}])

    def test_parallel(self):
        # there is a test for dups, but it doesn't catch this. it may not matter.
        # also, not that "3" occurs in the question as "33"
        run_parallel(parallel_kwargs,
                     [{"name": "add", "arguments": {"n1": 3, "n2": 4}},
                      {"name": "add", "arguments": {"n2": 4, "n1": 3}}])

    def test_yes_no(self):
        validator = ApiGenSpecYesNoValidation(name="test_yes_no")

        # "Yes" is incorrectly rejected.
        for arg_inp in ["Yes"]:

            test_instance = [
                {
                    # "a": TEST_APIS,
                    "a": "this is ignored",
                    # "b": "this is a test question",
                    "b": 123, # ignored
                    "c": arg_inp,
                }
            ]
            validator.generate(
                test_instance,
                arg_fields=["a", "b", "c"],
                result_field="result",
            )
            # This incorrectly fails, since it doesn't convert the case of the input
            assert test_instance[0]["result"] == (arg_inp.upper() in ["YES", "NO"])


TEST_APIS = {
    "add": {
        "name": "add",
        "parameters": {
            "properties": {
                "n1": {"type": "number"},
                "n2": {"type": "number"},
            },
            "required": ["n1", "n2"],
        },
        "output_parameters": {"properties": {"result": {"type": "number"}}},
    },
    "add_event": {
        "name": "add_event",
        "parameters": {
            "properties": {"event": {"type": "string"}},
            "required": ["event"],
        },
        "output_parameters": {"properties": {"added_successfully": {"type": "bool"}}},
    },
}
