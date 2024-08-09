# Standard
from typing import Any, Dict, List, Optional, Union

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import register_block


@register_block("fuzzy_dedup")
class FuzzyDedupPostprocessing(BaseBlock):
    """Base Class for all Postprocessors"""

    def __init__(
        self,
        input_folder_path: str,
        intermediate_folder_path: str,
        output_folder_path: str,
        num_permutations: int = 64,
        threshold: float = 0.8,
        shingles_size: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_folder_path = input_folder_path
        self.intermediate_folder_path = intermediate_folder_path
        self.output_folder_path = output_folder_path
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.shingles_size = shingles_size

    def doc_id(self, input_folder: str, output_folder: str, input_params: dict):
        # Standard
        import sys

        # Third Party
        from data_processing.utils import ParamsUtils
        from data_processing_ray.runtime.ray import RayTransformLauncher
        from doc_id_transform_ray import DocIDRayTransformConfiguration

        local_conf = {
            "input_folder": input_folder,
            "output_folder": output_folder,
        }
        worker_options = {"num_cpus": 0.8}
        params = input_params | {
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            "run_locally": True,
            "runtime_worker_options": ParamsUtils.convert_to_ast(worker_options),
            "runtime_num_workers": 3,
            # doc id configuration
            "doc_id_doc_column": "output",
            "doc_id_hash_column": "hash_column",
            "doc_id_int_column": "int_id_column",
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = RayTransformLauncher(DocIDRayTransformConfiguration())
        # launch
        launcher.launch()

    def fdedup(self, input_folder: str, output_folder: str, input_params: dict):
        # Standard
        import sys

        # Third Party
        from data_processing.utils import ParamsUtils
        from data_processing_ray.runtime.ray import RayTransformLauncher
        from fdedup_transform_ray import FdedupRayTransformConfiguration

        local_conf = {
            "input_folder": input_folder,
            "output_folder": output_folder,
        }
        worker_options = {"num_cpus": 0.8}
        params = input_params | {
            # where to run
            "run_locally": True,
            # Data access. Only required parameters are specified
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            # Orchestration parameters
            "runtime_worker_options": ParamsUtils.convert_to_ast({"num_cpus": 0.8}),
            "runtime_num_workers": 1,
            "runtime_creation_delay": 0,
            # columns used
            "fdedup_doc_column": "output",
            "fdedup_id_column": "int_id_column",
            "fdedup_cluster_column": "cluster",
            # infrastructure
            "fdedup_bucket_cpu": 0.5,
            "fdedup_doc_cpu": 0.5,
            "fdedup_mhash_cpu": 0.5,
            "fdedup_num_doc_actors": 1,
            "fdedup_num_bucket_actors": 1,
            "fdedup_num_minhash_actors": 1,
            "fdedup_num_preprocessors": 2,
            # fuzzy parameters
            "fdedup_num_permutations": self.num_permutations,
            "fdedup_threshold": self.threshold,
            "fdedup_shingles_size": self.shingles_size,
            "fdedup_delimiters": " ",
            # Random delay between reads
            "fdedup_random_delay_limit": 5,
            # snapshotting
            "fdedup_snapshot_delay": 1,
            "fdedup_use_doc_snapshot": False,
            "fdedup_use_bucket_snapshot": False,
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = RayTransformLauncher(FdedupRayTransformConfiguration())
        # launch
        launcher.launch()

    def fdedup_embeddable(
        self,
        input_folder_path: str = "./output/",
        runtime_code_location: str = "{'github': 'github', 'commit_hash': '12345', 'path': 'path'}",
        runtime_pipeline_id: str = "pipeline_id",
        runtime_job_id: str = "job_id",
    ):
        args = locals()
        args.pop("input_folder_path", "")
        args.pop("self")
        doc_id_output_folder = self.intermediate_folder_path + "/doc_id"
        fdedup_output_folder = self.output_folder_path
        doc_id_task = self.doc_id(
            input_folder=input_folder_path,
            output_folder=doc_id_output_folder,
            input_params=args,
        )
        fdedup_task = self.fdedup(
            input_folder=doc_id_output_folder,
            output_folder=fdedup_output_folder,
            input_params=args,
        )

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ):

        # Standard
        import shutil

        self._postprocess()
        shutil.rmtree(self.intermediate_folder_path)

        return self.output_folder_path

    def _postprocess(self) -> bool:
        self.fdedup_embeddable(input_folder_path=self.input_folder_path)
        return True
