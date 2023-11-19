# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from in2n4d.in2n4d_datamanager import InstructNeRF2NeRF4DDataManagerConfig, InstructNeRF2NeRF4DDataManager
from in2n4d.in2n4d import InstructNeRF2NeRF4DModelConfig
from in2n4d.in2n4d_pipeline import InstructNeRF2NeRF4DPipelineConfig
from in2n4d.in2n4d_trainer import InstructNeRF2NeRFTrainerConfig


in2n4d_method_extra_tiny = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n4d-extra-tiny",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRF4DPipelineConfig(
            datamanager=InstructNeRF2NeRF4DDataManagerConfig(
                dataparser=DycheckDataParserConfig(downscale_factor=2),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
            ),
            model=InstructNeRF2NeRF4DModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                depth_weight=0,
            ),
            ip2p_use_full_precision=False
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF 4D extra tiny method, uses LPIPs, IP2P at half precision",
)
