# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema



class FactoryFrankaMobile(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "franka_mobile",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = "/home/darko/isaac_mobile_manipulation/OmniIsaacGymEnvs/assets/darko_base/darko_base_factory_franka.usd"
        print(self._usd_path)
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation
        
        # if self._usd_path is None:
        #     assets_root_path = get_assets_root_path()
        #     if assets_root_path is None:
        #         carb.log_error("Could not find Isaac Sim assets folder")
        #     self._usd_path = assets_root_path + "/Isaac/Robots/FactoryFranka/factory_franka.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "darkoBASEONLY/factory_franka_instanceable/darko_panda_link0/panda_joint1",
            "darkoBASEONLY/factory_franka_instanceable/panda_link1/panda_joint2",
            "darkoBASEONLY/factory_franka_instanceable/panda_link2/panda_joint3",
            "darkoBASEONLY/factory_franka_instanceable/panda_link3/panda_joint4",
            "darkoBASEONLY/factory_franka_instanceable/panda_link4/panda_joint5",
            "darkoBASEONLY/factory_franka_instanceable/panda_link5/panda_joint6",
            "darkoBASEONLY/factory_franka_instanceable/panda_link6/panda_joint7",
            "darkoBASEONLY/factory_franka_instanceable/panda_hand/panda_finger_joint1",
            "darkoBASEONLY/factory_franka_instanceable/panda_hand/panda_finger_joint2",
        ]

        drive_type = ["angular"] * 7 + ["linear"] * 2
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02]
        stiffness = [40 * np.pi / 180] * 7 + [500] * 2
        damping = [80 * np.pi / 180] * 7 + [20] * 2
        max_force = [87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]
        #max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.15, 0.15]
        # max_force *= 2
        # max_velocity *= 2

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )


        #base

        dof_paths = [#questi qui o devo chiamarli dalla base??
            "darkoBASEONLY/wheel_LH/omniwheel/wheel_LH_joint",
            "darkoBASEONLY/wheel_RH/omniwheel/wheel_RH_joint",
            "darkoBASEONLY/wheel_LF/omniwheel/wheel_LF_joint",
            "darkoBASEONLY/wheel_RF/omniwheel/wheel_RF_joint"
           
        ]

        drive_type = 4* ["angular"]
        default_dof_pos = [0.0] * 4
        stiffness = [0.0] * 4
        damping = [1000000.0] * 4
        max_force = [4800.0] * 4

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="velocity",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )
8