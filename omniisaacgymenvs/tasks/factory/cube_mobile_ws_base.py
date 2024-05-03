


import carb
import hydra
import math
import numpy as np
import torch

from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.objects import FixedCylinder
from omni.isaac.core.objects import GroundPlane

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.factory_franka_mobile import FactoryFrankaMobile


from pxr import PhysxSchema, UsdPhysics
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_schema_class_base import FactoryABCBase

from omniisaacgymenvs.tasks.factory.factory_schema_config_base import (
    FactorySchemaConfigBase,
)

# cuRobo
# from curobo.types.base import TensorDeviceType
# from curobo.types.math import Pose
# from curobo.types.robot import RobotConfig
# from curobo.util_file import get_robot_configs_path, join_path, load_yaml
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

class CubeMobileBase(RLTask, FactoryABCBase):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize instance variables. Initialize RLTask superclass."""

        # Set instance variables from base YAML
        self._get_base_yaml_params()
        self._env_spacing = self.cfg_base.env.env_spacing

        # Set instance variables from task and train YAMLs
        self._sim_config = sim_config
        self._cfg = sim_config.config  # CL args, task config, and train config
        self._task_cfg = sim_config.task_config  # just task config
        self._num_envs = sim_config.task_config["env"]["numEnvs"]
        self._num_observations = sim_config.task_config["env"]["numObservations"]
        self._num_actions = sim_config.task_config["env"]["numActions"]
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        


        self.USE_CUROBO = False

        # if self.USE_CUROBO:
        #     self.device = TensorDeviceType.CUDA
        #     self.robot_config = RobotConfig(
        #         join_path(get_robot_configs_path(), "franka_panda.yaml")
        #     )
        #     self.ik_solver_config = IKSolverConfig(
        #         join_path(get_robot_configs_path(), "franka_panda_ik.yaml")
        #     )
        #     self.ik_solver = IKSolver(self.robot_config, self.ik_solver_config)

        #     tensor_args = TensorDeviceType()

        #     config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        #     urdf_file = config_file["robot_cfg"]["kinematics"][
        #         "urdf_path"
        #     ]  # Send global path starting with "/"
        #     base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        #     ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        #     robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

        #     ik_config = IKSolverConfig.load_from_robot_config(
        #         robot_cfg,
        #         None,
        #         rotation_threshold=0.05,
        #         position_threshold=0.005,
        #         num_seeds=20,
        #         self_collision_check=False,
        #         self_collision_opt=False,
        #         tensor_args=tensor_args,
        #         use_cuda_graph=True,
        #     )
        #     ik_solver = IKSolver(ik_config)
        super().__init__(name, env)

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        config_path = (
            "task/CubeMobileBase.yaml"  # relative to Gym's Hydra search path (cfg dir)
        )
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/cube_asset_info.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table[""][""][""][
            "tasks"
        ]["factory"][
            "yaml"
        ]  # strip superfluous nesting

    def import_mobile_franka_assets(self, add_to_stage=True):
        """Set Franka and table asset options. Import assets."""

        self._stage = get_current_stage()

        if add_to_stage:
            franka_translation = np.array([self.cfg_base.env.franka_depth, 0.0, 0.0])
            franka_orientation = np.array([0.0, 0.0, 0.0, 1.0])

            franka_mobile = FactoryFrankaMobile(
                prim_path=self.default_zero_env_path + "/franka_mobile",
                name="franka_mobile",
                translation=franka_translation,
                orientation=franka_orientation,
            )
            self._sim_config.apply_articulation_settings(
                "franka_mobile",
                get_prim_at_path(franka_mobile.prim_path),
                self._sim_config.parse_actor_config("franka_mobile"),
            )

            for link_prim in franka_mobile.prim.GetChildren():
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(
                        self._stage, link_prim.GetPrimPath()
                    )
                    rb.GetDisableGravityAttr().Set(True)
                    rb.GetRetainAccelerationsAttr().Set(False)
                    if self.cfg_base.sim.add_damping:
                        rb.GetLinearDampingAttr().Set(
                            1.0
                        )  # default = 0.0; increased to improve stability
                        rb.GetMaxLinearVelocityAttr().Set(
                            1.0
                        )  # default = 1000.0; reduced to prevent CUDA errors
                        rb.GetAngularDampingAttr().Set(
                            5.0
                        )  # default = 0.5; increased to improve stability
                        rb.GetMaxAngularVelocityAttr().Set(
                            2 / math.pi * 180
                        )  # default = 64.0; reduced to prevent CUDA errors
                    else:
                        rb.GetLinearDampingAttr().Set(0.0)
                        rb.GetMaxLinearVelocityAttr().Set(1000.0)
                        rb.GetAngularDampingAttr().Set(0.5)
                        rb.GetMaxAngularVelocityAttr().Set(64 / math.pi * 180)

            table_translation = np.array(  #qui serve per allontanare il tavolo dal franka, se diminuisco la x si allontana ancora
                [-0.2 - 0.65 , 0.0, self.cfg_base.env.table_height * 0.5]
            )
            table_orientation = np.array([1.0, 0.0, 0.0, 0.0])

            # table = FixedCylinder(
            #     prim_path=self.default_zero_env_path + "/table",
            #     name="table",
            #     translation=table_translation,
            #     orientation=table_orientation,
            #     radius=self.asset_info_franka_table.table_radius,
            #     height=self.cfg_base.env.table_height,
            #     color=np.array([0.3, 0.3, 0.3]),
            # )

            table = FixedCuboid(
                prim_path=self.default_zero_env_path + "/table",
                name="table",
                translation=table_translation,
                orientation=table_orientation,
                scale=np.array(
                    [
                        1.3,
                        1.3,
                        self.cfg_base.env.table_height
                    ]),
                color=np.array([0.1, 0.1, 0.1]),
            )
            plane = GroundPlane(prim_path="/World/GroundPlane", z_position=0)

        self.parse_controller_spec(add_to_stage=add_to_stage)

    def acquire_base_tensors(self):
        """Acquire tensors."""
        #sum of joints, 9 for frankas, 4 for wheels
        self.num_dofs_franka = 9
        self.num_dof_base = 4
        self.env_pos = self._env_pos

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs_franka), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs_franka), device=self.device)
        self.dof_torque = torch.zeros(
            (self.num_envs, self.num_dofs_franka), device=self.device
        )
        self.fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.ctrl_target_dof_pos = torch.zeros(
            (self.num_envs, self.num_dofs_franka), device=self.device
        )
        self.ctrl_target_gripper_dof_pos = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self.ctrl_target_fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        
        #PRESO DA RUOTE PER ANYMAL
        self.commands = torch.zeros((self.num_envs, self.num_dof_base), dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
       

        self.wheels_vel = torch.zeros(
            (self.num_envs, self.num_dof_base), dtype=torch.float, device=self.device, requires_grad=False) #velocità lineare ruote (in che frame??)

        self.wheels_acc = torch.zeros(
            (self.num_envs,  self.num_dof_base), dtype=torch.float, device=self.device, requires_grad=False)
        #sono 4 perchè 4 ruote e 3 perchè contiene le posizioni
        self.wheels_mat = torch.ones(
            (self.num_envs, 3, self.num_dof_base), dtype=torch.float, device=self.device, requires_grad=False) / self._omniwheels_radius
        self.wheels_mat[:, 1, [0, 3]] *= -1.
        self.wheels_local_pos = torch.zeros((self.num_envs, 4, 3), 
                                            dtype=torch.float, device=self.device)
        self.wheels_pos = torch.zeros((self.num_envs * self.num_dof_base, 3), dtype=torch.float, device=self.device)
        self.wheels_quat = torch.zeros((self.num_envs * self.num_dof_base, 4), dtype=torch.float, device=self.device)
        self.wheels_local_pos = torch.zeros((self.num_envs, self.num_dof_base, 3), dtype=torch.float, device=self.device)


    def refresh_base_tensors(self):
        """Refresh tensors."""

        if not self.world.is_playing():
            return

        #preso da anymal
        self.last_dof_vel[:] = self.dof_vel[:]
        self.dof_pos[:] = self.frankas_mobile.get_joint_positions(clone=False)
        self.dof_vel[:] = self.frankas_mobile.get_joint_velocities(clone=False)
        self.wheels_acc[:] = (self.dof_vel[:, 12:16] - self.last_dof_vel[:, 12:16]) / self.control_dt


        # Jacobian shape: [4, 11, 6, 9] (root has no Jacobian)
        self.franka_jacobian = self.frankas_mobile.get_jacobians()
        self.franka_mass_matrix = self.frankas_mobile.get_mass_matrices(clone=False)

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_mass_matrix = self.franka_mass_matrix[
            :, 0:7, 0:7
        ]  # for Franka arm (not gripper)

        # self.franka_coriolis_forces = self.frankas.get_coriolis_and_centrifugal_forces(clone=False)
        # self.franka_gravity_torque = self.frankas.get_generalized_gravity_forces(clone=False)
        
        self.arm_coriolis_forces = torch.zeros((self.num_envs, 7), device=self.device)

        self.hand_pos, self.hand_quat = self.frankas_mobile._hands.get_world_poses(clone=False)
        self.hand_pos -= self.env_pos
        hand_velocities = self.frankas_mobile._hands.get_velocities(clone=False)
        self.hand_linvel = hand_velocities[:, 0:3]
        self.hand_angvel = hand_velocities[:, 3:6]

        (
            self.left_finger_pos,
            self.left_finger_quat,
        ) = self.frankas_mobile._lfingers.get_world_poses(clone=False)
        self.left_finger_pos -= self.env_pos
        left_finger_velocities = self.frankas_mobile._lfingers.get_velocities(clone=False)
        self.left_finger_linvel = left_finger_velocities[:, 0:3]
        self.left_finger_angvel = left_finger_velocities[:, 3:6]
        self.left_finger_jacobian = self.franka_jacobian[:, 8, 0:6, 0:7]
        left_finger_forces = self.frankas_mobile._lfingers.get_net_contact_forces(clone=False)
        self.left_finger_force = left_finger_forces[:, 0:3]

        (
            self.right_finger_pos,
            self.right_finger_quat,
        ) = self.frankas_mobile._rfingers.get_world_poses(clone=False)
        self.right_finger_pos -= self.env_pos
        right_finger_velocities = self.frankas_mobile._rfingers.get_velocities(clone=False)
        self.right_finger_linvel = right_finger_velocities[:, 0:3]
        self.right_finger_angvel = right_finger_velocities[:, 3:6]
        self.right_finger_jacobian = self.franka_jacobian[:, 9, 0:6, 0:7]
        right_finger_forces = self.frankas_mobile._rfingers.get_net_contact_forces(clone=False)
        self.right_finger_force = right_finger_forces[:, 0:3]

        self.gripper_dof_pos = self.dof_pos[:, 7:9]

        (
            self.fingertip_centered_pos,
            self.fingertip_centered_quat,
        ) = self.frankas_mobile._fingertip_centered.get_world_poses(clone=False)
        self.fingertip_centered_pos -= self.env_pos
        fingertip_centered_velocities = self.frankas_mobile._fingertip_centered.get_velocities(
            clone=False
        )
        self.fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        self.fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]
        self.fingertip_centered_jacobian = self.franka_jacobian[:, 10, 0:6, 0:7]

        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) / 2
        self.fingertip_midpoint_pos = fc.translate_along_local_z(
            pos=self.finger_midpoint_pos,
            quat=self.hand_quat,
            offset=self.asset_info_franka_table.franka_finger_length,
            device=self.device,
        )
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal

        # TODO: Add relative velocity term (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(
            self.fingertip_centered_angvel,
            (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
            dim=1,
        )

        # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal

        self.fingertip_midpoint_jacobian = (
            self.left_finger_jacobian + self.right_finger_jacobian
        ) * 0.5

    def parse_controller_spec(self, add_to_stage):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {
            "num_envs",
            "jacobian_type",
            "gripper_prop_gains",
            "gripper_deriv_gains",
            "motor_ctrl_mode",
            "gain_space",
            "ik_method",
            "joint_prop_gains",
            "joint_deriv_gains",
            "do_motion_ctrl",
            "task_prop_gains",
            "task_deriv_gains",
            "do_inertial_comp",
            "motion_ctrl_axes",
            "do_force_ctrl",
            "force_ctrl_method",
            "wrench_prop_gains",
            "force_ctrl_axes",
            "use_computed_torque"
        }
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl["num_envs"] = self.num_envs
        self.cfg_ctrl["jacobian_type"] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_deriv_gains, device=self.device
        ).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == "gym_default":
            self.cfg_ctrl["motor_ctrl_mode"] = "gym"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.gripper_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.gripper_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "joint_space_ik":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = False
        elif ctrl_type == "joint_space_id": 
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
            self.cfg_ctrl["use_computed_torque"] = True
        elif ctrl_type == "task_space_impedance":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = False
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = False
        elif ctrl_type == "operational_space_motion":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = False
        elif ctrl_type == "open_loop_force":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = False
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "open"
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.open_loop_force.force_ctrl_axes, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "closed_loop_force":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = False
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "closed"
            self.cfg_ctrl["wrench_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "hybrid_force_motion":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "closed"
            self.cfg_ctrl["wrench_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))

        if add_to_stage:
            if self.cfg_ctrl["motor_ctrl_mode"] == "gym":
                for i in range(7):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_link{i}/panda_joint{i+1}"
                    )
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                    drive.GetStiffnessAttr().Set(
                        self.cfg_ctrl["joint_prop_gains"][0, i].item() * np.pi / 180
                    )
                    drive.GetDampingAttr().Set(
                        self.cfg_ctrl["joint_deriv_gains"][0, i].item() * np.pi / 180
                    )

                for i in range(2):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_hand/panda_finger_joint{i+1}"
                    )
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
                    drive.GetStiffnessAttr().Set(
                        self.cfg_ctrl["gripper_deriv_gains"][0, i].item()
                    )
                    drive.GetDampingAttr().Set(
                        self.cfg_ctrl["gripper_deriv_gains"][0, i].item()
                    )

            elif self.cfg_ctrl["motor_ctrl_mode"] == "manual":
                for i in range(7):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_link{i}/panda_joint{i+1}"
                    )
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

                for i in range(2):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_hand/panda_finger_joint{i+1}"
                    )
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets or DOF torques."""

        # Get desired Jacobian
        if self.cfg_ctrl["jacobian_type"] == "geometric":
            self.fingertip_midpoint_jacobian_tf = self.fingertip_midpoint_jacobian
        elif self.cfg_ctrl["jacobian_type"] == "analytic":
            self.fingertip_midpoint_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_quat,
                fingertip_jacobian=self.fingertip_midpoint_jacobian,
                num_envs=self.num_envs,
                device=self.device,
            )

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl["motor_ctrl_mode"] == "gym":
            self._set_dof_pos_target()
        elif self.cfg_ctrl["motor_ctrl_mode"] == "manual":
            self._set_dof_torque()

    def _set_dof_pos_target(self):
        """Set Franka DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device,
        )

        self.frankas_mobile.set_joint_position_targets(positions=self.ctrl_target_dof_pos)

    def _set_dof_torque(self):
        """Set Franka DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            arm_coriolis_forces=self.arm_coriolis_forces,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device,
            dt = self._task_cfg["sim"]["dt"],
        )

        self.frankas_mobile.set_joint_efforts(efforts=self.dof_torque)

    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        gravity = [0.0, 0.0, -gravity_mag]
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    