# Copyright (c) 2023, Simone Tolomei

""" Time-Driven Manipulation

This script implements a time-driven manipulation task for the Franka Panda robot in Isaac Gym. 
The robot learns to perform a movement task with a cube. The cube is spawned at a random
location on the table and the robot has to pick it up and place it at a target location.
The training script doesn't enforce any pick-and-place behavior, but the robot learns it
by itself. The task is considered successful if the cube is within a specific distance from
the target location at the end of the episode.

To run this script, execute the following command from the root of the repository:
 /isaac-sim/python.sh omniisaacgymenvst/scripts/rlgames_train.py task=FactoryCube
"""


import hydra
import omegaconf
import os
import torch
import numpy as np

from omniisaacgymenvs.tasks.factory.factory_cube_env import FactoryCube
from omniisaacgymenvs.tasks.factory.cube_mobile_ws_env import CubeMobileWS
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
import omniisaacgymenvs.tasks.factory.factory_control_mobile as fc
from omni.isaac.core.utils.torch.maths import torch_rand_float

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.transformations import *
import omni.isaac.core.utils.torch as torch_utils

# FOR VIDEO REC:
#from omni.kit.viewport.utility.camera_state import ViewportCameraState
#from omni.kit.viewport.utility import get_viewport_from_window_name
#from pxr import Sdf

class CubeMobileTask(CubeMobileWS, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        super().__init__(name, sim_config, env)
        self._get_task_yaml_params()
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.max_dof_vel = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        self.max_linvel = torch.zeros((self.num_envs, 1), device=self.device)   
        self.cube_max_linvel = torch.zeros((self.num_envs, 1), device=self.device)   
        
        
        
    
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  

        ppo_path = 'train/FactoryTaskNutBoltPickPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting
        self.level = 0.0
        self.freezed_reward = torch.zeros((self.num_envs,1), device=self.device)
        self.episode_sums = {
            "dist_from_goal": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "cube_vel_kickstart": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "pos_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "rot_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "torque_saturated": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "action_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "singularity_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            ###da aggiungere arte di reward per la base
        }
        self.decimation = self._task_cfg["env"]["decimation"]
        # default joint velocities
        self.named_default_joint_velocities = self._task_cfg["env"]["defaultJointVelocities"]

        # other
        #self.sim_dt = self._task_cfg["sim"]["dt"]
        #self.real_control_frequency_inv = self._task_cfg["env"]["realControlFrequencyInv"]
        #self.control_dt = self.real_control_frequency_inv * self.sim_dt
        #self.policy_dt = self.decimation * self.control_dt
        self.Kp_pos = self._task_cfg["env"]["control"]["stiffnessPos"]
        self.Kd_pos = self._task_cfg["env"]["control"]["dampingPos"]
        self.Kp_vel = self._task_cfg["env"]["control"]["stiffnessVel"]
        self.Kd_vel = self._task_cfg["env"]["control"]["dampingVel"]
        self.base_threshold = 0.2
        #self.knee_threshold = 0.1

        # normalization
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.dof_acc_scale = self._task_cfg["env"]["learn"]["dofAccelerationScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale_pos = self._task_cfg["env"]["control"]["actionScalePos"]
        self.action_scale_vel = self._task_cfg["env"]["control"]["actionScaleVel"]


        #from anymal
        self._omniwheels_radius = self._task_cfg["env"]["omniwheels"]["radius"]
        #command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]


    def post_reset(self):
        """
        This method is called only one time right before sim begins. 
        The 'reset' here is referring to the reset of the world, which occurs before the sim starts.
        """

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()
        # if self._cfg.enable_recording: 
        #     self._update_camera()
            
        # super().post_reset()
        self.acquire_base_tensors()
        self._acquire_task_tensors()
        self.init_curriculum()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # step curriculum every 
        self.reset_idx(indices)
    
    def init_curriculum(self):
        self.goal_noise_curriculum_scale = 1.0
        self.goal_noise_curriculum_scale_step = self.cfg_task.randomize.goal_noise_curriculum_scale_step
        self.goal_noise_curriculum_scale_max = self.cfg_task.randomize.goal_noise_curriculum_scale_max
        self.goal_noise_curriculum_interval = self.cfg_task.randomize.goal_noise_curriculum_interval

    def step_curriculum(self, env_ids):
        if self.cfg_task.randomize.goal_noise_curriculum:
            # scale every goal_noise_curriculum_interval episodes
            if self.level % self.goal_noise_curriculum_interval == 0:
                if self.goal_noise_curriculum_scale < self.goal_noise_curriculum_scale_max:
                    self.goal_noise_curriculum_scale += self.goal_noise_curriculum_scale_step

    def _acquire_task_tensors(self):
        """Acquire tensors."""       
        # Grasp pose tensors
        self.cube_grasp_pos_local = torch.tensor([0.0, 0.0, 0.01], device=self._device).repeat((self._num_envs, 1))
        self.cube_grasp_quat_local = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device).repeat((self._num_envs, 1))
        self.goal_cube_pos = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height + 0.01 ], device=self.device).repeat(self.num_envs,1)
        self.cube_pos_initial = torch.tensor([0.0, 0.0, 0.0,], device=self.device).repeat(self.num_envs,1)
        self.initial_dist = torch.norm(self.goal_cube_pos - self.cube_pos_initial, dim=1)

        self.min_r = self.cfg_task.randomize.cube_pos_min_r
        r_w = 0.9 # workspace radius
        #ensure that half of the goals are inside the workspace and half outside
        self.max_r = np.sqrt(2*r_w**2 - self.min_r**2) 
        print("max_r: ", self.max_r)

    
    
    def compute_wheels_velocities(self): #after getting the 4 llocal wheels positions with respect to the base frame, a matrix of positions is created 
        self.wheels_local_pos[:] = (self.wheels_pos.view(self.num_envs, 4, 3) - self.base_pos.unsqueeze(1).repeat((1, 4, 1))).abs()

        self.wheels_mat[:, 2, 0] = - (self.wheels_local_pos[:, 0, 0] + self.wheels_local_pos[:, 0, 1]) / self._omniwheels_radius
        self.wheels_mat[:, 2, 1] = - (self.wheels_local_pos[:, 1, 0] + self.wheels_local_pos[:, 1, 1]) / self._omniwheels_radius
        self.wheels_mat[:, 2, 2] = (self.wheels_local_pos[:, 2, 0] + self.wheels_local_pos[:, 2, 1]) / self._omniwheels_radius
        self.wheels_mat[:, 2, 3] = (self.wheels_local_pos[:, 3, 0] + self.wheels_local_pos[:, 3, 1]) / self._omniwheels_radius

        # for i, ((LF_x, LF_y, _), (LH_x, LH_y, _), (RF_x, RF_y, _), (RH_x, RH_y, _)) in enumerate(self.wheels_local_pos):
        #     self.wheels_vel[i] = self.commands[i, :3] @ torch.tensor([[1., 1., 1., 1.],
        #                                                                     [-1., 1., 1., -1.],
        #                                                                     [-(LF_x + LF_y), -(LH_x + LH_y), (RF_x + RF_y), (RH_x + RH_y)]],
        #                                                              dtype=torch.float, device=self.device) / self._omniwheels_radius

        self.wheels_vel[:] = self.commands[:, None, :3].bmm(self.wheels_mat).squeeze()
    
    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        # Every 10 steps apply a force to the cube
        if self.progress_buf[0] % 10 == 0:
            is_cube_lifted = torch.where(self.cube_pos[:, 2] > self.cfg_base.env.table_height + 0.02,\
                                      torch.ones_like(self.cube_pos[:, 2]), torch.zeros_like(self.cube_pos[:, 2]))
            is_cube_grasped = torch.where(torch.norm(self.cube_pos - self.fingertip_midpoint_pos, dim=1) < 0.02,\
                                        torch.ones_like(self.cube_pos[:, 2]), torch.zeros_like(self.cube_pos[:, 2]))
            random_force =  (torch.rand((self.num_envs,3), dtype=torch.float32, device=self.device) * 2.0 - 1.0) @ \
                            torch.diag(torch.tensor(self.cfg_task.randomize.cube_force_max, device=self.device))
            self._cube.apply_forces(random_force, is_cube_lifted * is_cube_grasped)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        self.previous_actions = self.actions.clone().to(self.device)
        #this is the step when wheels velocity targets are genereted 
        self.compute_wheels_velocities()
        for i in range(self.decimation):
            if self.world.is_playing():
                self._apply_actions_as_ctrl_targets(
                    actions=self.actions,
                    ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                    do_scale=True
                )

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self.step_curriculum(env_ids)
        
        self._reset_object(env_ids)
        self._reset_task()
        SimulationContext.step(self.world, render=False)
        self.refresh_base_tensors()
        self.refresh_env_tensors() #n
        self._refresh_task_tensors()
        self._reset_mobile_franka(env_ids)
        self._replace_cube(env_ids)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self._randomize_gripper_pose(sim_steps=5)  #10 for cube30, otherwise 5

        # Reset buffers and metrics
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
            )
            self.episode_sums[key][env_ids] = 0.0

        self._reset_buffers(env_ids)
    

    def _reset_task(self):
        """
        Resets the task by initializing the maximum episode length and randomizing it.
        """
        self.max_episode_length = self.cfg_task.rl.max_episode_length
        # episode_lenght_noise = torch.randint(
        #     - self.cfg_task.randomize.ep_lenght_noise,
        #       self.cfg_task.randomize.ep_lenght_noise,
        #     size=(1,1),dtype=torch.int32, device=self.device).item()
        # self.max_episode_length += episode_lenght_noise


    def _reset_mobile_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        indices = env_ids.to(dtype=torch.int32)

        self.dof_pos[env_ids] = torch.cat(
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
             torch.tensot(self.cfg_task.randomize.base_initial_dof_pos, device = self.device),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        self.frankas_mobile.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas_mobile.set_joint_velocities(self.dof_vel[env_ids], indices=indices)
        SimulationContext.step(self.world, render=False)

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero


        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Now compute dof pos to grasp the cube
        gripper_initial_grasp_quat = self.cube_grasp_quat[env_ids,:].clone().to(device=self.device)
        
        target_p = self.cube_grasp_pos.clone().to(device=self.device)
        target_p[:,2] += 0.04  #5    # 0.02
        # target_p[:,2] += self.cfg_task.randomize.fingertip_midpoint_pos_noise[2]/2.0 + 0.02

        # move gripper to grasp pose with CLIK
        self.set_gripper_to(target_p, gripper_initial_grasp_quat, sim_steps=10)
        # self.refresh_base_tensors()
        # self.refresh_env_tensors()
        # self._refresh_task_tensors()


    def _reset_object(self, env_ids):
        """Reset root states of cube."""
        indices = env_ids.to(dtype=torch.int32)

        
        cube_noise_rot = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        cube_noise_rot = cube_noise_rot @ torch.diag(
            torch.tensor(self.cfg_task.randomize.cube_rot_noise, device=self.device))
        
        self.cube_quat[env_ids, :] = torch_utils.quat_from_euler_xyz(
            cube_noise_rot[:, 0],
            cube_noise_rot[:, 1],
            cube_noise_rot[:, 2]
        )



        
        init_mx_r = self.cfg_task.randomize.cube_pos_initial_max_r
        #randomize reeeeeeaaally close to the center. We want only to make the policy robust
        cube_init_r = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) * \
            (init_mx_r - 0.98*init_mx_r) + init_mx_r*0.98
        cube_init_theta = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) * 0.1 - 0.05

        cube_init_xy  = cube_init_r * torch.cat((torch.cos(cube_init_theta), torch.sin(cube_init_theta)), dim=1)

        self.cube_pos[env_ids, 0]   = - cube_init_xy[env_ids,0]
        self.cube_pos[env_ids, 1]   = cube_init_xy[env_ids,1]
        self.cube_pos[env_ids, 2]   = self.cfg_base.env.table_height + 0.005 
        self.cube_pos_initial = self.cube_pos.clone().to(device=self.device)

        self.cube_linvel[env_ids, :] = 0.0
        self.cube_angvel[env_ids, :] = 0.0

        
        # cube_mass = torch.rand((self.num_envs), dtype=torch.float32, device=self.device) * 0.019 + 0.001
        # cube_mass = torch.rand((self.num_envs), dtype=torch.float32, device=self.device) \
        #             * (self.cfg_task.randomize.cube_mass_max - self.cfg_task.randomize.cube_mass_min) \
        #             + self.cfg_task.randomize.cube_mass_min

        # Set to cube view
        self._cube.set_world_poses(self.cube_pos[env_ids] + self.env_pos[env_ids], self.cube_quat[env_ids], indices)
        self._cube.set_velocities(torch.cat((self.cube_linvel[env_ids], self.cube_angvel[env_ids]), dim=1), indices)

        # self._cube.set_masses(cube_mass, indices)
          
        # Randomize goal position
        goal_min_r = self.min_r * 1.2
        r_w = 0.9 # workspace radius

        if self.cfg_base.env.spawn_obstacle:
            goal_min_r = 0.9
        
        goal_r = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) * (self.max_r - goal_min_r) + goal_min_r
        # goal_r = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) * (r_w - goal_min_r) + goal_min_r
        # goal_r = self.max_r*torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)*1.2
        goal_theta = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) *  1.0 - 0.5
        goal_noise_xy = goal_r * torch.cat((torch.cos(goal_theta), torch.sin(goal_theta)), dim=1)        
        self.goal_cube_pos[env_ids, 0] = - goal_noise_xy[env_ids, 0]
        self.goal_cube_pos[env_ids, 1] =  goal_noise_xy[env_ids, 1]
    
        self.initial_dist = torch.norm(self.goal_cube_pos - self.cube_pos_initial, dim=1)
        
        # set sphere color to green where goal_r is less than r_w
        # red = torch.tensor([1.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        # green = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        # sphere_color = torch.where(goal_r < r_w, green, red)
        # self.sphere_material
        
        self._sphere.set_world_poses(self.goal_cube_pos[env_ids] + self.env_pos[env_ids], self.cube_grasp_quat_local[env_ids], indices)
      
    def _replace_cube(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        self._cube.set_world_poses(self.cube_pos[env_ids] + self.env_pos[env_ids], self.cube_quat[env_ids], indices)
        self._cube.set_velocities(torch.cat((self.cube_linvel[env_ids], self.cube_angvel[env_ids]), dim=1), indices)



    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.max_linvel[env_ids] = 0.0
        self.cube_max_linvel[env_ids] = 0.0

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """
        Applies actions from policy as position/rotation targets.

        Args:
            actions (torch.Tensor): The actions to apply as position/rotation targets.
            ctrl_target_gripper_dof_pos (torch.Tensor): The gripper DOF positions to apply.
            do_scale (bool): Whether to scale the actions.

        Returns:
            None
        """
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions


        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)


        if self.cfg_task.ctrl.control_gripper:
            # Retrieve gripper DOF from actions and apply:
            gripper_actions = actions[:,6:]
            if do_scale:
                gripper_actions = gripper_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.gripper_action_scale, device=self.device))

            self.ctrl_target_gripper_dof_pos = gripper_actions
        
        ## USED TUNING CT GAINS:
        # dir = (self.goal_cube_pos + self.cube_grasp_pos ) / 2.0
        # self.ctrl_target_fingertip_midpoint_pos = dir
        # self.ctrl_target_fingertip_midpoint_quat = self.cube_grasp_quat.clone().to(device=self.device)

        # if self.test:
        #     self._gripper_cyl.set_world_poses(self.ctrl_target_fingertip_midpoint_pos + self.env_pos,
        #                              self.ctrl_target_fingertip_midpoint_quat,
        #                             torch.arange(self._num_envs, dtype=torch.int64, device=self._device))
            
                        

        self.generate_ctrl_signals()


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self.world.is_playing():
            # Refresh data
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            # Compute observations and reward
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            # Log data
            self.get_extras()

        self.last_actions[:] = self.actions[:]
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of cube grasping frame
        self.cube_grasp_quat, self.cube_grasp_pos = tf_combine(
            self.cube_quat,
            self.cube_pos,
            self.cube_grasp_quat_local,
            self.cube_grasp_pos_local,
        )
        



    def get_observations(self):
        """Compute observations."""
        # Shallow copies of tensors
        rem_time = torch.unsqueeze(self.max_episode_length - self.progress_buf,1)
        d_to_goal = (self.goal_cube_pos - self.cube_pos)
        d_to_cube = (self.fingertip_midpoint_pos - self.cube_grasp_pos)
        ep_length_tensor = torch.tensor(self.max_episode_length, device=self.device).repeat(self.num_envs,1)
        
        franka_dof_pos = self.dof_pos[:, 0:7]
        franka_dof_vel = self.dof_vel[:, 0:7]
        # normalized_fingertip_midpoint_quat =
        # normalized_fingertip_midpoint_linvel =
        # normalized_fingertip_midpoint_angvel = 
        # normalized_cube_grasp_pos =
        # normalized_cube_grasp_quat 
        
        obs_tensors = [ #franka_dof_pos, 3+4+3+1+3+4+1+3+1+8
                        # franka_dof_vel, la somma deve venire 37
                        self.fingertip_midpoint_pos,
                        self.fingertip_midpoint_quat,
                        self.fingertip_midpoint_linvel,
                        self.fingertip_midpoint_angvel,
                        self.cube_grasp_pos,
                        self.cube_grasp_quat,
                        rem_time,
                        d_to_goal,
                        ep_length_tensor, 
                        self.last_actions]

    

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  

        observations = {
            self.frankas_mobile.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def calculate_metrics(self) -> None:
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()


    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
    

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        # *** ACTION PENALTY ***       
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        self.rew_buf[:] = - action_penalty * self.cfg_task.rl.action_penalty_scale / self.max_episode_length 
        self.episode_sums['action_penalty'] -= action_penalty * self.cfg_task.rl.action_penalty_scale / self.max_episode_length        

        # *** SINGULARITY PENALTY ***
        # penalty as i get close to singularities
        singularity_proximity = torch.linalg.cond(self.fingertip_midpoint_jacobian, 2)
        sing_penalty = -singularity_proximity * self.cfg_task.rl.singularity_penalty_scale
        self.rew_buf[:] += sing_penalty / self.max_episode_length
        self.episode_sums['singularity_penalty'] += sing_penalty / self.max_episode_length

        # *** DISTANCE REWARD ***
        # Normalize dist_penalty with respect to initial distance
        dist_penalty = torch.norm(self.goal_cube_pos - self.cube_pos,dim=1) / self.initial_dist
        dist_reward = (1.0 / (1.0 + self.cfg_task.rl.dist_reward_width_scale * dist_penalty**2))**2

        # Update maximum lin velocity of gripper and cube
        self.max_linvel = torch.max(torch.norm(self.fingertip_midpoint_linvel, dim=1), self.max_linvel)
        self.cube_max_linvel = torch.max(torch.norm(self.cube_linvel, dim=1), self.cube_max_linvel)
        
        # max_torque = torch.max(self.dof_torque, dim=1).values
        # max_torque = max_torque[:,0:7]
        torque_limits = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=self.device) 
        torque_saturation = torch.where(self.dof_torque[:,0:7] >= torque_limits, torch.ones_like(self.dof_torque[:,0:7]), torch.zeros_like(self.dof_torque[:,0:7]))
        torque_saturation = torch.sum(torque_saturation, dim=1)
        self.episode_sums['torque_saturated'] += torque_saturation / self.max_episode_length

        # Check if episode is ending and apply sparse reward
        is_ending = (self.progress_buf[0] >= self.max_episode_length - self.cfg_task.rl.ending_length)
        if is_ending:
            self.rew_buf[:] += dist_reward * self.cfg_task.rl.goal_scale
            self.episode_sums['dist_from_goal'] += dist_reward * self.cfg_task.rl.goal_scale 
            # box_success = self._check_in_box(self.cube_pos, self.goal_cube_pos, torch.tensor([0.02, 0.02, 0.02], device=self.device))
            # self.rew_buf[:] += box_success * 500.0

        if (self.level <= self.cfg_task.rl.kickstart_max_level):
            self.freezed_reward = torch.norm(self.cube_linvel, dim=1)*200.0*(self.cfg_task.rl.kickstart_max_level-self.level)/self.max_episode_length
            # cap at a maximum of 200
            self.freezed_reward = torch.where(self.freezed_reward > self.cfg_task.rl.kickstart_reward_max, self.cfg_task.rl.kickstart_reward_max*torch.ones_like(self.freezed_reward), self.freezed_reward)            
            self.freezed_reward *= self.cfg_task.rl.kickstart_reward_scale

        self.rew_buf[:] += self.freezed_reward
        self.episode_sums['cube_vel_kickstart'] += self.freezed_reward
        track_err = torch.norm(self.ctrl_target_fingertip_midpoint_pos - self.fingertip_midpoint_pos, dim=1)
        rot_track_err = torch.norm(self.ctrl_target_fingertip_midpoint_quat - self.fingertip_midpoint_quat, dim=1)
        self.episode_sums['pos_tracking_error'] += track_err
        self.episode_sums['rot_tracking_error'] += rot_track_err
        
        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            # Check if cube is picked up and above table
            self.rew_buf[:] = torch.where(dist_penalty < 0.08, self.rew_buf[:] + 2000 * torch.ones_like(self.rew_buf), self.rew_buf)
            # lift_success = self._check_lift_success(height_multiple=1.0)
            #self.level += torch.mean(lift_success.float())
            self.level += 1
            self.extras['final_mean_dists'] = torch.mean(dist_penalty.float())
            # log non-normalized dist penalty
            self.extras['abs_dist'] = torch.mean(torch.norm(self.goal_cube_pos - self.cube_pos, dim=1).float())
            self.extras['max_linvel'] = torch.mean(self.max_linvel.float())
            self.extras['cube_max_linvel'] = torch.mean(self.cube_max_linvel.float())
            # self.extras['cube_lifted'] = torch.mean(lift_success.float())
            self.extras['level'] = self.level
            # if self.test:
            #     print("Mean final dists: ", torch.mean(dist_penalty.float()))


    def _check_lift_success(self, height_multiple):
        """Check if cube is above table by more than specified multiple times height of cube."""

        lift_success = torch.where(
            self.cube_pos[:, 2] > self.cfg_base.env.table_height + height_multiple * 0.02,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success
    

    def _randomize_gripper_pose(self, sim_steps=5):
        """
        Adds a random offset to the current gripper pose.

        Args:
            sim_steps (int): Number of simulation steps to perform.

        Returns:
            None
        """
        # print("randomizing gripper")
        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos.clone().to(device=self.device)

        fingertip_midpoint_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
            device=self.device)
        )
        # set z of all envs from 0 to 1
        #fingertip_midpoint_pos_noise[:, 2] = torch.abs(fingertip_midpoint_pos_noise[:, 2])
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_quat = self.fingertip_midpoint_quat.clone().to(device=self.device)

        ctrl_target_fingertip_midpoint_euler = torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat)
        # it's a tuple (roll, pitch, yaw) of torch tensor of shapes (n,1). It has to be converted to a single tensor
        ctrl_target_fingertip_midpoint_euler = torch.stack(ctrl_target_fingertip_midpoint_euler, dim=1)
        
        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )
        # Step sim and render
        for i in range(sim_steps):
            # print("randomizing gripper step ", i)
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            SimulationContext.step(self.world, render=False)

        self.dof_vel[:, :] = torch.zeros_like(self.dof_vel[:])
        idx = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.frankas_mobile.set_joint_velocities(self.dof_vel[:], indices=idx)

        # step once to update physx with the newly set joint velocities
        # print("gripper randomization done")
        SimulationContext.step(self.world, render=False)



    def set_gripper_to(self, target_gripper_pose, target_quat, sim_steps=100):
        """
        Moves the gripper to a specific pose using CLIK (Closed-Loop Inverse Kinematics).

        Args:
            target_gripper_pose (torch.Tensor): The desired position of the gripper in the world frame, as a 3D tensor.
            target_quat (torch.Tensor): The desired orientation of the gripper in the world frame, as a quaternion tensor.
            sim_steps (int, optional): The number of simulation steps to perform. Defaults to 100.

        Returns:
            None
        """
        ctrl_target_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)

        for _ in range(sim_steps):
            # Refresh data
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            # compute pose error
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos = target_gripper_pose,
                ctrl_target_fingertip_midpoint_quat = target_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Use dfferential IK to compute delta dof
            delta_arm_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method='pinv',
                jacobian=self.fingertip_midpoint_jacobian,
                device=self.device,
            )
            # Compute new DOFs
            ctrl_target_dof_pos[:, 0:7] = self.dof_pos[:,0:7] + delta_arm_dof_pos
            ctrl_target_dof_pos[:, 7:9] = self.asset_info_franka_table.franka_gripper_width_max  # gripper finger joints
            # set to robots and render
            self.frankas_mobile.set_joint_positions(positions=ctrl_target_dof_pos)
            self.frankas_mobile.set_joint_position_targets(positions=ctrl_target_dof_pos)
            self.frankas_mobile.set_joint_velocities(torch.zeros_like(self.dof_pos, device=self.device))

            SimulationContext.step(self.world, render=False)
            
    def _check_in_box(self, cube_pos, box_pos, box_size):
        """Check if cube is inside box using weighted infinity norm."""
        # Calculate absolute difference between cube and box positions
        dist = torch.abs(cube_pos - box_pos)
        dist = torch.max(dist - box_size, torch.zeros_like(dist))
        dist.clamp_(min=0)
        # Sum elements of dist along the last dimension
        dist = dist.sum(dim=-1)
        is_inside = torch.where(dist == 0, torch.ones_like(dist), torch.zeros_like(dist))
        return is_inside
    
    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        self.view_port = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.view_port.set_active_camera(self.perspective_path)


    def _update_camera(self):
        base_pos = self.base_pos[self._selected_id, :].clone()
        base_quat = self.base_quat[self._selected_id, :].clone()

        camera_local_transform = torch.tensor([-1.8, 0.0, 0.6], device=self.device)
        camera_pos = quat_apply(base_quat, camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.view_port)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item()+0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)