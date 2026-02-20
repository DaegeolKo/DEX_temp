# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from pxr import Gf
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from ..kinematics.FK_IK_Solver import Go2Solver

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.solver = Go2Solver(device = self.device)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._processed_actions = self._robot.data.default_joint_pos.clone()

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
                "dof_torques_l2", "dof_acc_l2", "action_rate_l2", "feet_air_time",
                "undesired_contacts", "flat_orientation_l2", "base_height", "torques",
                "stop_penalty_lin", "stop_penalty_ang", "dof_close_to_default"
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_(thigh|calf)")

        # Forward kinematics : default_joint_pos => default_foot_position 
        default_joint_pos = self._robot.data.default_joint_pos
        # default_joint_pos = self._robot.data.default_joint_pos * 0.0 + torch.tensor([[ 0.1000, -0.1000,  0.1050, -0.1050,  0.8909,  0.8909,  1.1620, 1.1620,
        #  -1.679, -1.679, -1.415, -1.415]], device='cuda:0')
        print("default :", default_joint_pos[0])
        default_foot_pos_flat = self.solver.go2_fk_new(joint_angles = default_joint_pos)
        self._default_foot_positions = default_foot_pos_flat.reshape(self.num_envs, 4, 3) 
        
        # scailing of sampling space(x,y,z)    
        self.scales = torch.tensor([0.3, 0.3, 0.1], device=self.device)
        
        # ik 발산 시, 발생한 index 확인용
        self.ik_failure_penalty = torch.zeros(self.num_envs, device=self.device)        

    def _setup_scene(self):
        
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        if isinstance(self.cfg, Go2RoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
            
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

    def _pre_physics_step(self, actions: torch.Tensor):
        
        self._actions = actions.clone()
        actions_reshaped = self._actions.reshape(self.num_envs, 4, 3)
        normalized_actions_reshaped = actions_reshaped * self.scales * self.cfg.action_scale
        
        target_foot_positions = self._default_foot_positions + normalized_actions_reshaped * 0.3
        
        target_joint_angles = self.solver.go2_ik_new(target_foot_positions = target_foot_positions)

        self.ik_failure_penalty[:] = 0.0
        
        # print(self._robot.data.joint_names)
        # print(self._robot.data.joint_pos_limits)  # (min, max) shape: (num_joints, 2)

        Agent_NAN = torch.isnan(target_joint_angles)
        
        if Agent_NAN.any():
            problem_env_indices = Agent_NAN.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            
            self.ik_failure_penalty[problem_env_indices] = -0.1  
            self.reset_buf[problem_env_indices] = 1             
    
            target_joint_angles[problem_env_indices] = self._processed_actions[problem_env_indices]
                        
        # joint space
        # self._processed_actions = self._actions * 0.0 + self._robot.data.default_joint_pos
        
        # target_joint_angles[:, 11] = 10.0
        
        # 3d space
        self._processed_actions = target_joint_angles


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)


    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None

        if isinstance(self.cfg, Go2RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, 
                    self._robot.data.joint_vel,                                  
                    self._commands,
                    height_data,
                    self._previous_actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations


    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # stop error
        lin_vel_norm_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        stop_penalty_lin = torch.exp(-2.0 * lin_vel_norm_sq)
        ang_vel_norm_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        stop_penalty_ang = torch.exp(-2.0 * ang_vel_norm_sq)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # base height
        base_height = torch.mean(self._robot.data.root_link_pose_w[:, 2].unsqueeze(1) , dim=1)
        rew_base_height = torch.square(base_height - 0.3)
        # torque
        rew_torque = torch.sum(self._robot.data.applied_torque, dim=1)
        rew_dof_close_to_default = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos),dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "torques": rew_torque * self.cfg.torque_reward_scale * self.step_dt,
            "stop_penalty_lin": stop_penalty_lin * self.cfg.stop_penalty_reward_scale * self.step_dt,
            "stop_penalty_ang": stop_penalty_ang * self.cfg.stop_penalty_reward_scale * self.step_dt,
            "dof_close_to_default" : rew_dof_close_to_default * self.cfg.dof_close_to_default_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        reward = reward + self.ik_failure_penalty
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        num_resets = len(env_ids)
        lin_x_range = [2.5, 2.5]
        rand_x = (torch.rand(num_resets, 1, device=self.device) * (lin_x_range[1] - lin_x_range[0])) + lin_x_range[0]
        lin_y_range = [0.0, 0.0]
        rand_y = (torch.rand(num_resets, 1, device=self.device) * (lin_y_range[1] - lin_y_range[0])) + lin_y_range[0]
        ang_vel_range = [0.0, 0.0]
        rand_yaw = (torch.rand(num_resets, 1, device=self.device) * (ang_vel_range[1] - ang_vel_range[0])) + ang_vel_range[0]
        self._commands[env_ids] = torch.cat([rand_x, rand_y, rand_yaw], dim=1)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._processed_actions[env_ids] = joint_pos

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
