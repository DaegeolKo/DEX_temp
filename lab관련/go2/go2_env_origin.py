# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from ..kinematics.solver import go2_ik, go2_fk, HIP_OFFSETS

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # action space -> joint order 확인################################
        print("="*50)
        print("Detected Joint Order:")
        for i, name in enumerate(self._robot.joint_names):
            print(f"  Action Index {i}: {name}")
        print("="*50)
        ##################################################################

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "base_height",
                "torques",
                "stop_penalty_lin",
                "stop_penalty_ang",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        
        
        # 1. 원본 default_joint_pos (관절 그룹 순서)
        
        #######################################3
        # Detected Joint Order:
        # Action Index 0: FL_hip_joint
        # Action Index 1: FR_hip_joint
        # Action Index 2: RL_hip_joint
        # Action Index 3: RR_hip_joint
        # Action Index 4: FL_thigh_joint
        # Action Index 5: FR_thigh_joint
        # Action Index 6: RL_thigh_joint
        # Action Index 7: RR_thigh_joint
        # Action Index 8: FL_calf_joint
        # Action Index 9: FR_calf_joint
        # Action Index 10: RL_calf_joint
        # Action Index 11: RR_calf_joint
        #######################################3
        
        default_joint_pos = self._robot.data.default_joint_pos
        print(f"\n[1] Original default_joint_pos (grouped by joint type):\n"
              f"  - Shape: {default_joint_pos.shape}\n"
              f"  - Values (env 0): {default_joint_pos[0]}")

        # 2. FK 함수로 default_foot_positions 계산
        default_foot_pos_flat = go2_fk(default_joint_pos, HIP_OFFSETS.to(self.device))
        
        # go2 fk의 결과, 각 발 끝 좌표값 ############3##############################    
        # [0, 1, 2]	FL 
        # [3, 4, 5]	FR 
        # [6, 7, 8]	RL 
        # [9, 10, 11] RR 
        #####################################################################
        
        # default_foot_pos_flat = go2_fk(joint_pos_for_kinematics, HIP_OFFSETS.to(self.device))
        self._default_foot_positions = default_foot_pos_flat.reshape(self.num_envs, 4, 3)
        
        # _default_foot_positions 의 결과 #################################################
        
        # [ 
        #     [FL_x, FL_y, FL_z],  # 0번 행: 앞왼쪽(FL) 다리의 좌표
        #     [FR_x, FR_y, FR_z],  # 1번 행: 앞오른쪽(FR) 다리의 좌표
        #     [RL_x, RL_y, RL_z],  # 2번 행: 뒤왼쪽(RL) 다리의 좌표
        #     [RR_x, RR_y, RR_z]   # 3번 행: 뒤오른쪽(RR) 다리의 좌표
        # ]
        ####################################################################################
        
        # go2 configuration에 맞춘 발끝 좌표 normalization (generalization 용)
        self.scales = torch.tensor([0.3, 0.3, 0.14], device=self.device) 

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go2RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
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
        
        
    # joint level sampling ###############################################################
    
    # def _pre_physics_step(self, actions: torch.Tensor):
    #     self._actions = actions.clone()
    #     self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
    
    #######################################################################################

    # 3d space sampling ####################################################################
    def _pre_physics_step(self, actions: torch.Tensor):
        # RL 정책의 출력(displacement)을 복사하고 모양을 (N, 4, 3)으로 변경
        self._actions = actions.clone()
        
        actions_reshaped = self._actions.reshape(self.num_envs, 4, 3)
        
        # print("actions_reshaped :", actions_reshaped[0])  # 첫 번째 환경의 actions_reshaped 출력
        
        # 기본 발 위치에 displacement를 더하여 목표 발 위치 계산
        
        normalized_actions_reshaped = torch.clip(actions_reshaped, -0.5, 0.5) * self.scales
        
        target_foot_positions = self._default_foot_positions  + normalized_actions_reshaped
        
        
        
        # target_foot_positions의 경우에는 아래에 normalized_actions_reshaped를 더해준 값이므로 샘플링된 발끝 좌표값이라고 볼 수 있음.
        # [ 
        #     [FL_x, FL_y, FL_z],  # 0번 행: 앞왼쪽(FL) 다리의 좌표
        #     [FR_x, FR_y, FR_z],  # 1번 행: 앞오른쪽(FR) 다리의 좌표
        #     [RL_x, RL_y, RL_z],  # 2번 행: 뒤왼쪽(RL) 다리의 좌표
        #     [RR_x, RR_y, RR_z]   # 3번 행: 뒤오른쪽(RR) 다리의 좌표
        # ]
        
        # target_foot_positions = self._default_foot_positions
        # --- IK 계산 및 데이터 순서 역변환 ---
        target_joint_angles = go2_ik(target_foot_positions, HIP_OFFSETS.to(self.device))
        # 최종적으로 변환된 값을 시뮬레이터에 전달
        
        # ik 값 결과 : [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]

        self._processed_actions = target_joint_angles
    # ########################################################################################


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        
        # print("lin_vel_x :", self._robot.data.root_lin_vel_b[0, :1])
        # print("ang_vel_yaw :", self._robot.data.root_ang_vel_b[0, 2])
        
        # current_foot_pos_flat = go2_fk(self._robot.data.joint_pos, HIP_OFFSETS.to(self.device))
        # current_foot_positions = current_foot_pos_flat.reshape(self.num_envs, 4, 3)
        # # 기본 위치로부터의 변위를 관측값으로 사용하는 것이 더 유용할 수 있음
        # foot_pos_error = current_foot_positions - self._default_foot_positions
        # print("size of foot_pos_error :", foot_pos_error[0])  # 첫 번째 환경의 foot_pos_error 출력
        
        if isinstance(self.cfg, Go2RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b, #3
                    self._robot.data.root_ang_vel_b, #3
                    self._robot.data.projected_gravity_b, #3
                    self._commands, #3
                    # self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    # self._robot.data.joint_vel,
                    # foot_pos_error.reshape(self.num_envs, 12), #12
                    height_data, #0
                    self._actions, #12
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
        
        # 직접 구현 한 reward function : base height, torque #################################################################

        # base height
        
        # Penalize base height away from target
        # base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # rew_base_height = torch.square(base_height - self.cfg.rewards.base_height_target)
        
        base_height = torch.mean(self._robot.data.root_link_pose_w[:, 2].unsqueeze(1) , dim=1)
        # print("base_height :", base_height[0])
        rew_base_height = torch.square(base_height - 0.3)
        
        # torque
        
        rew_torque = torch.sum(self._robot.data.applied_torque, dim=1)
        ##########################################################################################################
        
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
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
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
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # 여기에서 command 처리할때 원래 크기는 envs, 3이라서 값이 이 샘플링을 통해 다 바뀌어버림.
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(1.0, 1.0)
        
        # 각각 random sampling 구현 ###########################################################################
        num_resets = len(env_ids)

        lin_x_range = [1.0, 1.0]
        rand_x = (torch.rand(num_resets, 1, device=self.device) * (lin_x_range[1] - lin_x_range[0])) + lin_x_range[0]

        lin_y_range = [0.0, 0.0]
        rand_y = (torch.rand(num_resets, 1, device=self.device) * (lin_y_range[1] - lin_y_range[0])) + lin_y_range[0]

        ang_vel_range = [0.0, 0.0]
        rand_yaw = (torch.rand(num_resets, 1, device=self.device) * (ang_vel_range[1] - ang_vel_range[0])) + ang_vel_range[0]

        self._commands[env_ids] = torch.cat([rand_x, rand_y, rand_yaw], dim=1)
        
        ######################################################################################################
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
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


