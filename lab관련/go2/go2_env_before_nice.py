# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# import gymnasium as gym
# import torch


# from pxr import Gf
# from isaacsim.core.utils.prims import get_prim_at_path
# from isaacsim.core.prims import XFormPrim

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sensors import ContactSensor, RayCaster

# from ..kinematics.solver import go2_ik, go2_fk, HIP_OFFSETS

# from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


# class Go2Env(DirectRLEnv):
#     cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

#     def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         # action space -> joint order 확인
#         print("="*50)
#         print("Detected Joint Order:")
#         for i, name in enumerate(self._robot.joint_names):
#             print(f"  Action Index {i}: {name}")
#         print("="*50)

#         # Joint position command (deviation from default joint positions)
#         self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
#         self._previous_actions = torch.zeros(
#             self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
#         )

#         # X/Y linear velocity and yaw angular velocity commands
#         self._commands = torch.zeros(self.num_envs, 3, device=self.device)

#         # Logging
#         self._episode_sums = {
#             key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#             for key in [
#                 "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
#                 "dof_torques_l2", "dof_acc_l2", "action_rate_l2", "feet_air_time",
#                 "undesired_contacts", "flat_orientation_l2", "base_height", "torques",
#                 "stop_penalty_lin", "stop_penalty_ang",
#             ]
#         }
#         # Get specific body indices
#         self._base_id, _ = self._contact_sensor.find_bodies("base")
#         self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
#         self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")

#         # FK 함수로 default_foot_positions 계산
#         default_joint_pos = self._robot.data.default_joint_pos
#         default_foot_pos_flat = go2_fk(default_joint_pos, HIP_OFFSETS.to(self.device))
#         self._default_foot_positions = default_foot_pos_flat.reshape(self.num_envs, 4, 3)
        
#         # self._default_foot_positions = torch.tensor([[[ 0.1936,  0.1701, -0.2735],
#         #                                              [ 0.1935, -0.1700, -0.2735],
#         #                                              [-0.2267,  0.1706, -0.2673],
#         #                                              [-0.2267, -0.1705, -0.2673]]], device='cuda:0')
        
#         # print("defaut_foot_positions: ", self._default_foot_positions[0])
        
#         # single_env_positions = torch.tensor([[[ 0.1936,  0.1701, -0.2735],
#         #                               [ 0.1935, -0.1700, -0.2735],
#         #                               [-0.2267,  0.1706, -0.2673],
#         #                               [-0.2267, -0.1705, -0.2673]]], device=self.device)

#         # 2. repeat() 함수를 사용해 num_envs 개수만큼 복제합니다.
#         #    (num_envs, 1, 1) -> 첫 번째 차원은 num_envs배, 나머지 차원은 그대로 유지
#         # self._default_foot_positions = single_env_positions.repeat(self.num_envs, 1, 1)

#         # print("defaut_foot_positions shape: ", self._default_foot_positions.shape)
                
        

#         # 발끝 좌표 normalization
#         self.scales = torch.tensor([0.3, 0.3, 0.14], device=self.device)


#         self._foot_target_markers = {}
#         colors = [
#             Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0),
#             Gf.Vec3f(0.0, 0.0, 1.0), Gf.Vec3f(1.0, 1.0, 0.0)
#         ]
#         foot_names = ["FL", "FR", "RL", "RR"]
#         env_ns = self.scene.env_ns.replace("/*", "/env_0")

#         for i, name in enumerate(foot_names):
#             sphere_prim_path = f"{env_ns}/foot_target_{name}"
#             material_cfg = sim_utils.PreviewSurfaceCfg()
#             sphere_cfg = sim_utils.SphereCfg(radius=0.05, visual_material=material_cfg)

#             marker_usd_prim = sim_utils.spawn_sphere(
#                 prim_path=sphere_prim_path,
#                 cfg=sphere_cfg
#             )

#             shader_prim = get_prim_at_path(f"{sphere_prim_path}/material/Shader")
#             if shader_prim:
#                 color_attr = shader_prim.GetAttribute("inputs:diffuse_color")
#                 color_attr.Set(colors[i])

#             self._foot_target_markers[name] = XFormPrim(
#                 sphere_prim_path,
#                 name=f"foot_target_marker_{name}"
#             )

#     def _setup_scene(self):
#         self._robot = Articulation(self.cfg.robot)
#         self.scene.articulations["robot"] = self._robot
#         self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
#         self.scene.sensors["contact_sensor"] = self._contact_sensor
#         if isinstance(self.cfg, Go2RoughEnvCfg):
#             self._height_scanner = RayCaster(self.cfg.height_scanner)
#             self.scene.sensors["height_scanner"] = self._height_scanner
#         self.cfg.terrain.num_envs = self.scene.cfg.num_envs
#         self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
#         self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
#         # we need to explicitly filter collisions for CPU simulation
#         if self.device == "cpu":
#             self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)
        
        
#         from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
        
#         frame_transformer_cfg = FrameTransformerCfg(
#             prim_path="/World/envs/env_.*/Robot/base",
#             target_frames=[
#                 FrameTransformerCfg.FrameCfg(
#                     name="FL_foot", prim_path="/World/envs/env_.*/Robot/FL_foot"
#                 ),
#                 FrameTransformerCfg.FrameCfg(
#                     name="FR_foot", prim_path="/World/envs/env_.*/Robot/FR_foot"
#                 ),
#                 FrameTransformerCfg.FrameCfg(
#                     name="RL_foot", prim_path="/World/envs/env_.*/Robot/RL_foot"
#                 ),
#                 FrameTransformerCfg.FrameCfg(
#                     name="RR_foot", prim_path="/World/envs/env_.*/Robot/RR_foot"
#                 ),
#             ],
#             debug_vis=True,
#         )

#         self.robot_transforms = frame_transformer_cfg.class_type(cfg=frame_transformer_cfg)
#         self.scene.sensors["robot_transforms"] = self.robot_transforms


#     def _pre_physics_step(self, actions: torch.Tensor):
#         self._actions = actions.clone()
#         actions_reshaped = self._actions.reshape(self.num_envs, 4, 3)
#         normalized_actions_reshaped = torch.clip(actions_reshaped, -0.5, 0.5) * self.scales * 0.0
#         target_foot_positions = self._default_foot_positions + normalized_actions_reshaped + 0.0
#         print("target_foot_positions: ", target_foot_positions[0])
#         target_joint_angles = go2_ik(target_foot_positions, HIP_OFFSETS.to(self.device))

#         from isaacsim.core.utils.torch import tf_apply

#         target_foot_pos_env0 = target_foot_positions[0]
#         robot_pose_w = self._robot.data.root_pose_w[0]
#         robot_pos = robot_pose_w[0:3]
#         robot_quat = robot_pose_w[3:7]

#         target_foot_pos_world = tf_apply(
#             robot_quat.float(),
#             robot_pos.float(),
#             target_foot_pos_env0.float()
#         )

#         foot_names = ["FL", "FR", "RL", "RR"]
#         for i, name in enumerate(foot_names):
#             marker = self._foot_target_markers.get(name)
#             if marker:
#                 marker.set_world_poses(positions=target_foot_pos_world[i].unsqueeze(0))
        
#         #### fk, ik 검증 ##################################################################################
        
#         print("-------------------------------")
#         print(self.robot_transforms)
#         print("relative transforms:", self.robot_transforms.data.target_pos_source)

#         self._processed_actions = target_joint_angles

#     def _apply_action(self):
#         self._robot.set_joint_position_target(self._processed_actions)

#     def _get_observations(self) -> dict:
#         self._previous_actions = self._actions.clone()
#         height_data = None

#         if isinstance(self.cfg, Go2RoughEnvCfg):
#             height_data = (
#                 self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
#             ).clip(-1.0, 1.0)
#         obs = torch.cat(
#             [
#                 tensor
#                 for tensor in (
#                     self._robot.data.root_lin_vel_b,
#                     self._robot.data.root_ang_vel_b,
#                     self._robot.data.projected_gravity_b,
#                     self._robot.data.joint_pos - self._robot.data.default_joint_pos,
#                     self._robot.data.joint_vel,                    
#                     self._commands,
#                     height_data,
#                     self._actions,
#                 )
#                 if tensor is not None
#             ],
#             dim=-1,
#         )
#         observations = {"policy": obs}
#         return observations

#     def _get_rewards(self) -> torch.Tensor:
#         # linear velocity tracking
#         lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
#         lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
#         # yaw rate tracking
#         yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
#         yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
#         # stop error
#         lin_vel_norm_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2]), dim=1)
#         stop_penalty_lin = torch.exp(-2.0 * lin_vel_norm_sq)
#         ang_vel_norm_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
#         stop_penalty_ang = torch.exp(-2.0 * ang_vel_norm_sq)
#         # z velocity tracking
#         z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
#         # angular velocity x/y
#         ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
#         # joint torques
#         joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
#         # joint acceleration
#         joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
#         # action rate
#         action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
#         # feet air time
#         first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
#         last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
#         air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
#             torch.norm(self._commands[:, :2], dim=1) > 0.1
#         )
#         # undesired contacts
#         net_contact_forces = self._contact_sensor.data.net_forces_w_history
#         is_contact = (
#             torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
#         )
#         contacts = torch.sum(is_contact, dim=1)
#         # flat orientation
#         flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
#         # base height
#         base_height = torch.mean(self._robot.data.root_link_pose_w[:, 2].unsqueeze(1) , dim=1)
#         rew_base_height = torch.square(base_height - 0.3)
#         # torque
#         rew_torque = torch.sum(self._robot.data.applied_torque, dim=1)

#         rewards = {
#             "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
#             "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
#             "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
#             "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
#             "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
#             "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
#             "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
#             "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
#             "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
#             "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
#             "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
#             "torques": rew_torque * self.cfg.torque_reward_scale * self.step_dt,
#             "stop_penalty_lin": stop_penalty_lin * self.cfg.stop_penalty_reward_scale * self.step_dt,
#             "stop_penalty_ang": stop_penalty_ang * self.cfg.stop_penalty_reward_scale * self.step_dt,
#         }
#         reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
#         # Logging
#         for key, value in rewards.items():
#             self._episode_sums[key] += value
#         return reward

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         time_out = self.episode_length_buf >= self.max_episode_length - 1
#         net_contact_forces = self._contact_sensor.data.net_forces_w_history
#         died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
#         return died, time_out

#     def _reset_idx(self, env_ids: torch.Tensor | None):
#         if env_ids is None or len(env_ids) == self.num_envs:
#             env_ids = self._robot._ALL_INDICES
#         self._robot.reset(env_ids)
#         super()._reset_idx(env_ids)
#         if len(env_ids) == self.num_envs:
#             self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
#         self._actions[env_ids] = 0.0
#         self._previous_actions[env_ids] = 0.0

#         num_resets = len(env_ids)
#         lin_x_range = [-1.0, 1.0]
#         rand_x = (torch.rand(num_resets, 1, device=self.device) * (lin_x_range[1] - lin_x_range[0])) + lin_x_range[0]
#         lin_y_range = [-1.0, 1.0]
#         rand_y = (torch.rand(num_resets, 1, device=self.device) * (lin_y_range[1] - lin_y_range[0])) + lin_y_range[0]
#         ang_vel_range = [-1.0, 1.0]
#         rand_yaw = (torch.rand(num_resets, 1, device=self.device) * (ang_vel_range[1] - ang_vel_range[0])) + ang_vel_range[0]
#         self._commands[env_ids] = torch.cat([rand_x, rand_y, rand_yaw], dim=1)

#         # Reset robot state
#         joint_pos = self._robot.data.default_joint_pos[env_ids]
#         joint_vel = self._robot.data.default_joint_vel[env_ids]
#         default_root_state = self._robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self._terrain.env_origins[env_ids]
#         # print("joint_pos", joint_pos)
#         # print("joint_vel", joint_vel)
#         self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#         self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
#         # Logging
#         extras = dict()
#         for key in self._episode_sums.keys():
#             episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
#             extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
#             self._episode_sums[key][env_ids] = 0.0
#         self.extras["log"] = dict()
#         self.extras["log"].update(extras)
#         extras = dict()
#         extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
#         extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
#         self.extras["log"].update(extras)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from pxr import Gf
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.prims import XFormPrim
# -----------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from ..kinematics.solver import go2_ik, HIP_OFFSETS, go2_fk_new, go2_ik_new

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action space -> joint order 확인
        print("="*50)
        print("Detected Joint Order:")
        for i, name in enumerate(self._robot.joint_names):
            print(f"  Action Index {i}: {name}")
        print("="*50)

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
                "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
                "dof_torques_l2", "dof_acc_l2", "action_rate_l2", "feet_air_time",
                "undesired_contacts", "flat_orientation_l2", "base_height", "torques",
                "stop_penalty_lin", "stop_penalty_ang",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")

        default_joint_pos = self._robot.data.default_joint_pos
        # print("default_joint_pos: ", default_joint_pos[0])
        default_foot_pos_flat = go2_fk_new(default_joint_pos, HIP_OFFSETS.to(self.device))
        self._default_foot_positions = default_foot_pos_flat.reshape(self.num_envs, 4, 3) 
        # self._default_foot_positions = torch.tensor([[ 0.2024,  0.1701, -0.2733],
        # [ 0.2023, -0.1701, -0.2733],
        # [-0.2182,  0.1707, -0.2630],
        # [-0.2182, -0.1706, -0.2630]], device='cuda:0')
            
        self.scales = torch.tensor([0.3, 0.3, 0.15], device=self.device)

        self._foot_target_markers = {}
        target_marker_colors = [
            Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), # Red, Green
            Gf.Vec3f(0.0, 0.0, 1.0), Gf.Vec3f(1.0, 1.0, 0.0)  # Blue, Yellow
        ]
        foot_names = ["FL", "FR", "RL", "RR"]
        env_ns = self.scene.env_ns.replace("/*", "/env_0")

        for i, name in enumerate(foot_names):
            sphere_prim_path = f"{env_ns}/foot_target_{name}"
            material_cfg = sim_utils.PreviewSurfaceCfg()
            sphere_cfg = sim_utils.SphereCfg(radius=0.05, visual_material=material_cfg)
            marker_usd_prim = sim_utils.spawn_sphere(prim_path=sphere_prim_path, cfg=sphere_cfg)

            shader_prim = get_prim_at_path(f"{sphere_prim_path}/material/Shader")
            if shader_prim:
                color_attr = shader_prim.GetAttribute("inputs:diffuse_color")
                color_attr.Set(target_marker_colors[i])

            self._foot_target_markers[name] = XFormPrim(sphere_prim_path, name=f"foot_target_marker_{name}")

        self._actual_foot_markers = {}
        actual_marker_color = Gf.Vec3f(0.0, 1.0, 0.0) # Green

        for i, name in enumerate(foot_names):
            sphere_prim_path = f"{env_ns}/actual_foot_{name}"
            material_cfg = sim_utils.PreviewSurfaceCfg()
            sphere_cfg = sim_utils.SphereCfg(radius=0.03, visual_material=material_cfg) # 크기를 약간 작게 설정
            marker_usd_prim = sim_utils.spawn_sphere(prim_path=sphere_prim_path, cfg=sphere_cfg)

            shader_prim = get_prim_at_path(f"{sphere_prim_path}/material/Shader")
            if shader_prim:
                color_attr = shader_prim.GetAttribute("inputs:diffuse_color")
                color_attr.Set(actual_marker_color)

            self._actual_foot_markers[name] = XFormPrim(sphere_prim_path, name=f"actual_foot_marker_{name}")


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
        
        from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
        
        frame_transformer_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(name="FR_foot", prim_path="/World/envs/env_.*/Robot/FR_foot"),
                FrameTransformerCfg.FrameCfg(name="FL_foot", prim_path="/World/envs/env_.*/Robot/FL_foot"),
                FrameTransformerCfg.FrameCfg(name="RL_foot", prim_path="/World/envs/env_.*/Robot/RL_foot"),
                FrameTransformerCfg.FrameCfg(name="RR_foot", prim_path="/World/envs/env_.*/Robot/RR_foot"),
            ],
            debug_vis=False,
        )
        self.robot_transforms = frame_transformer_cfg.class_type(cfg=frame_transformer_cfg)
        self.scene.sensors["robot_transforms"] = self.robot_transforms

        hip_transformer_cfg = FrameTransformerCfg(
                prim_path="/World/envs/env_.*/Robot/base", # 소스 프레임: 로봇 base
                target_frames=[
                    FrameTransformerCfg.FrameCfg(name="FL_thigh", prim_path="/World/envs/env_.*/Robot/FL_thigh"),
                    FrameTransformerCfg.FrameCfg(name="FR_thigh", prim_path="/World/envs/env_.*/Robot/FR_thigh"),
                    FrameTransformerCfg.FrameCfg(name="RL_thigh", prim_path="/World/envs/env_.*/Robot/RL_thigh"),
                    FrameTransformerCfg.FrameCfg(name="RR_thigh", prim_path="/World/envs/env_.*/Robot/RR_thigh"),
                ]
            )
            # self.hip_transforms 라는 이름으로 인스턴스 생성 및 씬에 추가
        self.hip_transforms = hip_transformer_cfg.class_type(cfg=hip_transformer_cfg)
        self.scene.sensors["hip_transforms"] = self.hip_transforms



    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        actions_reshaped = self._actions.reshape(self.num_envs, 4, 3)
        normalized_actions_reshaped = torch.clip(actions_reshaped, -0.3, 0.3) * self.scales
        
        # print("normalized_actions_reshaped : ", normalized_actions_reshaped)
        
        joint_pos = self._robot.data.joint_pos
        print("default_joint_pos: ", joint_pos[0])
        default_foot_pos_flat = go2_fk_new(joint_pos, HIP_OFFSETS.to(self.device))
        _default_foot_positions = default_foot_pos_flat.reshape(self.num_envs, 4, 3) 
        
        
        
        
        
        # target_foot_positions = self._default_foot_positions + normalized_actions_reshaped * 0.0
        target_foot_positions = _default_foot_positions + actions_reshaped * 0.0

        # target_foot_positions = self.robot_transforms.data.target_pos_source

        # gt_position = torch.tensor([[[ 0.1936,  0.1701, -0.2735],
        #                            [ 0.1935, -0.1700, -0.2735],
        #                            [-0.2267,  0.1706, -0.2673],
        #                           [-0.2267, -0.1705, -0.2673]]], device=self.device)

        # target_foot_positions = gt_position.repeat(self.num_envs, 1, 1)

        
        # print("target_foot_positions: ", target_foot_positions[0])
        
        # target_foot_positions = self.robot_transforms.data.target_pos_source
        
        target_joint_angles = go2_ik_new(target_foot_positions, HIP_OFFSETS.to(self.device))
        
        # target_joint_angles = go2_ik(target_foot_positions, HIP_OFFSETS.to(self.device))

        from isaacsim.core.utils.torch import tf_apply

        robot_pose_w = self._robot.data.root_pose_w[0]
        robot_pos = robot_pose_w[0:3]
        robot_quat = robot_pose_w[3:7]
        foot_names = ["FL", "FR", "RL", "RR"]

        # target_foot_pos_local_env0 = target_foot_positions[0]
        # target_foot_pos_world = tf_apply(
        #     robot_quat.float(), robot_pos.float(), target_foot_pos_local_env0.float()
        # )
        # for i, name in enumerate(foot_names):
        #     marker = self._foot_target_markers.get(name)
        #     if marker:
        #         marker.set_world_poses(positions=target_foot_pos_world[i].unsqueeze(0))
        
        # actual_foot_pos_local_env0 = self.robot_transforms.data.target_pos_source[0]
        
        # # print("size of actual_foot_pos_local_env0: ", actual_foot_pos_local_env0.shape)
        
        # # print("actual_foot_pos_local_env0: ", actual_foot_pos_local_env0)
        
        # actual_foot_pos_world = tf_apply(
        #     robot_quat.float(), robot_pos.float(), actual_foot_pos_local_env0.float()
        # )
        # for i, name in enumerate(foot_names):
        #     marker = self._actual_foot_markers.get(name)
        #     if marker:
        #         marker.set_world_poses(positions=actual_foot_pos_world[i].unsqueeze(0))


        # hip_positions_in_base_frame = self.hip_transforms.data.target_pos_source
                
        # # 콘솔 출력이 너무 많아지는 것을 방지하기 위해 20스텝마다 한 번씩 출력
        # if self.common_step_counter % 20 == 0:
        #     print("="*50)
        #     print("Hip Positions relative to Base (from FrameTransformer):")
        #     # 0번 환경의 결과 출력
        #     print(hip_positions_in_base_frame[0])
        #     print("="*50)
  
        self._processed_actions = 0.0 * self._actions + self._robot.data.default_joint_pos


        # self._processed_actions = target_joint_angles


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
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        
        # print("gravity_vector :", self._robot.data.projected_gravity_b )

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
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        num_resets = len(env_ids)
        lin_x_range = [-3.0, 3.0]
        rand_x = (torch.rand(num_resets, 1, device=self.device) * (lin_x_range[1] - lin_x_range[0])) + lin_x_range[0]
        lin_y_range = [-2.0, 2.0]
        rand_y = (torch.rand(num_resets, 1, device=self.device) * (lin_y_range[1] - lin_y_range[0])) + lin_y_range[0]
        ang_vel_range = [-2.0, 2.0]
        rand_yaw = (torch.rand(num_resets, 1, device=self.device) * (ang_vel_range[1] - ang_vel_range[0])) + ang_vel_range[0]
        self._commands[env_ids] = torch.cat([rand_x, rand_y, rand_yaw], dim=1)

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