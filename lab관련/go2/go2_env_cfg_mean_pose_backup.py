# NOTE:
# One-time snapshot/backup of `go2_env_cfg.py` that overrides the robot init_state to a
# mean-pose IK solution (used for a z-offset-free NF decoder ablation).
# This file is not used by the registered Gym tasks unless you explicitly point an entry-point to it.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.3
    action_space = 12
    observation_space = 48
    state_space = 0

    # --- Normalizing-flow decoder (nsf12_stand) ---
    # PPO action (12D) -> (z or eps) -> NF decode -> foot positions (4x3) -> IK -> joint targets
    use_nf_decoder = True
    nf_decoder_ckpt_path = "/home/kdg/normalizing-flows/sota/nsf12_stand_gauss_fixed_actnorm_ik08_ep100.pt"
    # "eps": PPO output을 eps~N(0,1)로 보고 z = mu + sigma*eps 후 decode
    # "z"  : PPO output을 z로 보고 바로 decode
    nf_action_mode = "z"
    # z-mode 안정화: policy 출력(z_delta)을 그대로 쓰면 초기 z≈0에서 비정상 자세가 나올 수 있으므로
    # default_joint_pos의 FK로 만든 기본 발 위치를 만족하는 z_offset을 NF inverse로 계산해서 더해줌.
    # (즉, z = z_offset + z_delta)
    nf_z_offset_from_default_pose = False
    # eps/z를 너무 크게 보내면 IK가 쉽게 발산하므로 기본은 [-3,3] 클리핑
    # (0이면 클리핑 비활성화)
    nf_action_clip = 0
    nf_std_eps = 1e-6

    # --- Termination: base contact ---
    base_contact_termination = True
    base_contact_force_threshold = 1.0
    base_contact_termination_delay_s = 0.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=2**20,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot (mean-pose IK override)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            joint_pos={
                "FL_hip_joint": 0.483538,
                "FR_hip_joint": -0.483538,
                "RL_hip_joint": 0.488843,
                "RR_hip_joint": -0.488843,
                "FL_thigh_joint": 1.311795,
                "FR_thigh_joint": 1.311795,
                "RL_thigh_joint": 1.468597,
                "RR_thigh_joint": 1.468597,
                "FL_calf_joint": -2.479579,
                "FR_calf_joint": -2.479579,
                "RL_calf_joint": -2.275811,
                "RR_calf_joint": -2.275811,
            },
            joint_vel={".*": 0.0},
        ),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 4.0
    yaw_rate_reward_scale = 2.0
    z_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -0.01
    flat_orientation_reward_scale = -0.5
    base_height_reward_scale = -0.1
    torque_reward_scale = 0.00
    stop_penalty_reward_scale = -0.0
    dof_close_to_default_reward_scale = -0.01

    # pt load 추가 ################################
    load_run = "2026-01-28_22-42-04"
    load_checkpoint = -1
    ###############################################


@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

