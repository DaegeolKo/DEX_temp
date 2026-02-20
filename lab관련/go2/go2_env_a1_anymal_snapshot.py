# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import re
import torch
import torch.nn as nn

from pxr import Gf
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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
        # IK backend:
        # - "analytic_go2": use the analytic Go2 FK/IK solver (Go2Solver).
        # - "dls": use Jacobian-based damped least squares IK (robot-agnostic, slower but portable).
        self._ik_solver = str(getattr(self.cfg, "ik_solver", "") or "").strip().lower()
        if not self._ik_solver:
            usd_path = str(getattr(getattr(self.cfg.robot, "spawn", None), "usd_path", "") or "")
            self._ik_solver = "analytic_go2" if ("go2" in usd_path.lower()) else "dls"
        if self._ik_solver not in {"analytic_go2", "dls"}:
            raise ValueError(f"Unsupported ik_solver={self._ik_solver!r}. Expected one of: 'analytic_go2', 'dls'.")
        # DLS IK params (used when ik_solver='dls')
        self._ik_dls_lambda = float(getattr(self.cfg, "ik_dls_lambda", 0.05) or 0.05)
        self._ik_dls_step = float(getattr(self.cfg, "ik_dls_step", 1.0) or 1.0)

        # --- Reference joint pose (robot default) ---
        # The env cfg may override init_state/default_joint_pos (e.g. mean-pose experiments),
        # but for rewards/metrics we often want the robot's canonical "standing" pose.
        self._reference_default_joint_pos = None
        try:
            usd_path = str(getattr(getattr(self.cfg.robot, "spawn", None), "usd_path", "") or "")
            usd_path_l = usd_path.lower()

            robot_ref_cfg = None
            if "go2" in usd_path_l:
                from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as robot_ref_cfg  # type: ignore
            elif "a1" in usd_path_l:
                from isaaclab_assets.robots.unitree import UNITREE_A1_CFG as robot_ref_cfg  # type: ignore
            elif "anymal" in usd_path_l:
                # Default to ANYmal-C reference (covers our intended use-case here).
                from isaaclab_assets.robots.anymal import ANYMAL_C_CFG as robot_ref_cfg  # type: ignore
            else:
                robot_ref_cfg = self.cfg.robot

            joint_names = list(self._robot.data.joint_names)
            pattern_to_val = dict(getattr(getattr(robot_ref_cfg, "init_state", None), "joint_pos", {}) or {})
            if joint_names and pattern_to_val:
                patterns: list[tuple[re.Pattern[str], float]] = [
                    (re.compile(pat), float(val)) for pat, val in pattern_to_val.items()
                ]
                ref_vals: list[float] = []
                for name in joint_names:
                    matched = False
                    for pat, val in patterns:
                        if pat.fullmatch(name) or pat.match(name):
                            ref_vals.append(val)
                            matched = True
                            break
                    if not matched:
                        ref_vals.append(0.0)
                self._reference_default_joint_pos = torch.tensor(
                    ref_vals, device=self.device, dtype=torch.float32
                ).view(1, -1)
        except Exception:
            # Fallback: use the current default joint pose.
            self._reference_default_joint_pos = None

        # --- Termination params ---
        self._base_contact_termination = bool(getattr(self.cfg, "base_contact_termination", True))
        self._base_contact_force_threshold = float(getattr(self.cfg, "base_contact_force_threshold", 1.0))
        self._base_contact_termination_delay_s = float(getattr(self.cfg, "base_contact_termination_delay_s", 0.0) or 0.0)
        self._base_contact_termination_delay_steps = int(round(self._base_contact_termination_delay_s / self.step_dt))

        # --- Optional: decode PPO action -> foot positions via NF decoder (nsf12_stand) ---
        self._use_nf_decoder = bool(getattr(self.cfg, "use_nf_decoder", False))
        self._nf_decoder_ckpt_path = str(getattr(self.cfg, "nf_decoder_ckpt_path", "") or "")
        self._nf_action_mode = str(getattr(self.cfg, "nf_action_mode", "eps") or "eps").strip().lower()
        self._nf_z_offset_from_default_pose = bool(getattr(self.cfg, "nf_z_offset_from_default_pose", False))
        self._nf_action_clip = float(getattr(self.cfg, "nf_action_clip", 0.0) or 0.0)
        self._nf_std_eps = float(getattr(self.cfg, "nf_std_eps", 1e-6) or 1e-6)
        self._nf_model = None
        self._nf_mean = None
        self._nf_std = None
        self._nf_loc = None
        self._nf_scale = None
        self._nf_z_offset = None

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
        # - Go2: base link is usually named "base"
        # - A1: base link is named "trunk"
        try:
            self._base_id, _ = self._contact_sensor.find_bodies("base")
        except ValueError:
            self._base_id, _ = self._contact_sensor.find_bodies("trunk")

        # Resolve feet/joints naming conventions.
        # We support:
        # - Unitree Go2/A1 style: FL/FR/RL/RR with *_hip_joint/*_thigh_joint/*_calf_joint and *_foot bodies.
        # - ANYmal style: LF/RF/LH/RH with *_HAA/*_HFE/*_KFE and *_FOOT bodies.
        anymal_feet = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        go2_feet = [".*FL_foot", ".*FR_foot", ".*RL_foot", ".*RR_foot"]

        self._leg_joint_ids = []
        self._leg_joint_names = []
        try:
            # ANYmal naming
            self._feet_body_ids_ordered, self._feet_body_names_ordered = self._robot.find_bodies(
                anymal_feet, preserve_order=True
            )
            self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
            self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

            leg_joint_specs = {
                "LF": ["LF_HAA", "LF_HFE", "LF_KFE"],
                "RF": ["RF_HAA", "RF_HFE", "RF_KFE"],
                "LH": ["LH_HAA", "LH_HFE", "LH_KFE"],
                "RH": ["RH_HAA", "RH_HFE", "RH_KFE"],
            }
            for leg in ("LF", "RF", "LH", "RH"):
                ids, names = self._robot.find_joints(leg_joint_specs[leg], preserve_order=True)
                if len(ids) != 3:
                    raise ValueError(f"Failed to resolve 3 joints for leg {leg}. Got {len(ids)}: {names}")
                self._leg_joint_ids.append(ids)
                self._leg_joint_names.append(names)
        except ValueError:
            # Go2/A1 naming (default)
            self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
            self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_(thigh|calf)")

            # Resolve feet body indices in a deterministic leg order. This defines the (4,3) ordering used
            # throughout the environment: [FL, FR, RL, RR] x [x,y,z].
            self._feet_body_ids_ordered, self._feet_body_names_ordered = self._robot.find_bodies(
                go2_feet, preserve_order=True
            )
            if len(self._feet_body_ids_ordered) != 4:
                raise ValueError(
                    f"Failed to resolve 4 feet bodies. Got {len(self._feet_body_ids_ordered)}: {self._feet_body_names_ordered}"
                )

            # Resolve per-leg joint indices for DLS IK (same leg order as above).
            for leg in ("FL", "FR", "RL", "RR"):
                ids, names = self._robot.find_joints(
                    [f".*{leg}_hip_joint", f".*{leg}_thigh_joint", f".*{leg}_calf_joint"], preserve_order=True
                )
                if len(ids) != 3:
                    raise ValueError(f"Failed to resolve 3 joints for leg {leg}. Got {len(ids)}: {names}")
                self._leg_joint_ids.append(ids)
                self._leg_joint_names.append(names)

        # Default foot positions (base frame). For most use-cases we can lazily compute these from sim on first step.
        # However, if z-offset-from-default is enabled, the NF loader needs these immediately.
        self._default_foot_positions: torch.Tensor | None = None
        if self._use_nf_decoder and self._nf_z_offset_from_default_pose:
            self._default_foot_positions = self._compute_default_foot_positions_from_sim()
        
        # scailing of sampling space(x,y,z)    
        self.scales = torch.tensor([0.3, 0.3, 0.1], device=self.device)
        
        # ik 발산 시, 발생한 index 확인용
        self.ik_failure_penalty = torch.zeros(self.num_envs, device=self.device)

        if self._use_nf_decoder:
            self._load_nf_decoder()
            print(
                "[INFO][Go2Env] NF decoder enabled: "
                f"mode={self._nf_action_mode!r} clip={self._nf_action_clip} ckpt={self._nf_decoder_ckpt_path}"
            )

    def _compute_default_foot_positions_from_sim(self) -> torch.Tensor:
        """Compute default foot positions (base frame) from the current simulated state."""
        # base pose (root link frame) in world
        base_pose = self._robot.data.root_link_pose_w
        base_pos_w = base_pose[:, :3]
        base_quat_w = base_pose[:, 3:7]
        # foot positions in world (ordered)
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids_ordered, :]
        # transform into base frame
        rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
        quat_rep = base_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        rel_b = math_utils.quat_apply_inverse(quat_rep, rel_w.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        return rel_b.to(dtype=torch.float32)

    def _solve_ik_dls(self, target_foot_positions_b: torch.Tensor) -> torch.Tensor:
        """Jacobian-based damped least-squares IK for 4 feet (position-only)."""
        # Ensure shapes
        if target_foot_positions_b.shape != (self.num_envs, 4, 3):
            raise ValueError(
                f"target_foot_positions_b must have shape (num_envs,4,3), got {tuple(target_foot_positions_b.shape)}"
            )

        # Base pose
        base_pose = self._robot.data.root_link_pose_w
        base_pos_w = base_pose[:, :3]
        base_quat_w = base_pose[:, 3:7]

        # Target positions in world
        quat_rep = base_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        target_offset_w = math_utils.quat_apply(quat_rep, target_foot_positions_b.reshape(-1, 3)).reshape(
            self.num_envs, 4, 3
        )
        target_pos_w = base_pos_w.unsqueeze(1) + target_offset_w

        # Current foot positions in world
        curr_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids_ordered, :]
        pos_err = (target_pos_w - curr_pos_w).to(dtype=torch.float32)  # (N,4,3)

        # Jacobians: (N, num_bodies, 6, 6+num_joints)
        jacobians = self._robot.root_physx_view.get_jacobians()

        # Start from current joint positions and update per leg
        q_des = self._robot.data.joint_pos.clone()
        lambda_sq = float(self._ik_dls_lambda) ** 2
        eye3 = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)

        for leg_idx in range(4):
            body_id = self._feet_body_ids_ordered[leg_idx]
            joint_ids = self._leg_joint_ids[leg_idx]
            # PhysX jacobian includes 6 base dofs first for floating base articulations
            jacobi_joint_ids = [jid + 6 for jid in joint_ids]
            J = jacobians[:, body_id, :3, jacobi_joint_ids].to(dtype=torch.float32)  # (N,3,3)
            e = pos_err[:, leg_idx, :].unsqueeze(-1)  # (N,3,1)

            JJt = torch.bmm(J, J.transpose(1, 2))  # (N,3,3)
            A = JJt + lambda_sq * eye3  # (N,3,3)
            y = torch.linalg.solve(A, e)  # (N,3,1)
            dq = torch.bmm(J.transpose(1, 2), y).squeeze(-1)  # (N,3)
            dq = dq * float(self._ik_dls_step)

            q_leg = self._robot.data.joint_pos[:, joint_ids].to(dtype=torch.float32)
            q_leg_des = q_leg + dq
            # clamp to joint limits
            lim = self._robot.data.joint_pos_limits[:, joint_ids, :].to(dtype=torch.float32)
            q_leg_des = torch.clamp(q_leg_des, min=lim[:, :, 0], max=lim[:, :, 1])
            q_des[:, joint_ids] = q_leg_des

        return q_des

    def _load_nf_decoder(self):
        if not self._nf_decoder_ckpt_path:
            raise ValueError("use_nf_decoder=True 이면 cfg.nf_decoder_ckpt_path 가 필요합니다.")

        try:
            import normflows as nf
        except ImportError as e:
            raise ImportError(
                "normflows를 import 할 수 없습니다. IsaacLab python에서 다음을 실행하세요:\n"
                "  ./isaaclab.sh -p -m pip install -e /home/kdg/normalizing-flows"
            ) from e

        class FixedPermutation1d(nn.Module):
            """Fixed (invertible) permutation over the feature dimension for (B, D) tensors."""

            def __init__(self, perm: torch.Tensor):
                super().__init__()
                if perm.dim() != 1:
                    raise ValueError(f"perm must be 1D, got shape={tuple(perm.shape)}")
                perm = perm.to(dtype=torch.long)
                self.register_buffer("perm", perm)
                self.register_buffer("inv_perm", torch.argsort(perm))

            def forward(self, x: torch.Tensor):
                x = x.index_select(1, self.perm)
                log_det = torch.zeros(len(x), device=x.device, dtype=x.dtype)
                return x, log_det

            def inverse(self, x: torch.Tensor):
                x = x.index_select(1, self.inv_perm)
                log_det = torch.zeros(len(x), device=x.device, dtype=x.dtype)
                return x, log_det

        class VectorInvertible1x1Conv(nn.Module):
            """Adapter: apply nf.flows.Invertible1x1Conv to vector data (B, D) via (B, D, 1, 1)."""

            def __init__(self, dim: int, *, use_lu: bool = True):
                super().__init__()
                self.dim = int(dim)
                self._conv = nf.flows.Invertible1x1Conv(self.dim, use_lu=bool(use_lu))

            def forward(self, x: torch.Tensor):
                if x.dim() != 2 or x.size(1) != self.dim:
                    raise ValueError(f"Expected (B,{self.dim}) input, got {tuple(x.shape)}")
                x4 = x[:, :, None, None]
                y4, log_det = self._conv(x4)
                y = y4[:, :, 0, 0]
                return y, log_det

            def inverse(self, y: torch.Tensor):
                if y.dim() != 2 or y.size(1) != self.dim:
                    raise ValueError(f"Expected (B,{self.dim}) input, got {tuple(y.shape)}")
                y4 = y[:, :, None, None]
                x4, log_det = self._conv.inverse(y4)
                x = x4[:, :, 0, 0]
                return x, log_det

        ckpt = torch.load(self._nf_decoder_ckpt_path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(
                f"NF 체크포인트 포맷이 예상과 다릅니다: {self._nf_decoder_ckpt_path} "
                "(dict with key 'model' 기대)"
            )

        state_dict = ckpt["model"]
        saved_args = ckpt.get("args", {})
        mean = ckpt.get("mean")
        std = ckpt.get("std")
        if mean is None or std is None:
            raise ValueError(
                f"nsf12_stand 체크포인트에 mean/std가 없습니다: {self._nf_decoder_ckpt_path}"
            )

        dim = int(saved_args.get("dim", 12))
        num_layers = int(saved_args.get("num_layers", 8))
        hidden = int(saved_args.get("hidden", 128))
        num_blocks = int(saved_args.get("num_blocks", 2))
        num_bins = int(saved_args.get("num_bins", 8))
        tail_bound = float(saved_args.get("tail_bound", 3.0))
        actnorm = bool(saved_args.get("actnorm", False))
        mixing = str(saved_args.get("mixing", "") or "").strip().lower()
        mixing_use_lu = bool(saved_args.get("mixing_use_lu", False))
        trainable_base = bool(saved_args.get("trainable_base", True))

        base = nf.distributions.base.DiagGaussian(dim, trainable=trainable_base)
        flows = []
        if actnorm:
            flows.append(nf.flows.ActNorm((dim,)))

        if not mixing:
            mixing = "none"

        if mixing not in {"none", "permute", "inv_affine", "inv_1x1conv"}:
            raise ValueError(
                f"지원하지 않는 NF mixing={mixing!r}. 현재 지원: 'none', 'permute', 'inv_affine', 'inv_1x1conv'. "
                f"(ckpt={self._nf_decoder_ckpt_path})"
            )

        for i in range(num_layers):
            flows.append(
                nf.flows.CoupledRationalQuadraticSpline(
                    num_input_channels=dim,
                    num_blocks=num_blocks,
                    num_hidden_channels=hidden,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    activation=nn.ReLU,
                    reverse_mask=bool(i % 2),
                    init_identity=True,
                )
            )
            if mixing != "none" and i < (num_layers - 1):
                if mixing == "permute":
                    flows.append(FixedPermutation1d(torch.randperm(dim)))
                elif mixing == "inv_affine":
                    flows.append(nf.flows.InvertibleAffine(dim, use_lu=mixing_use_lu))
                else:
                    flows.append(VectorInvertible1x1Conv(dim, use_lu=mixing_use_lu))
        model = nf.NormalizingFlow(base, flows).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.requires_grad_(False)

        self._nf_model = model
        self._nf_mean = mean.to(self.device).view(1, -1).to(dtype=torch.float32)
        self._nf_std = std.to(self.device).view(1, -1).to(dtype=torch.float32).clamp_min(self._nf_std_eps)
        self._nf_loc = model.q0.loc.detach().to(device=self.device, dtype=torch.float32)
        self._nf_scale = torch.exp(model.q0.log_scale.detach()).to(device=self.device, dtype=torch.float32)
        if self._nf_mean.numel() != dim or self._nf_std.numel() != dim:
            raise ValueError(
                f"NF mean/std dim mismatch: mean={tuple(self._nf_mean.shape)} std={tuple(self._nf_std.shape)} dim={dim}"
            )

        # Optional: compute z-offset so that z=0 maps to the default standing foot positions (z-mode only).
        self._nf_z_offset = torch.zeros(1, dim, device=self.device, dtype=torch.float32)
        if self._nf_z_offset_from_default_pose:
            if self._nf_action_mode != "z":
                raise ValueError("nf_z_offset_from_default_pose=True 는 nf_action_mode='z' 에서만 지원합니다.")
            with torch.no_grad():
                x_target_raw = self._default_foot_positions[0].reshape(1, -1).to(dtype=torch.float32)
                x_target_std = (x_target_raw - self._nf_mean) / self._nf_std
                inv_out = self._nf_model.inverse(x_target_std)
                z_target = inv_out[0] if isinstance(inv_out, tuple) else inv_out
                self._nf_z_offset = z_target.detach().to(device=self.device, dtype=torch.float32)
            print(
                "[INFO][Go2Env] NF z-offset enabled (from default pose): "
                f"min={float(self._nf_z_offset.min()):.3f} max={float(self._nf_z_offset.max()):.3f} "
                f"mean={float(self._nf_z_offset.mean()):.3f}"
            )

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

        if self._use_nf_decoder:
            if self._nf_action_clip > 0.0:
                self._actions = torch.clamp(self._actions, -self._nf_action_clip, self._nf_action_clip)

            with torch.no_grad():
                if self._nf_action_mode == "z":
                    z = self._actions
                    if self._nf_z_offset is not None:
                        z = z + self._nf_z_offset.to(dtype=z.dtype)
                elif self._nf_action_mode == "eps":
                    z = self._nf_loc.to(dtype=self._actions.dtype) + self._nf_scale.to(dtype=self._actions.dtype) * self._actions
                else:
                    raise ValueError(f"지원하지 않는 nf_action_mode={self._nf_action_mode!r} (기대: 'z' or 'eps')")

                x_std = self._nf_model(z)
                x_raw = x_std * self._nf_std + self._nf_mean
                target_foot_positions = x_raw.reshape(self.num_envs, 4, 3)
        else:
            if self._default_foot_positions is None:
                self._default_foot_positions = self._compute_default_foot_positions_from_sim()
            actions_reshaped = self._actions.reshape(self.num_envs, 4, 3)
            normalized_actions_reshaped = actions_reshaped * self.scales * self.cfg.action_scale
            target_foot_positions = self._default_foot_positions + normalized_actions_reshaped * 0.3
        
        if self._ik_solver == "analytic_go2":
            target_joint_angles = self.solver.go2_ik_new(target_foot_positions=target_foot_positions)
        else:
            target_joint_angles = self._solve_ik_dls(target_foot_positions_b=target_foot_positions)

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
        default_joint_ref = (
            self._reference_default_joint_pos
            if self._reference_default_joint_pos is not None
            else self._robot.data.default_joint_pos
        )
        rew_dof_close_to_default = torch.sum(torch.square(self._robot.data.joint_pos - default_joint_ref), dim=1)

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
        if self._base_contact_termination:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            died = torch.any(
                torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0]
                > self._base_contact_force_threshold,
                dim=1,
            )
            if self._base_contact_termination_delay_steps > 0:
                died = died & (self.episode_length_buf >= self._base_contact_termination_delay_steps)
        else:
            died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        lin_x_range = [1.5, 1.5]
        rand_x = (torch.rand(num_resets, 1, device=self.device) * (lin_x_range[1] - lin_x_range[0])) + lin_x_range[0]
        lin_y_range = [0.0, 0.0]
        rand_y = (torch.rand(num_resets, 1, device=self.device) * (lin_y_range[1] - lin_y_range[0])) + lin_y_range[0]
        ang_vel_range = [1.0, 1.0]
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
