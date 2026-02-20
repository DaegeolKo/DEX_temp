import torch

class Go2Solver :
    def __init__(self, device='cpu') : 
        self.device = device

        self.L_HIP = torch.tensor(0.0955, dtype=torch.float32, device=self.device)
        self.L_THIGH = torch.tensor(0.213, dtype=torch.float32, device=self.device)
        self.L_CALF = torch.tensor(0.213, dtype=torch.float32, device=self.device)

        self.HIP_OFFSETS = torch.tensor([
            [ 0.1934, 0.0465,  0.0 ],  # FL
            [ 0.1934,  -0.0465,  0.0 ],  # FR
            [-0.1934, 0.0465,  0.0 ],  # RL
            [-0.1934,  -0.0465,  0.0 ],  # RR
        ], dtype=torch.float32).unsqueeze(0)


    def go2_fk_new(self, joint_angles: torch.Tensor) -> torch.Tensor:
        batch_size = joint_angles.size(0)
        
        q_reshaped = joint_angles.reshape(batch_size, 3, 4).transpose(1, 2)
        q = q_reshaped
        
        q_hip, q_thigh, q_calf = q[:, :, 0], q[:, :, 1], q[:, :, 2]

        cos_q_thigh, sin_q_thigh = torch.cos(q_thigh), torch.sin(q_thigh)
        cos_q_thigh_calf, sin_q_thigh_calf = torch.cos(q_thigh + q_calf), torch.sin(q_thigh + q_calf)
        
        x_init = -1 * (self.L_THIGH * sin_q_thigh + self.L_CALF * sin_q_thigh_calf)
        z_init = (self.L_THIGH * cos_q_thigh + self.L_CALF * cos_q_thigh_calf)
        
        x_leg = x_init
        
        L_proj = z_init
        
        alpha = torch.abs(L_proj) - torch.abs(self.L_HIP * torch.tan(q_hip))
        
        z_leg = -1 * alpha * torch.cos(q_hip)     
            
        beta = torch.sqrt(self.L_HIP**2 + L_proj**2)
        
        y_leg = torch.sqrt(torch.clamp(beta**2 - z_leg**2, min=1e-6))

        side_sign = torch.tensor([1, -1, 1, -1], device=joint_angles.device, dtype=torch.float32).unsqueeze(0)

        x_hip_frame = x_leg
        y_hip_frame = side_sign * y_leg
        z_hip_frame = z_leg
        
        foot_pos_base_frame = torch.stack([x_hip_frame, y_hip_frame, z_hip_frame], dim=-1) + self.HIP_OFFSETS.to(self.device)
        
        return foot_pos_base_frame.view(batch_size, 12)

    
    def go2_ik_new(self, target_foot_positions: torch.Tensor) -> torch.Tensor:

        batch_size = target_foot_positions.size(0)

        local_target_positions = target_foot_positions - self.HIP_OFFSETS.to(self.device)
        x = local_target_positions[..., 0]
        y = local_target_positions[..., 1]
        z = local_target_positions[..., 2]

        side_sign = torch.tensor([1, -1, 1, -1], device=target_foot_positions.device, dtype=x.dtype).unsqueeze(0)
        y_signed = y * side_sign

        # --- 1. 실패 가능성이 있는 값들을 미리 계산 ---
        L_prime = x**2 + y_signed**2 + z**2 - self.L_HIP**2
        cos_theta_2 = (L_prime - self.L_THIGH**2 - self.L_CALF**2) / (2 * self.L_THIGH * self.L_CALF)
        sqrt_arg_hip = z**2 + y_signed**2 - self.L_HIP**2

        # --- 2. '진짜 도달 불가능'한 모든 경우를 마스크로 생성 ---
        mask_too_far = torch.abs(cos_theta_2) > 1.0
        mask_too_close = sqrt_arg_hip < 0
        unreachable_mask = mask_too_far | mask_too_close

        # --- 3. '안전하게 보정된' 값으로 모든 각도 계산 ---
        cos_theta_2_clamped = torch.clamp(cos_theta_2, -1.0, 1.0)
        safe_sqrt_arg_hip = torch.sqrt(torch.clamp(sqrt_arg_hip, min=1e-6))
        
        theta_2 = -torch.acos(cos_theta_2_clamped)
        
        alpha = torch.atan2(x, safe_sqrt_arg_hip)
        beta = torch.atan2(self.L_CALF * torch.sin(theta_2), self.L_THIGH + self.L_CALF * torch.cos(theta_2))
        theta_1 = -1 * (alpha + beta)

        theta_0 = torch.atan2(y_signed, -z) - torch.atan2(self.L_HIP, safe_sqrt_arg_hip)
        theta_0 = theta_0 * side_sign

        # --- 4. 최종 결과 생성 및 NaN 처리 ---
        joint_angles_leg_grouped = torch.stack((theta_0, theta_1, theta_2), dim=-1)

        # 수정 후: torch.where 사용
        # unreachable_mask가 True인 위치에는 NaN을, False인 위치에는 원래 값을 채워 넣음
        nan_tensor = torch.tensor(float('nan'), device=self.device, dtype=x.dtype)
        joint_angles_leg_grouped = torch.where(
            unreachable_mask.unsqueeze(-1), 
            nan_tensor, 
            joint_angles_leg_grouped
        )

        joint_angles_joint_grouped = joint_angles_leg_grouped.transpose(1, 2).contiguous().view(batch_size, -1)
        
        return joint_angles_joint_grouped