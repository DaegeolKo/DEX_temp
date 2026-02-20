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
        
        # [수정됨] transpose(1, 2) 삭제
        # 입력 데이터가 [Leg1_HTC, Leg2_HTC...] 순서이므로 (Batch, 4, 3)으로만 변경하면 됩니다.
        q_reshaped = joint_angles.reshape(batch_size, 4, 3) 
        
        q = q_reshaped
        
        # q[:, :, 0] -> (Batch, 4) : 모든 다리의 Hip 값 추출
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

        # 여기서 shape mismatch가 발생했었습니다. 이제 (Batch, 4)로 일치합니다.
        x_hip_frame = x_leg
        y_hip_frame = side_sign * y_leg
        z_hip_frame = z_leg
        
        foot_pos_base_frame = torch.stack([x_hip_frame, y_hip_frame, z_hip_frame], dim=-1) + self.HIP_OFFSETS.to(self.device)
        
        return foot_pos_base_frame.view(batch_size, 12)

    
    def go2_ik_new(self, target_foot_positions: torch.Tensor) -> torch.Tensor:

        batch_size = target_foot_positions.size(0)

        # 입력이 (Batch, 12)인 경우 reshape 필요 (코드 사용처에 따라 다름)
        if target_foot_positions.dim() == 2:
            target_foot_positions = target_foot_positions.view(batch_size, 4, 3)

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

        # --- 2. 마스킹 및 Clamping ---
        # 마스킹 로직(NaN 처리)은 제거하고, Clamping(값 보정)만 유지합니다.
        cos_theta_2_clamped = torch.clamp(cos_theta_2, -1.0, 1.0)
        safe_sqrt_arg_hip = torch.sqrt(torch.clamp(sqrt_arg_hip, min=1e-6))
        
        theta_2 = -torch.acos(cos_theta_2_clamped)
        
        alpha = torch.atan2(x, safe_sqrt_arg_hip)
        beta = torch.atan2(self.L_CALF * torch.sin(theta_2), self.L_THIGH + self.L_CALF * torch.cos(theta_2))
        theta_1 = -1 * (alpha + beta)

        theta_0 = torch.atan2(y_signed, -z) - torch.atan2(self.L_HIP, safe_sqrt_arg_hip)
        theta_0 = theta_0 * side_sign

        # --- 3. 최종 결과 생성 ---
        # (Batch, 4, 3) -> [Leg, Joint] 순서 유지
        joint_angles_leg_grouped = torch.stack((theta_0, theta_1, theta_2), dim=-1)

        # 평탄화하여 (Batch, 12) 반환
        joint_angles_final = joint_angles_leg_grouped.contiguous().view(batch_size, -1)
        
        return joint_angles_final