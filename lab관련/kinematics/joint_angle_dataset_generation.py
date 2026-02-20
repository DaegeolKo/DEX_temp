import torch
import math
import numpy as np

NUM_SAMPLES = 100_000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_FILENAME_PT = 'joint_angles_100k_ordered.pt'
OUTPUT_FILENAME_TXT = 'joint_angles_100k_ordered.txt'

hip_limit_rad = [math.radians(-48), math.radians(48)]
thigh_limit_rad = [math.radians(-200), math.radians(90)]
calf_limit_rad = [math.radians(-156), math.radians(-48)]

min_limits = torch.tensor([hip_limit_rad[0], thigh_limit_rad[0], calf_limit_rad[0]], device=DEVICE)
max_limits = torch.tensor([hip_limit_rad[1], thigh_limit_rad[1], calf_limit_rad[1]], device=DEVICE)

# 1. 처음부터 (Samples, 4, 3) 형태로 생성합니다.
# 이렇게 하면 [Leg1, Leg2, Leg3, Leg4] 순서가 자연스럽게 잡힙니다.
random_raw = torch.rand(NUM_SAMPLES, 4, 3, device=DEVICE)

# 2. 스케일링 (Broadcasting 자동 적용)
joint_angles_legs = random_raw * (max_limits - min_limits) + min_limits

# 3. (Samples, 12)로 펼칩니다.
# 결과 순서: [Hip1, Thigh1, Calf1, Hip2, Thigh2, Calf2, ...] -> 시뮬레이션과 완벽 호환
joint_angles_data = joint_angles_legs.view(NUM_SAMPLES, 12)

torch.save(joint_angles_data, OUTPUT_FILENAME_PT)
np.savetxt(OUTPUT_FILENAME_TXT, joint_angles_data.cpu().numpy(), fmt='%.6f')

print("Complete. Data generated in Leg-wise order (Isaac Lab compatible).")