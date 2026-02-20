import torch
import math
from tqdm import tqdm

from FK_IK_Solver import Go2Solver

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_FILENAME_PT = 'foot_positions_100k.pt' 

VALID_LIMITS_RAD = {
    'hip':   (math.radians(-48), math.radians(48)),
    'thigh': (math.radians(-200), math.radians(90)),
    'calf':  (math.radians(-156), math.radians(-48))
}

try:
    foot_positions_flat = torch.load(INPUT_FILENAME_PT).to(DEVICE)
    num_samples = len(foot_positions_flat)
except FileNotFoundError:
    print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다.")
    exit()

solver = Go2Solver(device=DEVICE)

with torch.no_grad():
    batch_size = foot_positions_flat.size(0)
    # 입력도 Leg-wise이므로 (Batch, 4, 3)으로 reshape
    foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
    
    # IK 계산 (결과는 Leg-wise: [H1, T1, C1, H2...])
    calculated_joint_angles = solver.go2_ik_new(foot_positions_reshaped)


# --- [수정된 부분] 데이터 파싱 로직 변경 ---
# 이전: angles_reshaped = calculated_joint_angles.reshape(-1, 3, 4).transpose(1, 2)
# 수정: Solver 출력 순서 그대로 (Batch, Leg, Joint)로 묶음
angles_reshaped = calculated_joint_angles.reshape(-1, 4, 3)

hip_angles = angles_reshaped[..., 0]
thigh_angles = angles_reshaped[..., 1]
calf_angles = angles_reshaped[..., 2]

# --- 이하 검증 로직 동일 ---
nan_samples_mask = torch.isnan(calculated_joint_angles).any(dim=1)
num_nan_samples = nan_samples_mask.sum().item()

valid_hip_mask = (hip_angles >= VALID_LIMITS_RAD['hip'][0]) & (hip_angles <= VALID_LIMITS_RAD['hip'][1])
valid_thigh_mask = (thigh_angles >= VALID_LIMITS_RAD['thigh'][0]) & (thigh_angles <= VALID_LIMITS_RAD['thigh'][1])
valid_calf_mask = (calf_angles >= VALID_LIMITS_RAD['calf'][0]) & (calf_angles <= VALID_LIMITS_RAD['calf'][1])

all_joints_valid_mask = valid_hip_mask.all(dim=1) & valid_thigh_mask.all(dim=1) & valid_calf_mask.all(dim=1)
truly_valid_samples_mask = all_joints_valid_mask & (~nan_samples_mask)
num_limit_violated_samples = num_samples - num_nan_samples - truly_valid_samples_mask.sum().item()


max_violation_rad = 0.0
avg_violation_rad = 0.0

if num_limit_violated_samples > 0:
    hip_over_max = torch.relu(hip_angles - VALID_LIMITS_RAD['hip'][1])
    hip_under_min = torch.relu(VALID_LIMITS_RAD['hip'][0] - hip_angles)
    hip_violations = hip_over_max + hip_under_min

    thigh_over_max = torch.relu(thigh_angles - VALID_LIMITS_RAD['thigh'][1])
    thigh_under_min = torch.relu(VALID_LIMITS_RAD['thigh'][0] - thigh_angles)
    thigh_violations = thigh_over_max + thigh_under_min

    calf_over_max = torch.relu(calf_angles - VALID_LIMITS_RAD['calf'][1])
    calf_under_min = torch.relu(VALID_LIMITS_RAD['calf'][0] - calf_angles)
    calf_violations = calf_over_max + calf_under_min

    all_violations = torch.cat([
        hip_violations.flatten(),
        thigh_violations.flatten(),
        calf_violations.flatten()
    ])

    actual_violations = all_violations[all_violations > 0]
    
    if len(actual_violations) > 0:
        max_violation_rad = actual_violations.max().item()
        avg_violation_rad = actual_violations.mean().item()

print("---------- IK 검증 결과 (수정됨) ----------")
print(f"총 샘플 수: {num_samples}")
print("-" * 34)
print(f"IK 해가 없어 'NaN'이 발생한 샘플: {num_nan_samples} 개")
print(f"관절 제한(Joint Limit)을 벗어난 샘플: {num_limit_violated_samples} 개")
print(f"유효한 해가 나온 샘플 (Success): {truly_valid_samples_mask.sum().item()} 개")
print("------------------------------------")

if num_limit_violated_samples > 0:
    print("--- 벗어난 샘플 상세 분석 ---")
    print(f"최대 위반 정도: {max_violation_rad:.6f} rad (~{math.degrees(max_violation_rad):.2f}°)")
    print(f"평균 위반 정도: {avg_violation_rad:.6f} rad (~{math.degrees(avg_violation_rad):.2f}°)")
    print("------------------------------------")

print("\n Complete")

print("\n---------- [중요] 발 위치 정밀도 검증 ----------")

# 1. NaN이 없는 정상적인 샘플만 골라냅니다. (중요!)
# nan_samples_mask는 위에서 이미 구했습니다. (True면 NaN 포함)
valid_indices = ~nan_samples_mask  # NaN이 아닌 것들만 True

if valid_indices.sum() == 0:
    print("모든 샘플이 NaN입니다.")
else:
    # 2. 정상 샘플만 가지고 FK 재계산 및 오차 측정
    # 계산된 각도 중 정상인 것만 선택
    valid_joint_angles = calculated_joint_angles[valid_indices]
    
    # 목표했던 발 위치 중 정상인 것만 선택
    valid_target_positions = foot_positions_reshaped[valid_indices]

    with torch.no_grad():
        # FK 계산
        reconstructed_positions_flat = solver.go2_fk_new(valid_joint_angles)
        reconstructed_positions = reconstructed_positions_flat.view(-1, 4, 3)

        # 오차 계산 (유클리드 거리)
        pos_error = torch.norm(valid_target_positions - reconstructed_positions, dim=-1)

    # 3. 통계 출력
    mean_pos_error = pos_error.mean().item()
    max_pos_error = pos_error.max().item()

    print(f"평균 위치 오차: {mean_pos_error:.6f} meters")
    print(f"최대 위치 오차: {max_pos_error:.6f} meters")

    if mean_pos_error < 1e-3: # 1mm 이내
        print("=> 결론: Solver는 수학적으로 완벽합니다. (Limit 위반은 다중 해 문제입니다.)")
    else:
        print("=> 결론: Solver 수식에 문제가 있습니다.")

print("------------------------------------------------")
print("------------------------------------------------")