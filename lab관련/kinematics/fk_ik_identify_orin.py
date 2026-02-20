import torch
import numpy as np
import math
from tqdm import tqdm

from FK_IK_Solver import Go2Solver

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 파일명은 사용자 환경에 맞게 확인하세요 (.txt 또는 .pt)
INPUT_FILENAME_TXT = 'joint_angles_100k_ordered.txt' 

TOLERANCE = 1e-4 # 허용 오차 약간 완화

try:
    q_original_np = np.loadtxt(INPUT_FILENAME_TXT)
    q_original = torch.from_numpy(q_original_np).float().to(DEVICE)
    num_samples = len(q_original)
except FileNotFoundError:
    print(f"오류: '{INPUT_FILENAME_TXT}' 파일을 찾을 수 없습니다.")
    exit()

solver = Go2Solver(device=DEVICE)

print("검증 시작...")
with torch.no_grad():
    # 1. FK: 원래 각도 -> 발 위치
    foot_positions_flat = solver.go2_fk_new(q_original)
    batch_size = foot_positions_flat.size(0)
    foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
    
    # 2. IK: 발 위치 -> 계산된 각도
    q_calculated = solver.go2_ik_new(foot_positions_reshaped)
    
    # 3. Re-FK: 계산된 각도 -> 다시 발 위치 (검증의 핵심!)
    foot_positions_recalc_flat = solver.go2_fk_new(q_calculated)
    foot_positions_recalc = foot_positions_recalc_flat.view(batch_size, 4, 3)

# --- 분석 ---

# 1. NaN 처리 (NaN이 있는 행은 통계에서 제외)
valid_mask = ~torch.isnan(q_calculated).any(dim=1)
num_valid = valid_mask.sum().item()
num_nan = num_samples - num_valid

q_orig_valid = q_original[valid_mask]
q_calc_valid = q_calculated[valid_mask]
pos_orig_valid = foot_positions_reshaped[valid_mask]
pos_recalc_valid = foot_positions_recalc[valid_mask]

# 2. 관절 각도 오차 (다중 해 때문에 클 수 있음 -> 정상이니 놀라지 마세요)
abs_error_joints = torch.abs(q_orig_valid - q_calc_valid)
mean_error_joints = abs_error_joints.mean(dim=0) # 관절별 평균

# 3. 발 위치 오차 (이게 작아야 진짜 성공!)
pos_error = torch.norm(pos_orig_valid - pos_recalc_valid, dim=-1) # 유클리드 거리
mean_pos_error = pos_error.mean().item()
max_pos_error = pos_error.max().item()

# 4. Perfect Match (각도까지 똑같은 경우)
perfect_match_mask = torch.all(torch.isclose(q_orig_valid, q_calc_valid, atol=TOLERANCE), dim=1)
num_perfect_matches = perfect_match_mask.sum().item()


print("\n========== [최종 검증 결과] ==========")
print(f"총 샘플: {num_samples}개")
print(f"NaN 발생 샘플: {num_nan}개 (제외됨)")
print("-" * 40)
print(f"1. [관절 각도 비교] (다중 해로 인해 오차가 클 수 있음)")
print(f"   - 완벽 일치(Perfect Match) 비율: {(num_perfect_matches/num_valid)*100:.2f}%")
print(f"   - (참고: IK는 같은 위치라도 다른 자세를 취할 수 있으므로, 이 비율이 낮아도 문제 아님)")

print("-" * 40)
print(f"2. [발 위치 정밀도] ★ 가장 중요 ★")
print(f"   - IK가 찾은 해가 실제로 목표 위치에 닿았는가?")
print(f"   - 평균 위치 오차: {mean_pos_error:.6f} meters")
print(f"   - 최대 위치 오차: {max_pos_error:.6f} meters")

if mean_pos_error < 1e-3:
    print("\n   => ✅ 결론: Solver 정상 동작! (위치 오차가 1mm 미만임)")
else:
    print("\n   => ❌ 결론: Solver 수식 확인 필요.")

print("-" * 40)
print("3. [관절별 평균 각도 차이 (단위: rad)]")
# 이름표 순서를 데이터 순서(Leg-wise)에 맞게 수정함
joint_names_ordered = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

for name, err in zip(joint_names_ordered, mean_error_joints):
    print(f"  - {name:<10}: {err:.6f}")
print("======================================")