import torch
import numpy as np
import math
from tqdm import tqdm

from FK_IK_Solver import Go2Solver

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- CHANGED: Updated filename to reflect .pt format ---
INPUT_FILENAME_PT = 'ik_valid_foot_positions.pt'  # <-- 여기에 사용할 .pt 파일 이름을 입력하세요.

TOLERANCE = 1e-5


try:
    # --- CHANGED: Switched from np.loadtxt to torch.load for .pt files ---
    # The original NumPy loading and conversion is no longer needed.
    # q_original_np = np.loadtxt(INPUT_FILENAME_TXT)
    # q_original = torch.from_numpy(q_original_np).float().to(DEVICE)
    
    # This single line loads the .pt file directly as a tensor.
    q_original = torch.load(INPUT_FILENAME_PT).float().to(DEVICE)
    num_samples = len(q_original)

except FileNotFoundError:
    # --- CHANGED: Updated the error message for the new filename ---
    print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다.")
    exit()

solver = Go2Solver(device=DEVICE)

# The rest of your validation logic remains the same.
with torch.no_grad():
    foot_positions_flat = solver.go2_fk_new(q_original)
    batch_size = foot_positions_flat.size(0)
    foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
    q_calculated = solver.go2_ik_new(foot_positions_reshaped)


abs_error = torch.abs(q_original - q_calculated)

perfect_match_mask = torch.all(torch.isclose(q_original, q_calculated, atol=TOLERANCE), dim=1)
num_perfect_matches = perfect_match_mask.sum().item()

max_error = abs_error.max().item()
mean_error = abs_error.mean().item()
mean_error_per_joint = abs_error.mean(dim=0) 

print("---------- FK-IK 왕복 변환 검증 결과 ----------")
print(f"총 샘플 수: {num_samples}")
print(f"허용 오차 ({TOLERANCE:.0e} rad) 내 완벽 복원 샘플: {num_perfect_matches} 개 ({(num_perfect_matches/num_samples)*100:.2f} %)")
print("-" * 47)
print(f"전체 오차 통계 (단위: 라디안)")
print(f"  - 평균 오차: {mean_error:.6f}")
print(f"  - 최대 오차: {max_error:.6f}")
print("-" * 47)
print("관절별 평균 오차 (단위: 라디안)")

joint_names_ordered = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf"
]
for name, err in zip(joint_names_ordered, mean_error_per_joint):
    print(f"  - {name:<10}: {err:.6f}")
print("-------------------------------------------------")