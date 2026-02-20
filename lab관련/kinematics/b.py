# import torch
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from tqdm import tqdm

# # FK_IK_Solver.py에서 Go2Solver 클래스를 불러와야 합니다.
# from FK_IK_Solver import Go2Solver

# # --- 1. 설정값 ---
# NUM_POINTS_TO_PLOT = 10000
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # 입력 파일 (이것은 관절 각도 데이터입니다)
# INPUT_FILENAME_PT = 'ik_valid_solutions.pt'

# FOOT_LABELS = ['Front-Left (FL)', 'Front-Right (FR)', 'Rear-Left (RL)', 'Rear-Right (RR)']
# FEET_TO_PLOT = [0, 1, 2, 3] # 모든 다리 표시

# # --- 2. 데이터 불러오기 ---
# print(f"'{INPUT_FILENAME_PT}' 파일에서 유효한 '관절 각도' 데이터를 불러오는 중...")
# try:
#     # 변수 이름을 명확하게 joint_angles로 변경
#     joint_angles = torch.load(INPUT_FILENAME_PT).to(DEVICE)
#     print("데이터 로딩 완료! ✅")
# except FileNotFoundError:
#     print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다.")
#     exit()

# # --- 3. FK 계산 (추가된 핵심 부분) ---
# print("불러온 관절 각도를 실제 발 위치로 변환합니다 (FK 계산)...")
# solver = Go2Solver(device=DEVICE)

# with torch.no_grad():
#     # 데이터를 10,000개씩 작은 배치로 나누어 처리 (메모리 부족 방지)
#     batch_size = 10000
#     foot_positions_list = []
    
#     for i in tqdm(range(0, len(joint_angles), batch_size), desc="FK 계산 중"):
#         batch_joint_angles = joint_angles[i:i+batch_size]
#         batch_foot_positions = solver.go2_fk_new(batch_joint_angles)
#         foot_positions_list.append(batch_foot_positions)
    
#     # 계산된 결과들을 하나의 텐서로 합침
#     foot_positions = torch.cat(foot_positions_list, dim=0)

# print("FK 계산 완료! ✅\n")


# # --- 4. 시각화를 위한 전처리 및 샘플링 ---
# foot_positions_np = foot_positions.cpu().numpy()
# foot_positions_reshaped = foot_positions_np.reshape(-1, 4, 3)

# if len(foot_positions_reshaped) > NUM_POINTS_TO_PLOT:
#     random_indices = np.random.choice(len(foot_positions_reshaped), size=NUM_POINTS_TO_PLOT, replace=False)
#     sampled_positions = foot_positions_reshaped[random_indices]
# else:
#     sampled_positions = foot_positions_reshaped

# # --- 5. Plotly 3D 시각화 ---
# print(f"선택된 다리 { [FOOT_LABELS[i] for i in FEET_TO_PLOT] } 에 대해 시각화를 시작합니다...")
# plot_data = []

# for i in FEET_TO_PLOT:
#     x_coords = sampled_positions[:, i, 0]
#     y_coords = sampled_positions[:, i, 1]
#     z_coords = sampled_positions[:, i, 2]
    
#     trace = go.Scatter3d(
#         x=x_coords, y=y_coords, z=z_coords,
#         mode='markers',
#         marker=dict(size=2, opacity=0.8),
#         name=FOOT_LABELS[i]
#     )
#     plot_data.append(trace)

# origin_marker = go.Scatter3d(
#     x=[0], y=[0], z=[0],
#     mode='markers',
#     marker=dict(size=10, color='black', symbol='diamond'),
#     name='Origin (0,0,0)'
# )
# plot_data.append(origin_marker)

# layout = go.Layout(
#     title='Go2 Robot Foot Workspace (from Valid IK Solutions)',
#     scene=dict(
#         xaxis=dict(title='X-axis (m)'),
#         yaxis=dict(title='Y-axis (m)'),
#         zaxis=dict(title='Z-axis (m)'),
#         aspectmode='data'
#     ),
#     margin=dict(r=20, l=10, b=10, t=40)
# )

# fig = go.Figure(data=plot_data, layout=layout)
# fig.show()

# print("Complete")


# import torch
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from tqdm import tqdm
# import math

# # FK_IK_Solver.py에서 Go2Solver 클래스를 불러와야 합니다.
# from FK_IK_Solver import Go2Solver

# # --- 1. 설정값 ---
# NUM_POINTS_TO_PLOT = 10000
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # 입력 파일 (모든 '유효한' 발 위치가 포함된 원본 데이터)
# INPUT_FILENAME_PT = 'foot_positions_100k.pt'

# FOOT_LABELS = ['Front-Left (FL)', 'Front-Right (FR)', 'Rear-Left (RL)', 'Rear-Right (RR)']
# FEET_TO_PLOT = [0, 1, 2, 3] # 모든 다리 표시

# # 검증을 위한 유효한 관절 제한 (라디안)
# VALID_LIMITS_RAD = {
#     'hip':   (math.radians(-48), math.radians(48)),
#     'thigh': (math.radians(-200), math.radians(90)),
#     'calf':  (math.radians(-156), math.radians(-48))
# }

# # --- 2. 데이터 불러오기 및 IK 재계산 ---
# print(f"'{INPUT_FILENAME_PT}' 파일에서 원본 발 위치 데이터를 불러오는 중...")
# try:
#     foot_positions_flat = torch.load(INPUT_FILENAME_PT).to(DEVICE)
#     print("데이터 로딩 완료! ✅")
# except FileNotFoundError:
#     print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다.")
#     exit()

# print("IK 재계산을 통해 '유효하지 않은' 샘플을 식별합니다...")
# solver = Go2Solver(device=DEVICE)
# with torch.no_grad():
#     batch_size = foot_positions_flat.size(0)
#     foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
#     calculated_joint_angles = solver.go2_ik_new(foot_positions_reshaped)

# # --- 3. '유효하지 않은' 샘플 필터링 (핵심) ---
# # 기존 검증 로직을 그대로 사용하여 '유효한' 샘플 마스크를 생성
# angles_reshaped = calculated_joint_angles.reshape(-1, 3, 4).transpose(1, 2)
# hip_angles, thigh_angles, calf_angles = angles_reshaped[..., 0], angles_reshaped[..., 1], angles_reshaped[..., 2]
# nan_samples_mask = torch.isnan(calculated_joint_angles).any(dim=1)
# valid_hip_mask = (hip_angles >= VALID_LIMITS_RAD['hip'][0]) & (hip_angles <= VALID_LIMITS_RAD['hip'][1])
# valid_thigh_mask = (thigh_angles >= VALID_LIMITS_RAD['thigh'][0]) & (thigh_angles <= VALID_LIMITS_RAD['thigh'][1])
# valid_calf_mask = (calf_angles >= VALID_LIMITS_RAD['calf'][0]) & (calf_angles <= VALID_LIMITS_RAD['calf'][1])
# all_joints_valid_mask = valid_hip_mask.all(dim=1) & valid_thigh_mask.all(dim=1) & valid_calf_mask.all(dim=1)
# truly_valid_samples_mask = all_joints_valid_mask & (~nan_samples_mask)

# # '유효한' 샘플 마스크를 반전(~)시켜 '유효하지 않은' 샘플 마스크를 생성
# invalid_samples_mask = ~truly_valid_samples_mask

# # '유효하지 않은' 발 위치 데이터만 필터링
# invalid_foot_positions = foot_positions_flat[invalid_samples_mask]
# print(f"총 {len(invalid_foot_positions)}개의 '유효하지 않은' 샘플을 식별했습니다. ✅\n")


# # --- 4. 시각화를 위한 전처리 및 샘플링 ---
# foot_positions_np = invalid_foot_positions.cpu().numpy()
# foot_positions_reshaped = foot_positions_np.reshape(-1, 4, 3)

# if len(foot_positions_reshaped) > NUM_POINTS_TO_PLOT:
#     random_indices = np.random.choice(len(foot_positions_reshaped), size=NUM_POINTS_TO_PLOT, replace=False)
#     sampled_positions = foot_positions_reshaped[random_indices]
# else:
#     sampled_positions = foot_positions_reshaped

# # --- 5. Plotly 3D 시각화 ---
# print(f"'유효하지 않은' 샘플의 작업 공간 시각화를 시작합니다...")
# plot_data = []

# for i in FEET_TO_PLOT:
#     x_coords = sampled_positions[:, i, 0]
#     y_coords = sampled_positions[:, i, 1]
#     z_coords = sampled_positions[:, i, 2]
    
#     trace = go.Scatter3d(
#         x=x_coords, y=y_coords, z=z_coords,
#         mode='markers',
#         marker=dict(size=2, opacity=0.8),
#         name=FOOT_LABELS[i]
#     )
#     plot_data.append(trace)

# origin_marker = go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='black', symbol='diamond'), name='Origin (0,0,0)')
# plot_data.append(origin_marker)

# layout = go.Layout(
#     title='Go2 Robot Foot Workspace (INVALID IK Solutions)',
#     scene=dict(xaxis=dict(title='X-axis (m)'), yaxis=dict(title='Y-axis (m)'), zaxis=dict(title='Z-axis (m)'), aspectmode='data'),
#     margin=dict(r=20, l=10, b=10, t=40)
# )

# fig = go.Figure(data=plot_data, layout=layout)
# fig.show()

# print("Complete")

# import torch
# import numpy as np
# import plotly.graph_objects as go
# from tqdm import tqdm
# import math

# # FK_IK_Solver.py에서 Go2Solver 클래스를 불러와야 합니다.
# from FK_IK_Solver import Go2Solver

# # --- 1. 설정값 ---
# NUM_POINTS_TO_PLOT_PER_GROUP = 10000 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# INPUT_FILENAME_PT = 'foot_positions_100k.pt'

# # 검증을 위한 유효한 관절 제한 (라디안)
# VALID_LIMITS_RAD = {
#     'hip':   (math.radians(-48), math.radians(48)),
#     'thigh': (math.radians(-200), math.radians(90)),
#     'calf':  (math.radians(-156), math.radians(-48))
# }

# # --- 2. 데이터 불러오기 및 유효/무효 샘플 분리 ---
# print(f"'{INPUT_FILENAME_PT}' 파일에서 원본 발 위치 데이터를 불러오는 중...")
# try:
#     foot_positions_flat = torch.load(INPUT_FILENAME_PT).to(DEVICE)
#     print("데이터 로딩 완료! ✅")
# except FileNotFoundError:
#     print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다.")
#     exit()

# print("IK 재계산을 통해 '유효'/'유효하지 않은' 샘플을 식별합니다...")
# solver = Go2Solver(device=DEVICE)
# with torch.no_grad():
#     batch_size = foot_positions_flat.size(0)
#     foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
#     calculated_joint_angles = solver.go2_ik_new(foot_positions_reshaped)

# # '유효한' 샘플 마스크 생성
# angles_reshaped = calculated_joint_angles.reshape(-1, 3, 4).transpose(1, 2)
# hip_angles, thigh_angles, calf_angles = angles_reshaped[..., 0], angles_reshaped[..., 1], angles_reshaped[..., 2]
# nan_samples_mask = torch.isnan(calculated_joint_angles).any(dim=1)
# valid_hip_mask = (hip_angles >= VALID_LIMITS_RAD['hip'][0]) & (hip_angles <= VALID_LIMITS_RAD['hip'][1])
# valid_thigh_mask = (thigh_angles >= VALID_LIMITS_RAD['thigh'][0]) & (thigh_angles <= VALID_LIMITS_RAD['thigh'][1])
# valid_calf_mask = (calf_angles >= VALID_LIMITS_RAD['calf'][0]) & (calf_angles <= VALID_LIMITS_RAD['calf'][1])
# all_joints_valid_mask = valid_hip_mask.all(dim=1) & valid_thigh_mask.all(dim=1) & valid_calf_mask.all(dim=1)
# truly_valid_samples_mask = all_joints_valid_mask & (~nan_samples_mask)

# # 마스크를 사용해 두 그룹으로 데이터 분리
# valid_foot_positions = foot_positions_flat[truly_valid_samples_mask]
# invalid_foot_positions = foot_positions_flat[~truly_valid_samples_mask]

# # --- 3. 데이터 그룹 요약 출력 (추가된 부분) ---
# num_valid = len(valid_foot_positions)
# num_invalid = len(invalid_foot_positions)

# print("\n---------- 데이터 그룹 요약 ----------")
# print(f"유효한 IK 해가 나온 샘플: {num_valid} 개")
# print(f"유효하지 않은 IK 해가 나온 샘플: {num_invalid} 개")
# print("-------------------------------------\n")


# # --- 4. 시각화를 위한 데이터 준비 ---
# plot_data = []
# groups = {
#     f"Valid IK Solutions ({num_valid} points)": (valid_foot_positions, 'blue'),
#     f"Invalid IK Solutions ({num_invalid} points)": (invalid_foot_positions, 'red')
# }

# for name, (data_tensor, color) in groups.items():
#     if len(data_tensor) == 0:
#         continue

#     positions_np = data_tensor.cpu().numpy().reshape(-1, 4, 3)
#     if len(positions_np) > NUM_POINTS_TO_PLOT_PER_GROUP:
#         random_indices = np.random.choice(len(positions_np), size=NUM_POINTS_TO_PLOT_PER_GROUP, replace=False)
#         sampled_positions = positions_np[random_indices]
#     else:
#         sampled_positions = positions_np
    
#     x_coords, y_coords, z_coords = sampled_positions[..., 0].flatten(), sampled_positions[..., 1].flatten(), sampled_positions[..., 2].flatten()
    
#     trace = go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers', marker=dict(size=2, opacity=0.6, color=color), name=name)
#     plot_data.append(trace)


# # --- 5. Plotly 3D 시각화 ---
# print("두 그룹의 작업 공간을 함께 시각화합니다...")
# origin_marker = go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='black', symbol='diamond'), name='Origin (0,0,0)')
# plot_data.append(origin_marker)

# layout = go.Layout(
#     title='Go2 Robot Foot Workspace (Valid vs. Invalid IK Solutions)',
#     scene=dict(xaxis=dict(title='X-axis (m)'), yaxis=dict(title='Y-axis (m)'), zaxis=dict(title='Z-axis (m)'), aspectmode='data'),
#     margin=dict(r=20, l=10, b=10, t=40),
#     legend=dict(x=0, y=1)
# )

# fig = go.Figure(data=plot_data, layout=layout)
# fig.show()

# print("Complete")

import torch
import math
import numpy as np

# FK_IK_Solver.py에서 Go2Solver 클래스를 불러와야 합니다.
from FK_IK_Solver import Go2Solver

# --- 1. 설정값 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 검증을 위한 유효한 관절 제한 (라디안)
VALID_LIMITS_RAD = {
    'hip':   (math.radians(-48), math.radians(48)),
    'thigh': (math.radians(-200), math.radians(90)),
    'calf':  (math.radians(-156), math.radians(-48))
}

print(f"사용 장치: {DEVICE}")

# --- 2. 테스트 데이터 수동 생성 ---
# (1, 12) 모양의 텐서를 -10.0으로 채워서 생성
test_value = -10.0
foot_positions_flat = torch.full((1, 12), test_value, dtype=torch.float32).to(DEVICE)
num_samples = len(foot_positions_flat)
print(f"\n테스트할 입력 데이터:\n{foot_positions_flat.cpu().numpy()}")
print(f"총 검증할 샘플 개수: {num_samples}\n")


# --- 3. IK 계산 ---
print("역방향 기구학(IK) 계산을 시작합니다...")
solver = Go2Solver(device=DEVICE)
with torch.no_grad():
    batch_size = foot_positions_flat.size(0)
    foot_positions_reshaped = foot_positions_flat.view(batch_size, 4, 3)
    calculated_joint_angles = solver.go2_ik_new(foot_positions_reshaped)
print(f"IK 계산 결과 (관절 각도):\n{calculated_joint_angles.cpu().numpy()}")
print("IK 계산 완료! ✅\n")


# --- 4. 결과 검증 ---
print("IK 계산 결과 검증을 시작합니다...")
angles_reshaped = calculated_joint_angles.reshape(-1, 3, 4).transpose(1, 2)
hip_angles, thigh_angles, calf_angles = angles_reshaped[..., 0], angles_reshaped[..., 1], angles_reshaped[..., 2]

nan_samples_mask = torch.isnan(calculated_joint_angles).any(dim=1)
num_nan_samples = nan_samples_mask.sum().item()

# NaN이 아닌 경우에만 limit 체크를 수행
if num_nan_samples == 0:
    valid_hip_mask = (hip_angles >= VALID_LIMITS_RAD['hip'][0]) & (hip_angles <= VALID_LIMITS_RAD['hip'][1])
    valid_thigh_mask = (thigh_angles >= VALID_LIMITS_RAD['thigh'][0]) & (thigh_angles <= VALID_LIMITS_RAD['thigh'][1])
    valid_calf_mask = (calf_angles >= VALID_LIMITS_RAD['calf'][0]) & (calf_angles <= VALID_LIMITS_RAD['calf'][1])
    all_joints_valid_mask = valid_hip_mask.all(dim=1) & valid_thigh_mask.all(dim=1) & valid_calf_mask.all(dim=1)
    truly_valid_samples_mask = all_joints_valid_mask
else:
    # NaN이 있으면 무조건 유효하지 않음
    truly_valid_samples_mask = torch.tensor([False], device=DEVICE)

num_limit_violated_samples = num_samples - num_nan_samples - truly_valid_samples_mask.sum().item()


# --- 5. 최종 결과 출력 ---
print("\n---------- IK 검증 결과 ----------")
print(f"총 샘플 수: {num_samples}")
print("-" * 34)
print(f"IK 해가 없어 'NaN'이 발생한 샘플: {num_nan_samples} 개")
print(f"관절 제한(Joint Limit)을 벗어난 샘플: {num_limit_violated_samples} 개")
print(f"예상과 달리 유효한 해가 나온 샘플: {truly_valid_samples_mask.sum().item()} 개")
print("------------------------------------")
print("\nComplete")