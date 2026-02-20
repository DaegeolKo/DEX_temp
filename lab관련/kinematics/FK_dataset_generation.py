import torch
import numpy as np
from tqdm import tqdm

from FK_IK_Solver_dspace import Go2Solver

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_FILENAME_PT = 'joint_angles_100k_ordered.pt'
INPUT_FILENAME_TXT = 'joint_angles_100k_ordered.txt'

OUTPUT_FILENAME_PT = 'foot_positions_100k.pt'
OUTPUT_FILENAME_TXT = 'foot_positions_100k.txt'


try:
    joint_angles_data = torch.load(INPUT_FILENAME_PT)
    
except FileNotFoundError:
    print(f"오류: '{INPUT_FILENAME_PT}' 파일을 찾을 수 없습니다. 이전 단계에서 파일을 생성했는지 확인하세요.")
    exit()

joint_angles_data = joint_angles_data.to(DEVICE)
print(f"불러온 데이터 shape: {joint_angles_data.shape}\n")


solver = Go2Solver(device=DEVICE)

with torch.no_grad():
    batch_size = 10000
    foot_positions_list = []
    
    for i in tqdm(range(0, len(joint_angles_data), batch_size), desc="FK 계산 중"):
        batch_joint_angles = joint_angles_data[i:i+batch_size]
        batch_foot_positions = solver.go2_fk_new(batch_joint_angles)
        foot_positions_list.append(batch_foot_positions)
    
    foot_positions = torch.cat(foot_positions_list, dim=0)

torch.save(foot_positions, OUTPUT_FILENAME_PT)

np.savetxt(OUTPUT_FILENAME_TXT, foot_positions.cpu().numpy(), fmt='%.6f')
print("complete")