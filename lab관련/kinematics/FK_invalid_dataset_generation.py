import torch
import math
import random
import numpy as np 
from tqdm import tqdm

from FK_IK_Solver import Go2Solver

NUM_SAMPLES = 100_000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_FILENAME_PT = 'foot_positions_invalid_100k.pt'
OUTPUT_FILENAME_TXT = 'foot_positions_invalid_100k.txt'

VALID_LIMITS_RAD = {
    'hip':   [math.radians(-48), math.radians(48)],
    'thigh': [math.radians(-200), math.radians(90)],
    'calf':  [math.radians(-156), math.radians(-48)]
}

VIOLATION_OFFSET_MIN = 0.1
VIOLATION_OFFSET_MAX = 1.0



def generate_invalid_angle():
    
    angles = [
        random.uniform(*VALID_LIMITS_RAD['hip']),
        random.uniform(*VALID_LIMITS_RAD['thigh']),
        random.uniform(*VALID_LIMITS_RAD['calf'])
    ]
    joint_to_violate = random.randint(0, 2)
    limit_to_violate = random.randint(0, 1)
    offset = random.uniform(VIOLATION_OFFSET_MIN, VIOLATION_OFFSET_MAX)
    joint_name = ['hip', 'thigh', 'calf'][joint_to_violate]
    
    if limit_to_violate == 0:
        angles[joint_to_violate] = VALID_LIMITS_RAD[joint_name][0] - offset
    else:
        angles[joint_to_violate] = VALID_LIMITS_RAD[joint_name][1] + offset
        
    return angles

invalid_single_leg_angles = [generate_invalid_angle() for _ in tqdm(range(NUM_SAMPLES), desc="샘플링 중")]

single_leg_tensor = torch.tensor(invalid_single_leg_angles, dtype=torch.float32)

hip_block = single_leg_tensor[:, 0:1].repeat(1, 4)
thigh_block = single_leg_tensor[:, 1:2].repeat(1, 4)
calf_block = single_leg_tensor[:, 2:3].repeat(1, 4)
invalid_joint_angles_data = torch.cat([hip_block, thigh_block, calf_block], dim=1).to(DEVICE)

solver = Go2Solver(device=DEVICE)

with torch.no_grad():
    invalid_foot_positions = solver.go2_fk_new(invalid_joint_angles_data)



torch.save(invalid_foot_positions, OUTPUT_FILENAME_PT)

np.savetxt(OUTPUT_FILENAME_TXT, invalid_foot_positions.cpu().numpy(), fmt='%.6f')
print("Complete")