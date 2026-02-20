import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. 로봇 다리 설정 ---
L_THIGH = 0.213  # 허벅지 링크 길이 (m)
L_CALF = 0.213   # 종아리 링크 길이 (m)

# 엉덩이(hip) 관절은 원점 (0,0)에 고정
hip_pos = (0, 0)

# 목표 발 위치 (Target Foot Position)
# 이 값을 바꿔서 테스트해볼 수 있습니다.
target_foot_pos = (0.05, -0.3)


# --- 2. 간단한 2D IK 솔버 ---
def solve_2d_ik(target_pos):
    """
    주어진 목표 위치에 도달하는 두 가지 가능한 관절 각도를 계산합니다.
    """
    px, py = target_pos
    
    # 코사인 법칙을 사용하여 무릎 각도(theta2) 계산
    # (px^2 + py^2 - L1^2 - L2^2) / (2 * L1 * L2)
    cos_theta2_arg = (px**2 + py**2 - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    
    # 계산된 값이 [-1, 1] 범위를 벗어나면 도달 불가능
    if not (-1 <= cos_theta2_arg <= 1):
        print("도달 불가능한 목표 지점입니다.")
        return None, None

    # --- 두 가지 해답이 갈라지는 지점 ---
    # 해답 1: 무릎이 뒤로 꺾이는 '자연스러운' 자세 (-)
    theta2_sol1 = -math.acos(cos_theta2_arg) # Go2 로봇과 같은 방식
    
    # 해답 2: 무릎이 앞으로 꺾이는 '어색한' 자세 (+)
    theta2_sol2 = math.acos(cos_theta2_arg)  # 사람과 같은 방식

    # 각 해답에 대한 허벅지 각도(theta1) 계산
    k1 = L_THIGH + L_CALF * math.cos(theta2_sol1)
    k2 = L_CALF * math.sin(theta2_sol1)
    theta1_sol1 = math.atan2(py, px) - math.atan2(k2, k1)
    
    k1 = L_THIGH + L_CALF * math.cos(theta2_sol2)
    k2 = L_CALF * math.sin(theta2_sol2)
    theta1_sol2 = math.atan2(py, px) - math.atan2(k2, k1)
    
    solution1 = (theta1_sol1, theta2_sol1)
    solution2 = (theta1_sol2, theta2_sol2)
    
    return solution1, solution2


# --- 3. 두 가지 해답 시각화 ---
def plot_solutions(sol1, sol2, target_pos):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 해답 1번 플로팅 (자연스러운 자세)
    th1, th2 = sol1
    knee1_pos = (L_THIGH * np.cos(th1), L_THIGH * np.sin(th1))
    foot1_pos = (knee1_pos[0] + L_CALF * np.cos(th1 + th2), knee1_pos[1] + L_CALF * np.sin(th1 + th2))
    ax.plot([hip_pos[0], knee1_pos[0]], [hip_pos[1], knee1_pos[1]], 'b-', linewidth=4, label='Solution 1 (Natural Pose)')
    ax.plot([knee1_pos[0], foot1_pos[0]], [knee1_pos[1], foot1_pos[1]], 'b-', linewidth=4)
    
    # 해답 2번 플로팅 (어색한 자세)
    th1_2, th2_2 = sol2
    knee2_pos = (L_THIGH * np.cos(th1_2), L_THIGH * np.sin(th1_2))
    foot2_pos = (knee2_pos[0] + L_CALF * np.cos(th1_2 + th2_2), knee2_pos[1] + L_CALF * np.sin(th1_2 + th2_2))
    ax.plot([hip_pos[0], knee2_pos[0]], [hip_pos[1], knee2_pos[1]], 'r--', linewidth=3, label='Solution 2 (Unnatural Pose)')
    ax.plot([knee2_pos[0], foot2_pos[0]], [knee2_pos[1], foot2_pos[1]], 'r--', linewidth=3)
    
    # 관절 및 목표 지점 표시
    ax.plot(hip_pos[0], hip_pos[1], 'ko', markersize=10, label='Hip Joint')
    ax.plot(knee1_pos[0], knee1_pos[1], 'bo', markersize=8)
    ax.plot(knee2_pos[0], knee2_pos[1], 'ro', markersize=8)
    ax.plot(target_pos[0], target_pos[1], 'gx', markersize=15, mew=3, label='Target Foot Position')
    
    # 각도 정보 텍스트로 표시 (단위: 도)
    ax.text(0.1, 0.1, f"Solution 1 (Blue):\nThigh: {math.degrees(th1):.1f}°\nKnee: {math.degrees(th2):.1f}°", transform=ax.transAxes)
    ax.text(0.1, 0.0, f"Solution 2 (Red):\nThigh: {math.degrees(th1_2):.1f}°\nKnee: {math.degrees(th2_2):.1f}°", transform=ax.transAxes)

    # 그래프 설정
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Two IK Solutions for a Single Target Position")
    ax.legend()
    plt.show()

# --- 메인 실행 ---
if __name__ == '__main__':
    solution1, solution2 = solve_2d_ik(target_foot_pos)
    if solution1:
        plot_solutions(solution1, solution2, target_foot_pos)