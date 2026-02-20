# import torch
# import numpy as np
# import plotly.graph_objects as go

# def visualize_fast_3d(normal_file, overlimit_file, downsample_ratio=1):
#     print("데이터 로딩 중...")

#     # --- 데이터 로드 및 전처리 함수 (아까와 동일) ---
#     def get_points(path):
#         try:
#             data = torch.load(path)
#             if isinstance(data, torch.Tensor):
#                 data = data.cpu().detach().numpy()
            
#             # (N, 12) -> (N*4, 3) 변환
#             if data.shape[-1] == 12:
#                 print(f"  -> {path}: (N, 12) 형태를 3D 좌표로 변환함.")
#                 return data.reshape(-1, 3)
#             elif data.shape[-1] == 3:
#                 return data.reshape(-1, 3)
#             else:
#                 print(f"Shape Error: {data.shape}")
#                 return None
#         except Exception as e:
#             print(f"File Error: {e}")
#             return None

#     # 데이터 가져오기
#     p_norm = get_points(normal_file)
#     p_over = get_points(overlimit_file)

#     # --- 데이터가 너무 많으면 브라우저도 느려질 수 있으니 샘플링 (옵션) ---
#     # downsample_ratio=1이면 전체 다 표시, 10이면 1/10만 표시
#     if p_norm is not None and downsample_ratio > 1:
#         print(f"  -> 성능을 위해 Normal 데이터를 1/{downsample_ratio}로 줄여서 표시합니다.")
#         p_norm = p_norm[::downsample_ratio]

#     print("웹 브라우저용 그래프 생성 중...")
    
#     fig = go.Figure()

#     # 1. Normal 데이터 (파란색, 작고 투명하게)
#     if p_norm is not None:
#         fig.add_trace(go.Scatter3d(
#             x=p_norm[:, 0], y=p_norm[:, 1], z=p_norm[:, 2],
#             mode='markers',
#             name='Normal Range',
#             marker=dict(
#                 size=1.5,       # 점 크기 (작게)
#                 color='blue', 
#                 opacity=1.0     # 투명도 (0.1 ~ 0.2 추천)
#             )
#         ))

#     # 2. Overlimit 데이터 (빨간색, 진하게)
#     if p_over is not None:
#         fig.add_trace(go.Scatter3d(
#             x=p_over[:, 0], y=p_over[:, 1], z=p_over[:, 2],
#             mode='markers',
#             name='Overlimit',
#             marker=dict(
#                 size=3,         # 점 크기 (조금 더 크게)
#                 color='red',
#                 symbol='cross', # X 모양 (diamond, circle, cross 등 가능)
#                 opacity=1.0
#             )
#         ))

#     # 레이아웃 설정
#     fig.update_layout(
#         title="3D Foot Positions (Plotly Visualization)",
#         scene=dict(
#             xaxis_title='X Position',
#             yaxis_title='Y Position',
#             zaxis_title='Z Position'
#         ),
#         margin=dict(l=0, r=0, b=0, t=40)  # 여백 줄이기
#     )

#     print("완료! 브라우저를 확인하세요.")
#     fig.show()

# # --- 실행 ---
# if __name__ == "__main__":
#     visualize_fast_3d(
#         normal_file="samples_normal_foot_positions.pt",
#         overlimit_file="samples_overlimit_foot_positions.pt",
#         downsample_ratio=5  # 팁: Normal 데이터가 너무 많으면 5나 10으로 설정해보세요
#     )


import torch
import numpy as np
import plotly.graph_objects as go

def visualize_separate_3d(normal_file, overlimit_file, downsample_ratio=1):
    print("데이터 로딩 중...")

    # --- 데이터 로드 및 전처리 함수 ---
    def get_points(path):
        try:
            data = torch.load(path)
            if isinstance(data, torch.Tensor):
                data = data.cpu().detach().numpy()
            
            # (N, 12) -> (N*4, 3) 변환
            if data.shape[-1] == 12:
                # print(f"  -> {path}: (N, 12) 형태를 3D 좌표로 변환함.")
                return data.reshape(-1, 3)
            elif data.shape[-1] == 3:
                return data.reshape(-1, 3)
            else:
                print(f"Shape Error: {data.shape}")
                return None
        except Exception as e:
            print(f"File Error: {e}")
            return None

    # 데이터 가져오기
    p_norm = get_points(normal_file)
    p_over = get_points(overlimit_file)

    # --- 데이터 샘플링 ---
    if p_norm is not None and downsample_ratio > 1:
        print(f"  -> 성능을 위해 Normal 데이터를 1/{downsample_ratio}로 줄여서 표시합니다.")
        p_norm = p_norm[::downsample_ratio]

    print("웹 브라우저용 그래프 생성 중...")
    
    # ==========================================
    # 1. 첫 번째 Figure: Normal 데이터 (정상 범위)
    # ==========================================
    if p_norm is not None:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter3d(
            x=p_norm[:, 0], y=p_norm[:, 1], z=p_norm[:, 2],
            mode='markers',
            name='Normal Range',
            marker=dict(
                size=1.5,       
                color='blue', 
                opacity=0.5     # 겹침을 보기 위해 약간 투명하게 설정
            )
        ))
        
        fig1.update_layout(
            title="[Figure 1] Normal Range (Valid Points)",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        print("1. Normal 데이터 그래프 출력...")
        fig1.show()
    else:
        print("Normal 데이터가 없습니다.")

    # ==========================================
    # 2. 두 번째 Figure: Overlimit 데이터 (리미트 초과)
    # ==========================================
    if p_over is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter3d(
            x=p_over[:, 0], y=p_over[:, 1], z=p_over[:, 2],
            mode='markers',
            name='Overlimit',
            marker=dict(
                size=2,         
                color='red',
                symbol='cross', 
                opacity=0.8
            )
        ))

        fig2.update_layout(
            title="[Figure 2] Overlimit (Invalid Points)",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        print("2. Overlimit 데이터 그래프 출력...")
        fig2.show()
    else:
        print("Overlimit 데이터가 없습니다.")

    print("완료! 두 개의 그래프를 확인하세요.")

# --- 실행 ---
if __name__ == "__main__":
    visualize_separate_3d(
        normal_file="samples_normal_foot_positions.pt",
        overlimit_file="samples_overlimit_foot_positions.pt",
        downsample_ratio=5  # 필요에 따라 조절하세요
    )