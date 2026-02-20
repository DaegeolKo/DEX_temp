import torch

# 확인할 파일 이름
filename = 'generated_feet.pt'

try:
    # 파일을 불러옵니다.
    data = torch.load(filename)
    
    print(f"'{filename}' 파일 로딩 성공! ✅")
    print(f"---------------------------------")
    print(f"텐서의 전체 모양 (Shape): {data.shape}")
    print(f"텐서의 데이터 타입 (dtype): {data.dtype}")
    print(f"텐서가 사용하는 장치 (Device): {data.device}")
    print(f"\n--- 첫 3개 샘플 데이터 ---")
    print(data[:3])
    print("-------------------------")

except Exception as e:
    print(f"'{filename}' 파일 로딩 중 오류 발생: {e}")