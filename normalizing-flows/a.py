import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def draw_gaussian_latent(filename="fig3_gaussian_latent.png"):
    # 데이터 생성 (2D 표준 정규 분포)
    np.random.seed(42) # 재현성을 위해 시드 고정
    z_data = np.random.randn(1000, 2)

    # 설정
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # 스타일링
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    font_props = dict(fontsize=14, family='serif')

    # 1. 산점도 (Scatter Plot)
    # 중심부와 외곽부 색상 다르게 하여 밀도감 표현
    distances = np.linalg.norm(z_data, axis=1)
    colors = plt.cm.viridis(1 - distances / np.max(distances)) # 중심일수록 밝게

    ax.scatter(z_data[:, 0], z_data[:, 1], s=15, c=colors, alpha=0.6, edgecolor='none')

    # 2. 등고선 (Contour Plot - 밀도 표현)
    # 2D Gaussian 커널 밀도 추정 (간단히 구현)
    from scipy.stats import multivariate_normal
    x, y = np.mgrid[-3:3:.05, -3:3:.05]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    ax.contour(x, y, rv.pdf(pos), levels=4, colors='k', alpha=0.3, linewidths=1)


    # 3. 축 및 레이블
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"Latent dimension $z_1$", **font_props)
    ax.set_ylabel(r"Latent dimension $z_2$", **font_props)
    ax.set_title(r"Gaussian Latent Space $\mathcal{N}(0, I)$", fontsize=16, family='serif', pad=15)

    # 원점 표시
    ax.axhline(0, color='black', lw=0.5, ls='-')
    ax.axvline(0, color='black', lw=0.5, ls='-')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()

draw_gaussian_latent()