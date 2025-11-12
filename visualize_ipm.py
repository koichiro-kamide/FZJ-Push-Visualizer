import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fairmotion.data import bvh
import imageio.v2 as imageio


def load_skl_ipm():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(root_dir, 'single-person', 'data', 'raw_data', 'indiv_data')
    print(f'dataPath: {dataPath}')

    sub_names = ['X05', 'X07', 'X08', 'X09']
    motion_set = {}
    ipm_set = {}
    for sub_name in sub_names:
        seqs_path = os.path.join(dataPath, sub_name)
        print(f'Loading... {seqs_path}')
        seqs_list = os.listdir(seqs_path)
        motion_list = []
        seqs_ipm_sts_list = []
        for seq in seqs_list:
            seq_bvh_path = os.path.join(seqs_path, seq)
            # 3Dスケルトンデータ
            motion = bvh.load(seq_bvh_path)
            positions = motion.positions(local=False)
            positions_scale = positions / 1000
            motion_list.append(positions_scale)

            # 倒立振り子パラメータの計算
            bvh_data_hip = positions_scale[:, 0, :]
            bvh_data_cart = (positions_scale[:, 16, :] + positions_scale[:, 20, :]) / 2
            rod_direction_norm = np.zeros((bvh_data_hip.shape[0], 1))
            rod_direction_norm[:, 0] = np.linalg.norm((bvh_data_hip - bvh_data_cart), axis=-1)
            rod_direction = (bvh_data_hip - bvh_data_cart) / rod_direction_norm
            phi = np.arcsin(-rod_direction[:, 1:2])
            theta = np.arcsin(rod_direction[:, 0:1] / np.cos(phi))

            seq_ipm_sts1 = np.concatenate((bvh_data_cart[:, :-1], theta, phi, rod_direction_norm,
                                            bvh_data_cart[:, -1:], bvh_data_hip[:, :-1]), axis=1)
            seqs_ipm_sts_list.append(seq_ipm_sts1)

        motion_set[sub_name] = motion_list
        ipm_set[sub_name] = seqs_ipm_sts_list
    
    return motion_set, ipm_set


def visualize_ipm(seq_ipm_sts1, seq_ipm_sts2, sub_name, sample_idx, save_dir="visual_results"):
    """
    seq_ipm_sts: (frames, 8)
    """
    os.makedirs(save_dir, exist_ok=True)
    frames = []
    parents = [
        -1, 0, 1, 2, 3, 4, 4, 6, 7, 8, 4, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20
    ]

    # tqdmで進捗バー表示
    for f_idx in tqdm(range(min(len(seq_ipm_sts1), len(seq_ipm_sts2))), desc=f"Rendering {sub_name}"):
        x_cart1, y_cart1, theta1, phi1, rod_len1, z_cart1, x_hip1, y_hip1 = seq_ipm_sts1[f_idx]
        x_cart2, y_cart2, theta2, phi2, rod_len2, z_cart2, x_hip2, y_hip2 = seq_ipm_sts2[f_idx]

        cart1 = np.array([x_cart1, y_cart1, z_cart1])
        hip1 = np.array([x_hip1, y_hip1, z_cart1 + rod_len1])

        cart2 = np.array([x_cart2, y_cart2, z_cart2])
        hip2 = np.array([x_hip2, y_hip2, z_cart2 + rod_len2])

        fig = plt.figure(figsize=(10, 5))
        elev = 16  # 0
        azim = -30  # -90

        # ---- 倒立振り子モデル ----
        # --- 実際のIPMを表示 (real) ---
        ax = fig.add_subplot(111, projection='3d')
        # 質点を表示
        ax.scatter(hip1[0], hip1[1], hip1[2], 
                    c='r', s=60, label='Point mass (real)', alpha=0.5)
        # ロットを表示
        ax.plot([cart1[0], hip1[0]], 
                 [cart1[1], hip1[1]], 
                 [cart1[2], hip1[2]], 
                 'k-', linewidth=1.5, label='Rod (real)', alpha=0.5)
        # カートを表示
        ax.scatter(cart1[0], cart1[1], cart1[2], 
                    c='b', s=100, marker='s', label='Cart (real)', alpha=0.5)
        
        # --- 予測したIPMを表示 (pred) ---
        # 質点を表示
        ax.scatter(hip2[0], hip2[1], hip2[2], 
                    c='r', s=60, label='Point mass (pred)')
        # ロットを表示
        ax.plot([cart2[0], hip2[0]], 
                 [cart2[1], hip2[1]], 
                 [cart2[2], hip2[2]], 
                 'k-', linewidth=1.5, label='Rod (pred)')
        # カートを表示
        ax.scatter(cart2[0], cart2[1], cart2[2], 
                    c='b', s=100, marker='s', label='Cart (pred)')
        
        # 軸設定
        ax.set_xlim(-0.3, 1.8)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1.8)
        ax.legend(fontsize=16, ncol=2)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')

        tmp_path = f'frame_{f_idx:04d}.png'
        plt.tight_layout()
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames.append(imageio.imread(tmp_path))
        os.remove(tmp_path)

    save_path = os.path.join(save_dir, f'{sub_name}-ipm{sample_idx}.gif')
    imageio.mimsave(save_path, frames, fps=50, loop=0)  # 実際は60Hz
    print(f"✅ Saved {save_path}")


if __name__ == '__main__':
    # 3Dスケルトンデータと倒立振り子モデル（IPM）を準備
    motion_set, ipm_set = load_skl_ipm()

    #=== 1サンプルずつ可視化したい場合 ===
    sub_name = 'X08'
    sample_idx1 = 11
    sample_idx2 = 12
    seq_ipm_sts1 = ipm_set[sub_name][sample_idx1]
    seq_ipm_sts2 = ipm_set[sub_name][sample_idx2]
    visualize_ipm(seq_ipm_sts1, seq_ipm_sts2, sub_name, sample_idx1)
