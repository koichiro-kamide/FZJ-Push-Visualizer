import os
import matplotlib.pyplot as plt
from fairmotion.data import bvh
import imageio.v2 as imageio
from io import BytesIO
from PIL import Image

def load_skl():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(root_dir, 'single-person', 'data', 'raw_data', 'indiv_data')
    print(f'dataPath: {dataPath}')

    sub_names = ['X05', 'X07', 'X08', 'X09']
    motion_set = {}

    for sub_name in sub_names:
        seqs_path = os.path.join(dataPath, sub_name)
        print(f'Loading... {seqs_path}')
        seqs_list = os.listdir(seqs_path)
        motion_list = []
        for seq in seqs_list:
            seq_bvh_path = os.path.join(seqs_path, seq)
            motion = bvh.load(seq_bvh_path)
            positions = motion.positions(local=False)
            positions_scale = positions / 1000
            motion_list.append(positions_scale)
        motion_set[sub_name] = motion_list
    
    return motion_set


def visualize_motion(motion, sub_name="X05", out_name="motion.gif"):
    """
    motion: ndarray (frames, joints, 3)
    """
    parents = [
        -1, 0, 1, 2, 3, 4, 4, 6, 7, 8,
        4, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20
    ]

    images = []
    os.makedirs('visual_results', exist_ok=True)

    for f_idx, frame in enumerate(motion):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        # ノードを点で表示
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='b', s=20)

        # 親子関係に基づいてエッジを描く
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx == -1:
                continue
            p1, p2 = frame[child_idx], frame[parent_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)

        # 各ノードにindexを表示
        for i, (x, y, z) in enumerate(frame):
            ax.text(x, y, z, str(i), color='black', fontsize=7, ha='center', va='bottom')

        # 軸スケールをそろえる
        max_range = (frame.max(axis=0) - frame.min(axis=0)).max() / 2.0
        mid = frame.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.view_init(elev=16, azim=-10)
        ax.set_axis_off()
        ax.set_title(f"{sub_name} - frame {f_idx}")

        # --- 画像をバッファ経由で保存 ---
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close(fig)

    # --- GIFとして保存 ---
    out_path = os.path.join("visual_results", out_name)
    imageio.mimsave(out_path, images, fps=15)
    print(f"✅ GIF saved to: {out_path}")


if __name__ == '__main__':
    motion_set = load_skl()

    sub_name = 'X05'
    sample_idx = 0
    motion = motion_set[sub_name][sample_idx]
    visualize_motion(motion, sub_name=sub_name, out_name=f"{sub_name}_sample{sample_idx}.gif")
