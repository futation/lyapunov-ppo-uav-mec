"""make_demo_gif.py — 录制场景演示 GIF

用贪心策略跑一个 episode, 逐时隙调用 env.render() 抓帧,
输出 README 用的演示动图与首帧静态图。

运行 (在仓库根目录):
    python examples/make_demo_gif.py
输出:
    docs/img/demo.gif, docs/img/demo.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from config import NUM_BEAMS
from env import UAV_HAPMECEnv


def greedy_policy(env, uav_id, mask):
    valid = np.where(mask)[0]
    beam_actions = [a for a in valid if 1 <= a <= NUM_BEAMS]
    if not beam_actions:
        return 0
    uav = env.uavs[uav_id]
    best = min(beam_actions,
               key=lambda a: np.hypot(uav.pos[0] - env.HAP.cluster_queue[a - 1]['center_x'],
                                      uav.pos[1] - env.HAP.cluster_queue[a - 1]['center_y']))
    return int(best)


def main(out_dir='docs/img', slots=24, fps=4):
    os.makedirs(out_dir, exist_ok=True)

    env = UAV_HAPMECEnv(render_mode='rgb_array')
    env.reset(seed=7)

    frames = [Image.fromarray(env.render())]
    for _ in range(slots):
        for uav_id in range(env.num_uavs):
            mask = env.get_action_mask(uav_id)
            env.step(uav_id, greedy_policy(env, uav_id, mask))
        env.end_of_step()
        frames.append(Image.fromarray(env.render()))

    gif_path = os.path.join(out_dir, 'demo.gif')
    png_path = os.path.join(out_dir, 'demo.png')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    frames[len(frames) // 2].save(png_path)
    print(f"[OK] {gif_path}  ({len(frames)} frames)")
    print(f"[OK] {png_path}")


if __name__ == '__main__':
    main()
