import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame


@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(output, render_size, control_hz):
    """
    采集推T任务的演示。

    使用方式: python demo_pusht.py -o data/pusht_demo.zarr

    此脚本兼容Linux和MacOS。
    将鼠标悬停靠近蓝色圆圈以开始。
    将T型块推入绿色区域。
    如果任务成功，剧集将自动终止。
    按“Q”退出。
    按“R”重试。
    按住“Space”键暂停。
    """

    # 在读写模式下创建回放缓冲区
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # 创建带有关键点的推T环境
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # 一级循环
    while True:
        episode = list()
        # 按顺序记录种子，从0开始
        seed = replay_buffer.n_episodes
        print(f'开始种子 {seed}')

        # 为环境设置种子
        env.seed(seed)

        # 重置环境并获取观测值（包括用于记录的信息和渲染图像）
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        # 循环状态
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'计划指数:{plan_idx}')
        # 二级循环
        while not done:
            # 处理按键事件
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # 按住空格键暂停
                        plan_idx += 1
                        pygame.display.set_caption(f'计划指数:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # 按“R”重试
                        retry = True
                    elif event.key == pygame.K_q:
                        # 按“Q”退出
                        exit(0)
                if event.type is pygame.KEYUP:
                    if event.key is pygame.K_SPACE:
                        pause = False

            # 处理控制流
            if retry:
                break
            if pause:
                continue

            # 从鼠标获取动作
            # 如果鼠标不靠近代理，则为None
            act = agent.act(obs)
            if act is not None:
                # 开始远程操作
                # 状态维度为2+3
                state = np.concatenate([info['pos_agent'], info['block_pose']])
                # 丢弃不用的信息如可见性掩码和代理位置
                # 以保持兼容性
                keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)

            # 运行环境步骤并渲染
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            # 调整控制频率
            clock.tick(control_hz)
        if not retry:
            # 保存剧集缓冲到回放缓冲区（磁盘上）
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'已保存种子 {seed}')
        else:
            print(f'重试种子 {seed}')


if __name__ == "__main__":
    main()
