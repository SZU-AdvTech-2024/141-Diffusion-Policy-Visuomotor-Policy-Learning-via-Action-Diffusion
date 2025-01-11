"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# 使用行缓冲模式（line-buffering）为标准输出和标准错误进行缓冲
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 允许在配置文件中使用 ${eval:''} 来执行任意 Python 代码
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 立即解析配置，以确保所有使用 ${now:} 的解析器在同一时间生效
    OmegaConf.resolve(cfg)

    # 从配置中获取类路径并加载对应类
    cls = hydra.utils.get_class(cfg._target_)
    # 实例化工作空间（workspace），传入解析后的配置
    workspace: BaseWorkspace = cls(cfg)
    # 运行工作空间中的核心逻辑
    workspace.run()

if __name__ == "__main__":
    main()
