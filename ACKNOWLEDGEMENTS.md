# 来源与贡献范围

模型与算法来自 [Hugging Face LeRobot / SmolVLA](https://github.com/huggingface/lerobot)，基础权重为 [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)，视觉语言模型为 [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)。本项目未提出这些模型、LoRA 或 Flow Matching 算法。

数据来自 [lerobot/libero](https://huggingface.co/datasets/lerobot/libero)，仿真任务、资源与成功判定来自 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，物理仿真由 MuJoCo / robosuite 提供。模型、数据及各依赖的使用条件分别以其原始仓库与模型卡为准。

本仓库的工作包括固定版本与轨迹划分、源元数据适配、训练集归一化统计、默认动作专家与基于官方PEFT的LoRA配置复现、保存并重载 checkpoint、同步向量环境终止冻结修复、实际初值追踪、完整评测视频导出、受控采样步数对照、证据审计、失败案例分析和静态展示页。

上游 LeRobot 固定提交为 `f9e9d739ea7055b282ea1b165e8785412a674693`，在其上明确应用 `patches/` 内三份补丁。LIBERO 资源与提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01` 的585个文件逐一按 Git blob 哈希核对。版本、模型与数据来源见 `docs/data/`，补丁和回归验证见 `patches/`、`tests/`。

保留的上游许可文本：

- [LeRobot — Apache License 2.0](third_party/LeRobot-LICENSE)
- [LIBERO — MIT License](third_party/LIBERO-LICENSE)

本仓库不重新发布预训练权重或完整示范数据。公开视频为本实验产生的 LIBERO 仿真回合。
