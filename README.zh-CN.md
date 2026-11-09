# CUHK-X 小模型赛道 — 紧凑视觉与时序融合

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供 CUHK-X Challenge 2026 小模型赛道决赛二等奖团队 **YE Mao1023** 的代码和技术报告。提交系统结合 Depth_Color、红外视频特征与骨架时序识别。Kaggle 提交 **56036959** 的公开榜准确率为 **97.512%**，私有榜为 **96.568%**；冻结推理包（含检测器）为 **50,733,469 字节**。

## 技术报告

[技术报告 PDF](docs/public/TECHNICAL_REPORT.pdf)

报告介绍视觉分支、骨骼帧分类器、残差时序解码器及置信度融合。模型来源见**第 4 节**，训练与验证设置见**第 5 节**，发布内容与推理要求见**第 8 节**。

本仓库发布训练与推理源码、配置和技术报告，不附模型权重及比赛数据。推理需另行准备 `final_config.json` 对应的权重包；初始化要求及历史复现记录详见技术报告。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `inference.sh`、`final_config.json` | 提交模型的推理入口与产物标识 |
| `visual_temporal/data/` | 索引、预处理与缓存 |
| `visual_temporal/models/` | 视觉、骨架网络与共享融合组件 |
| `visual_temporal/training/` | 视觉适配与时序解码器训练 |
| `visual_temporal/release/` | 分支推理、融合、打包与原始输入重放 |
| `docs/public/TECHNICAL_REPORT.pdf` | 方法、历史结果、来源及复现设置 |

## 安装与推理

记录中的 Linux GPU 环境使用 Python 3.11 和 CUDA 12.9 软件包：

```bash
python -m pip install -r requirements-inference.txt
python -m pip install -e . --no-deps
```

通过 [Small Model Track 数据集](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track)取得比赛数据。下载脚本另需通过 `python -m pip install -U huggingface_hub` 安装 CLI。接受数据集访问条件、执行 `hf auth login` 后，运行 `bash download_all_data.sh` 将压缩包下载至 `data/Small-Model-Track/`。

将对应权重包放到 `final_config.json` 指定位置后，准备如下输入：

```text
Testing/
  test_file/test.csv
  test_file/sample_submission.csv
  data/small_model_track_test/<sample_id>/...
```

也接受官方 `Testing/data/small_model_track_test.zip` 格式。

```bash
bash inference.sh /absolute/path/to/Testing /absolute/path/to/prediction.csv
```

输出路径必须位于输入数据目录之外。`CUHKX_PYTHON` 指定 Python，`CUHKX_DEVICE` 指定 CUDA 设备，默认 `cuda:0`。入口在处理输入前校验权重文件大小。

## 训练实现

训练需要授权训练数据和原始初始化。将 `HAR.z01`–`HAR.z08` 与 `HAR.zip` 放在同一目录，在 `data/Small-Model-Track/Training/data/` 下依次执行 `zip -s 0 HAR.zip --out HAR_full.zip`、`unzip HAR_full.zip`。

| 环节 | `visual_temporal/` 下的实现 |
| --- | --- |
| 索引、裁剪、归一化及缓存 | `data/` |
| 视觉分类头适配 | `training/public_finetune.py` |
| 冻结骨骼帧分类器 | `models/public_dstformer.py`、`models/dstformer.py` |
| 残差 TCN 与受试者折分训练 | `training/temporal.py`、`training/temporal_suite.py` |
| 置信度融合与权重打包 | `release/blend.py`、`release/prune_fusion.py`、`release/bundle.py` |

查看入口参数：

```bash
python -m visual_temporal.training.public_finetune --help
python -m visual_temporal.training.temporal_suite --help
```

报告第 3–5 节说明原提交的参数设置和训练记录。训练工具还支持其他选项，其默认参数不等同于原提交配方。运行各阶段前需准备相应的数据与初始化文件，新训练模型按各自的实验设置评估。

## 许可

项目原创代码采用 [Apache 2.0](LICENSE)，报告原创文字与图示采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。第三方组件遵循各自适用条款，版权与许可声明详见 [NOTICE](NOTICE)。
