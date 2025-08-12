# collective_trace

[![Pylint](https://github.com/yangrudan/collective_trace/actions/workflows/pylint.yml/badge.svg)](https://github.com/yangrudan/collective_trace/actions/workflows/pylint.yml)

`collective_trace` 是一个分布式训练中的集体通信操作追踪工具，支持 PyTorch 等框架，可记录 `allreduce`、`broadcast` 等操作的类型、耗时、参与进程等信息，帮助开发者定位分布式训练性能瓶颈。

## 1. 环境要求

- Python 3.8+
- PyTorch 1.10+
- 支持分布式训练环境（单节点多进程或多节点）

## 2. 安装与使用

### 2.1 安装

```bash
git clone https://github.com/yangrudan/collective_trace.git
cd collective_trace
pip install -e .  #  editable 模式
```

### 2.2 使用 (集成到训练代码)​

在训练代码中导入并启用追踪功能（需在分布式框架导入前调用）：

```bash
import torch
import torch.distributed as dist

from collective_trace.collective_trace import trace_all_collectives

# 启用追踪，日志输出到 collective_trace.log
trace_all_collectives(trace_file='collective_trace.log')

import megatron  # Megatron此时导入的是已替换的函数
# Your training code here

```

### 2.3 编译发布

```bash
# Prepare
pip install setuptools wheel twine
# Build
python setup.py sdist bdist_wheel

```

## 3. 开发指南

```bash
git clone https://github.com/yangrudan/collective_trace.git
cd collective_trace
pip install -e .

# 运行测试用例（4 进程，PyTorch 环境, 默认异步）
torchrun --nproc_per_node=4 -m collective_trace.tests.test_in_torch

# 运行 CPU 环境测试（带同步模式）
torchrun --nproc_per_node=4 -m collective_trace.tests.test_in_cpu --sync_mode 
```

## 4. TODO

### 4.1 追踪结果实时显示

version 0.1 结果展示（image3.png）：​
横轴：训练时间（秒）​
纵轴：进程编号（rank）​
不同颜色块：不同类型的集体通信操作（如 allreduce 为蓝色，broadcast 为绿色）​
块长度：操作耗时

### 4.2 超时监测

追踪单个操作的耗时，并与预设阈值比较。若超过阈值，则记录相关信息（如操作类型、进程编号等），并可选择触发警告或自动优化建议。

### 4.3 跨节点通信分析

追踪跨节点的集体通信操作，包括但不限于 allreduce_coalesced 等高级通信模式。

## 5. 贡献与反馈

欢迎提交 issue 或 pull request 参与讨论和开发！

## 6. 致谢

感谢 Megatron-LM 项目中对分布式训练的探索，为 `collective_trace` 提供灵感和参考实现。

## 7. 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。
