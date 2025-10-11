# collective_trace

[![PyPI - Version](https://img.shields.io/pypi/v/collective-trace.svg)](https://pypi.org/project/collective-trace/)
[![Pylint](https://github.com/yangrudan/collective_trace/actions/workflows/pylint.yml/badge.svg)](https://github.com/yangrudan/collective_trace/actions/workflows/pylint.yml)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`collective_trace` 是一个轻量级分布式训练集体通信操作追踪工具，专注于帮助开发者解析和优化分布式训练中的通信瓶颈。通过对 PyTorch 等框架的集体通信操作（如 `allreduce`、`broadcast`、`all_gather` 等）进行无侵入式的 monkey-patching 追踪，可记录操作类型、耗时、参与进程、数据量等关键信息，为分布式训练性能分析提供数据支持。

## 1. 核心功能

- **全面追踪**：支持 PyTorch 主流集体通信操作（`all_reduce`、`all_gather`、`reduce_scatter` 等）及同步/异步模式
- **详细日志**：记录操作类型、耗时（毫秒级）、数据量（MB）、张量形状、参与进程组等信息
- **多环境支持**：兼容 GPU（NCCL 后端）和 CPU（Gloo 后端）环境
- **灵活集成**：一行代码即可接入现有训练流程，无需修改核心训练逻辑
- **数据分析**：提供日志解析工具，可统计不同操作的调用次数、总耗时和平均耗时
- **cuda流支持**：支持 CUDA 流的追踪计时

## 2. 环境要求

- Python 3.8+
- PyTorch 1.10+
- 分布式训练环境（单节点多进程或多节点）
- 可选依赖：
  - NCCL（GPU 分布式训练时需要）
  - Gloo（CPU 分布式训练时需要）

## 3. 安装与使用

### 3.1 安装方式

#### 源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/yangrudan/collective_trace.git

git submodule init + git submodule update 

cd collective_trace

#  editable 模式（方便开发调试）
pip install -e .

# 或常规安装
pip install .

cd lib_async_timer
python setup.py install
```

#### 编译发布（[已发布PyPI](https://pypi.org/project/collective-trace/)）

```bash
# 安装编译依赖
pip install setuptools wheel twine

# 构建分发包
python setup.py sdist bdist_wheel

# 上传至 PyPI（需配置凭据）
twine upload dist/*
```

### 3.2 快速使用

在训练代码中导入并启用追踪功能（**需在分布式框架导入前调用**）：

```python
import torch
import torch.distributed as dist

# ===========导入并启用追踪，日志输出到 collective_trace.log========================
from collective_trace.collective_trace import trace_all_collectives
tracer = trace_all_collectives(trace_file='collective_trace.log', verbose=True)
# ------------------------------------------------------------------------------

# 初始化分布式环境（示例）
dist.init_process_group(backend="nccl")

import megatron  # Megatron此时导入的是已替换的函数
# Your training code here


# 可选：导出追踪数据到 CSV
tracer.export_to_csv(f"trace_results_rank{dist.get_rank()}.csv")

# 销毁进程组
dist.destroy_process_group()
```

### 3.3 日志解析

提供两种日志解析工具，用于统计通信操作的关键指标：

#### 解析单个日志文件

```python
# 使用 utils/parse_single_file.py
# 修改文件中 LOG_FLODER_FILE 路径指向你的日志文件
python utils/parse_single_file.py
```

输出示例：

```bash
=== all_reduce ===
输出Shape (3,)                | count=       1 | total=      0.23 ms | avg=      0.23 ms
=== broadcast ===
输出Shape (1024, 1024)         | count=       4 | total=     12.56 ms | avg=      3.14 ms
```

#### 解析文件夹下所有日志

```python
# 使用 utils/parse_folder.py
# 修改文件中 LOG_FLODER_FILE 路径指向日志文件夹
python utils/parse_folder.py
```

## 4. 开发指南

### 4.1 本地开发环境搭建

```bash
# 克隆仓库
git clone https://github.com/yangrudan/collective_trace.git
cd collective_trace

# 安装开发依赖
pip install -e ".[dev]"

# 运行代码检查
pylint $(git ls-files '*.py' | grep -v '^tests/')
```

### 4.2 测试用例执行

#### GPU 环境测试（默认异步模式）

```bash
torchrun --nproc_per_node=4 -m collective_trace.tests.test_in_torch
```

#### CPU 环境测试（同步模式）

```bash
torchrun --nproc_per_node=4 -m collective_trace.tests.test_in_cpu --sync_mode
```

#### 特定操作测试（如 reduce_scatter）

```bash
torchrun --nproc_per_node=4 -m collective_trace.tests.test_rs_tensor
```

#### _coalescing_manager测试

```bash
torchrun --nproc_per_node=4 -m collective_trace.tests.test_coalescing
```

## 5. 日志格式说明

追踪日志包含通信操作的详细信息，格式示例：

```bash
[TRACE] global rank 1 in GROUP_2 - broadcast - async:0, Size: 0.03 MB, Shape: (1, 4096), Dtype: torch.int64, Duration: 0.196 ms, GROUP size 4  = [0, 1, 2, 3], call count: 2
[TRACE] global rank 1 in GROUP_2 - reduce_scatter_tensor - async:0, Size: 8.00 MB, Shape: (1024, 1, 4096), Dtype: torch.float16, Duration: 0.360 ms, GROUP size 4  = [0, 1, 2, 3], call count: 1

```

字段说明：

- `global rank`：进程全局编号
- `GROUP_x`：进程组编号
- `all_reduce`：通信操作类型
- `async:0`：同步模式（1 表示异步）
- `Size`：数据量（MB）
- `Shape`：张量形状
- `Duration`：操作耗时（毫秒）
- `GROUP size`：进程组大小及包含的进程编号
- `call count`：该形状的操作被调用次数

## 6. 贡献与反馈

欢迎通过以下方式参与项目开发：

- 提交 Issue 报告 bug 或提出功能建议
- 提交 Pull Request 贡献代码（请遵循项目代码风格）
- 参与讨论区交流使用经验

代码提交前请确保通过 lint 检查：

```bash
pylint $(git ls-files '*.py' | grep -v '^tests/')
```

## 7. 致谢

感谢 Megatron-LM 项目中对分布式训练的探索，为 `collective_trace` 提供了灵感和参考实现。

## 8. 许可证

本项目采用 GPL-3.0 许可证，详情请参阅 LICENSE 文件。
