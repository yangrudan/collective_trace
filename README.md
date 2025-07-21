# collective_trace

Trace collective operations for distributed training.

## Develop

```bash
# Develop
git clone https://github.com/yangrudan/collective_trace.git
cd collective_trace
pip install -e .

cd ..
torchrun --nproc_per_node=4 -m collective_trace.tests.test_in_torch
```

## Usage

```python
import torch
import torch.distributed as dist
from collective_trace.collective_trace import trace_collective

trace_all_collectives(trace_file='collective_trace.log')

import megatron  # Megatron此时导入的是已替换的函数
# Your training code here


```

**Prototype**
![Example](docs/image1.png)

**version 0.0**
![Trace](docs/image2.png)

**version 0.1 Results**
![Results](docs/image3.png)
