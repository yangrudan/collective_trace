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
from collective_trace import CollectiveTracer

tracer = CollectiveTracer(trace_file="trace.csv", verbose=True)
tracer.apply_hooks()

# Your training code here

tracer.remove_hooks()
tracer.export_to_csv("trace.csv")
```

**Prototype**
![Example](docs/image1.png)

**version 0.0**
![Trace](docs/image2.png)

**version 0.1 Results**
![Results](docs/image3.png)
