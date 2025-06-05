import torch
from vllm import _custom_ops as ops


a = torch.randn(2, 3, 4, 5, device="xpu")
b = torch.randn([3, 3], device="xpu")

ops.hello_vllm(a)
ops.vllm_cutlass(b)
