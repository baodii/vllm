#pragma once

#include <torch/all.h>

void hello_vllm(at::Tensor& a);

void vllm_cutlass(at::Tensor& a);
