#include "inference_scheduler/cuda_executor.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace inference_scheduler {

namespace {
constexpr std::size_t k_max_elems = 16u * 1024u * 1024u;  // bounded per-request

__global__ void prefill_kernel(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * 1.0001f + 0.5f;
}

__global__ void decode_kernel(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] + 1.0f;
}
}  // namespace

struct CudaExecutor::Impl {};

CudaExecutor::CudaExecutor() : p_(std::make_unique<Impl>()) {
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess || ndev < 1) {
    init_ok_ = false;
    init_error_ = std::string(cudaGetErrorString(e));
    return;
  }
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) { init_ok_ = false; init_error_ = std::string(cudaGetErrorString(e)); return; }
  device_name_ = prop.name;
  compute_major_ = prop.major;
  compute_minor_ = prop.minor;
  memory_total_ = static_cast<std::int64_t>(prop.totalGlobalMem);
  std::size_t freeb = 0, totb = 0;
  e = cudaMemGetInfo(&freeb, &totb);
  if (e == cudaSuccess) memory_available_ = static_cast<std::int64_t>(freeb); else memory_available_ = memory_total_;
  init_ok_ = true;
}

CudaExecutor::~CudaExecutor() = default;

std::string CudaExecutor::name() const { return "cuda"; }
std::string CudaExecutor::device_name() const { return device_name_; }

std::vector<WorkResult> CudaExecutor::run(const std::vector<WorkRequest>& batch) {
  std::vector<WorkResult> out;
  out.reserve(batch.size());
  for (const auto& w : batch) {
    WorkResult r;
    r.request_id = w.request_id;
    r.attempt_id = w.attempt_id;
    r.generation = w.generation;
    r.output_tokens_produced = w.tokens.output;
    r.work_units = w.cost_units;
    if (!init_ok_) { r.status = CompletionStatus::Failed; r.failure_class = FailureClass::RetryableWorker; r.error_message = "cuda unavailable: " + init_error_; out.push_back(r); continue; }
    // Bounded element count scaled by cost/tokens.
    std::size_t n = 1024u + static_cast<std::size_t>(w.tokens.input + w.tokens.output) * 64u;
    if (n > k_max_elems) n = k_max_elems;
    if (n == 0) n = 1024u;
    const int ni = static_cast<int>(n);
    float* h_in = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_out = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* d = nullptr;
    if (!h_in || !h_out || cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) { r.status = CompletionStatus::Failed; r.failure_class = FailureClass::RetryableWorker; r.error_message = "cuda alloc failed"; out.push_back(r); std::free(h_in); std::free(h_out); if (d) cudaFree(d); continue; }
    for (std::size_t i = 0; i < n; ++i) { h_in[i] = static_cast<float>(i & 0xFF); h_out[i] = 0.0f; }
    cudaMemcpy(d, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t e1 = cudaSuccess;
    const int threads = 256;
    const int blocks = (ni + threads - 1) / threads;
    if (w.phase == RequestPhase::Decode) {
      // decode-like: repeated small kernels
      for (int it = 0; it < 8; ++it) decode_kernel<<<blocks, threads>>>(d, ni);
    } else {
      prefill_kernel<<<blocks, threads>>>(d, ni);
    }
    e1 = cudaDeviceSynchronize();
    if (e1 == cudaSuccess) e1 = cudaMemcpy(h_out, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e1 != cudaSuccess) { r.status = CompletionStatus::Failed; r.failure_class = FailureClass::RetryableWorker; r.error_message = "cuda kernel failed: " + std::string(cudaGetErrorString(e1)); }
    else {
      double acc = 0.0;
      for (std::size_t i = 0; i < n; ++i) acc += h_out[i];
      r.status = CompletionStatus::Succeeded;
      r.duration_us = static_cast<std::int64_t>(acc);
    }
    cudaFree(d); std::free(h_in); std::free(h_out);
    out.push_back(r);
  }
  return out;
}

}  // namespace inference_scheduler