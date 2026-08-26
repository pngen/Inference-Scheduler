#include "inference_scheduler/cuda_executor.hpp"

#include <cstdio>

using namespace inference_scheduler;

int main() {
  CudaExecutor cu;
  if (!cu.init_ok()) {
    std::printf("CUDA UNAVAILABLE: %s\n", cu.init_error().c_str());
    std::printf("CUDA TEST SKIP\n");
    return 0;  // skip rather than fail when the machine has no usable CUDA device
  }
  std::printf("device=%s compute=%d.%d mem_total=%lld mem_avail=%lld\n",
              cu.device_name().c_str(), cu.compute_major(), cu.compute_minor(),
              (long long)cu.memory_total(), (long long)cu.memory_available());
  int fail = 0;
  if (cu.device_name().empty()) { std::printf("  FAIL: empty device name\n"); ++fail; }
  if (!(cu.compute_major() >= 8)) { std::printf("  FAIL: compute capability < 8\n"); ++fail; }
  if (!(cu.memory_total() > 0)) { std::printf("  FAIL: no device memory\n"); ++fail; }

  WorkRequest pre; pre.phase = RequestPhase::Prefill; pre.tokens.input = 2048; pre.tokens.output = 0; pre.cost_units = 1;
  WorkRequest dec; dec.phase = RequestPhase::Decode; dec.tokens.input = 0; dec.tokens.output = 64; dec.cost_units = 1;
  pre.request_id = RequestId(1); pre.attempt_id = AttemptId(1); pre.generation = Generation(1);
  dec.request_id = RequestId(2); dec.attempt_id = AttemptId(2); dec.generation = Generation(2);
  auto res = cu.run({pre, dec});
  if (res.size() != 2) { std::printf("  FAIL: expected 2 results\n"); ++fail; }
  for (const auto& r : res) {
    if (r.status != CompletionStatus::Succeeded) { std::printf("  FAIL: status %d: %s\n", (int)r.status, r.error_message.c_str()); ++fail; }
  }
  std::printf("  OK: ran %zu CUDA work items (prefill + decode)\n", res.size());
  if (fail == 0) { std::printf("CUDA TEST PASS\n"); return 0; }
  std::printf("CUDA TEST FAIL (%d)\n", fail);
  return 1;
}
