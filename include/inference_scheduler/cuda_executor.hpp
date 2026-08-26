#pragma once

#include "executor.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace inference_scheduler {

// CUDA-backed executor: performs real, bounded GPU work (prefill-like and
// decode-like kernels) on the installed NVIDIA accelerator. This is scheduler
// execution proof; it does not perform LLM inference or access model weights.
class CudaExecutor final : public Executor {
 public:
  CudaExecutor();
  ~CudaExecutor() override;
  std::string name() const override;
  std::vector<WorkResult> run(const std::vector<WorkRequest>& batch) override;

  bool init_ok() const { return init_ok_; }
  std::string init_error() const { return init_error_; }
  std::string device_name() const;
  int compute_major() const { return compute_major_; }
  int compute_minor() const { return compute_minor_; }
  std::int64_t memory_total() const { return memory_total_; }
  std::int64_t memory_available() const { return memory_available_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  bool init_ok_ = false;
  std::string init_error_;
  std::string device_name_;
  int compute_major_ = 0;
  int compute_minor_ = 0;
  std::int64_t memory_total_ = 0;
  std::int64_t memory_available_ = 0;
};

}  // namespace inference_scheduler
