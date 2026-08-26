#pragma once

#include <inference_scheduler/inference_scheduler.hpp>
#include <inference_scheduler/executor.hpp>
#include <inference_scheduler/local_driver.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ts = inference_scheduler;

inline int g_failures = 0;

#define REQUIRE(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); ++g_failures; } } while (0)
#define REQUIRE_OK(r) do { auto&& _r = (r); if (!_r.ok()) { std::printf("  FAIL: expected ok, got [%s] %s\n", ts::error_code_name(_r.error().code), _r.error().message.c_str()); ++g_failures; } } while (0)

inline ts::WorkerRegistration make_worker(ts::WorkerId id, ts::WorkerBootId boot, int units, const std::string& backend = "cpu") {
  ts::WorkerRegistration w;
  w.worker = id;
  w.node = ts::NodeId(id.value() * 100 + 1);
  w.accelerator = ts::AcceleratorId(id.value() * 100 + 2);
  w.boot_id = boot;
  w.state = ts::WorkerState::Online;
  w.capability.backend = backend;
  w.capability.device_name = backend + "-" + std::to_string(id.value());
  w.capability.total_capacity_units = units;
  w.capability.available_capacity_units = units;
  return w;
}

inline ts::RequestSpec make_spec(ts::TenantId tenant, ts::ModelIdentity model, ts::RequestPhase phase,
                                 double cost = 1.0, long long in = 16, long long out = 16) {
  ts::RequestSpec s;
  s.tenant = tenant;
  s.session = ts::SessionId(tenant.value() * 1000 + 1);
  s.sequence = ts::SequenceId(tenant.value() * 1000 + 2);
  s.model = model;
  s.revision = ts::ModelRevision(1);
  s.phase = phase;
  s.priority = ts::PriorityClass::Medium;
  s.latency_class = ts::LatencyClass::Standard;
  s.tokens.input = in;
  s.tokens.output = out;
  s.demand.cost_units = cost;
  s.estimated_memory_bytes = 0;
  return s;
}

inline int test_result(const char* name) {
  if (g_failures == 0) { std::printf("%s PASS\n", name); return 0; }
  std::printf("%s FAIL (%d)\n", name, g_failures);
  return 1;
}
