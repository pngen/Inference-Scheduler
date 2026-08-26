#include "test_util.hpp"
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  auto cfg = default_scheduler_config();
  cfg.batch.max_batch_size = 3;
  Scheduler s(cfg, clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 1000));

  // compatible: same model/revision/phase/latency
  RequestId a(7000), b(7001), c(7002);
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), a);
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), b);
  s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), c);
  auto d = s.plan_dispatch(a);
  REQUIRE(d.ok() && d.value().decision == DispatchDecision::Dispatch, "dispatch");
  REQUIRE(d.ok() && d.value().batch_members.size() == 3, "3 compatible batched");

  // incompatible model
  RequestId x(7003);
  auto xspec = make_spec(TenantId(2), ModelIdentity(999), RequestPhase::Prefill);
  s.submit(xspec, x);
  auto d2 = s.plan_dispatch(x);
  REQUIRE(d2.ok() && d2.value().batch_members.size() == 1, "incompatible model not batched");

  // deterministic membership
  auto clock2 = std::make_shared<SimulatedClock>();
  Scheduler s2(cfg, clock2);
  s2.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 1000));
  RequestId a2(8000), b2(8001);
  s2.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), a2);
  s2.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), b2);
  auto d3 = s2.plan_dispatch(a2);
  REQUIRE(d3.ok() && d3.value().batch_members.size() == 2, "deterministic batch size");
  REQUIRE(d3.ok() && d3.value().batch_members[0] == a2 && d3.value().batch_members[1] == b2, "deterministic membership");

  // max_batch_size=1 disables batching
  auto cfg2 = default_scheduler_config(); cfg2.batch.max_batch_size = 1;
  Scheduler s3(cfg2, clock2);
  s3.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 1000));
  s3.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(9000));
  s3.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), RequestId(9001));
  auto d4 = s3.plan_dispatch(RequestId(9000));
  REQUIRE(d4.ok() && d4.value().batch_members.size() == 1, "max_batch_size=1 => no batch");
  return test_result("batch_test");
}
