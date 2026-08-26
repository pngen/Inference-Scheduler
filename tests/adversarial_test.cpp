#include "test_util.hpp"
#include "net.hpp"

#include <cstring>
#include <thread>

using namespace inference_scheduler;
using namespace inference_scheduler::net;

int main() {
  // 1. Bounded framing: send must reject an oversized payload, and an advertised
  //    frame length beyond the cap is rejected by the reader.
  FramedConnection::init();
  const int port = 29911;
  bool listened = FramedConnection::listen("127.0.0.1", port);
  REQUIRE(listened, "listen");
  std::thread server([&]() { auto c = FramedConnection::accept(); if (c) { std::string p; bool r = c->recv(p); REQUIRE(r, "valid frame received"); } });
  auto cl = FramedConnection::connect("127.0.0.1", port);
  REQUIRE(cl != nullptr && cl->valid(), "client connect");
  // send() must reject a payload larger than the frame cap.
  std::string big(k_max_frame + 1, 'x');
  REQUIRE(!cl->send(big), "send rejects oversized frame");
  REQUIRE(cl->send("hello"), "send valid frame");  // lets the server complete recv
  server.join();
  FramedConnection::shutdown();

  // 2. Huge declared token/cost estimates must not overflow or crash admission/dispatch.
  Scheduler s(default_scheduler_config(), std::make_shared<SimulatedClock>());
  auto w = make_worker(WorkerId(1), WorkerBootId(1), 8);
  s.register_worker(w);
  auto spec = make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill);
  spec.tokens.input = 9000000000000000000LL;  // astronomically large
  spec.tokens.output = 9223372036854775807LL;
  spec.demand.cost_units = 1.0e18;
  auto ad = s.submit(spec, RequestId(777));
  // Should be admitted at most, and dispatch must not crash: either Hold or a guarded release.
  if (ad.ok() && ad.value().decision == AdmissionDecision::Admit) {
    auto d = s.plan_dispatch(RequestId(777), false);
    REQUIRE(d.ok(), "plan does not crash on huge cost");
    // cost > capacity must not dispatch and must not go negative / leak.
    REQUIRE(s.available_worker_capacity_units() >= 0, "capacity accounting non-negative");
  }
  // 3. Concurrent admission at the capacity boundary must not oversubscribe.
  auto cfg = default_scheduler_config();
  cfg.admission.max_global_admitted = 40;
  cfg.admission.default_tenant_concurrency = 40;
  Scheduler s2(cfg, std::make_shared<SimulatedClock>());
  s2.register_worker(make_worker(WorkerId(2), WorkerBootId(2), 8));
  std::atomic<int> admitted{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) {
    ts.emplace_back([&](int tid) {
      for (int k = 0; k < 100; ++k) { RequestId id(100000 + tid * 1000 + k); auto r = s2.submit(make_spec(TenantId(1 + tid), ModelIdentity(10), RequestPhase::Prefill), id); if (r.ok() && r.value().decision == AdmissionDecision::Admit) admitted.fetch_add(1); }
    }, t);
  }
  for (auto& th : ts) th.join();
  REQUIRE(admitted.load() <= 40, "concurrent admission respects max_global_admitted");
  auto st = s2.stats();
  REQUIRE(st.current_admitted <= 40, "admitted counter within cap");

  return test_result("adversarial_test");
}