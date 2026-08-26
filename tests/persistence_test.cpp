#include "test_util.hpp"
#include <fstream>
int main() {
  using namespace ts;
  auto clock = std::make_shared<SimulatedClock>();
  auto cfg = default_scheduler_config();
  cfg.persist_recovery = true;
  cfg.state_path = "state.bin";
  Scheduler s(cfg, clock);
  s.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));
  for (int i = 0; i < 5; ++i) {
    RequestId id(1000 + i);
    auto ad = s.submit(make_spec(TenantId(1), ModelIdentity(10), RequestPhase::Prefill), id);
    REQUIRE(ad.ok() && ad.value().decision == AdmissionDecision::Admit, "admit");
  }
  auto p = s.persist("state.bin");
  REQUIRE(p.ok(), "persist ok");

  // recover into a fresh scheduler
  Scheduler s2(cfg, std::make_shared<SimulatedClock>());
  s2.register_worker(make_worker(WorkerId(1), WorkerBootId(1), 100));
  auto rc = s2.recover("state.bin");
  if (!rc.ok()) std::printf("  recover error: [%s] %s\n", error_code_name(rc.error().code), rc.error().message.c_str());
  REQUIRE(rc.ok(), "recover ok");
  REQUIRE(s2.stats().requests_admitted == 5, "recovered admitted count");
  for (int i = 0; i < 5; ++i) {
    auto snap = s2.request_snapshot(RequestId(1000 + i));
    REQUIRE(snap.ok() && snap.value().state == RequestState::Queued, "request restored queued");
  }

  // corruption: bad magic
  { std::ofstream f("bad_magic.bin", std::ios::binary); f.write("XXXXYYYYZZZZ", 12); }
  Scheduler s3(cfg, std::make_shared<SimulatedClock>());
  auto rc3 = s3.recover("bad_magic.bin");
  REQUIRE(!rc3.ok(), "bad magic rejected");
  REQUIRE(rc3.error().code == ErrorCode::Corruption, "corruption code");

  // truncation
  { std::ofstream f("trunc.bin", std::ios::binary); f.write("ISFS", 4); }
  Scheduler s4(cfg, std::make_shared<SimulatedClock>());
  auto rc4 = s4.recover("trunc.bin");
  REQUIRE(!rc4.ok(), "truncation rejected");
  REQUIRE(rc4.error().code == ErrorCode::Corruption, "truncation corruption code");

  // checksum mismatch: take a valid file and flip one payload byte
  { std::ifstream in("state.bin", std::ios::binary); std::string data((std::istreambuf_iterator<char>(in)), {}); 
    if (data.size() > 20) data[16] = static_cast<char>(data[16] ^ 0x55);
    std::ofstream f("bad_checksum.bin", std::ios::binary); f.write(data.data(), static_cast<std::streamsize>(data.size())); }
  Scheduler s5(cfg, std::make_shared<SimulatedClock>());
  auto rc5 = s5.recover("bad_checksum.bin");
  REQUIRE(!rc5.ok(), "checksum mismatch rejected");
  REQUIRE(rc5.error().code == ErrorCode::Corruption, "checksum corruption code");

  std::remove("state.bin"); std::remove("state.bin.tmp"); std::remove("bad_magic.bin"); std::remove("trunc.bin"); std::remove("bad_checksum.bin");
  return test_result("persistence_test");
}
