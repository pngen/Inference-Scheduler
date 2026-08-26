#include "inference_scheduler/local_driver.hpp"

#include <chrono>
#include <cmath>

namespace inference_scheduler {

LocalDriver::LocalDriver(Scheduler& scheduler, std::shared_ptr<Executor> executor,
                         std::vector<WorkerRegistration> workers)
    : sched_(scheduler), ex_(std::move(executor)), workers_(std::move(workers)) {
  avail_units_.resize(workers_.size(), 1);
  for (std::size_t i = 0; i < workers_.size(); ++i) {
    avail_units_[i] = workers_[i].capability.available_capacity_units;
  }
}

const WorkerRegistration* LocalDriver::find_worker(WorkerId id) const {
  for (const auto& w : workers_) { if (w.worker == id) return &w; }
  return nullptr;
}

void LocalDriver::set_available_units(WorkerId worker, int capacity_units) {
  for (std::size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i].worker == worker) { avail_units_[i] = capacity_units; return; }
  }
}

bool LocalDriver::dispatch_one(std::size_t& bcount) {
  auto d = sched_.plan_next_dispatch(false);
  if (!d.ok()) return false;
  const DispatchOutcome& out = d.value();
  if (out.decision != DispatchDecision::Dispatch) return false;
  const WorkerRegistration* wr = find_worker(out.worker_id);
  if (!wr) return false;
  if (out.batch_members.empty()) return false;
  std::vector<WorkRequest> reqs;
  for (std::size_t i = 0; i < out.batch_members.size(); ++i) {
    RequestId rid = out.batch_members[i];
    auto snap = sched_.request_snapshot(rid);
    auto specr = sched_.request_spec(rid);
    if (!snap.ok() || !specr.ok()) continue;
    WorkRequest w;
    w.request_id = rid;
    w.attempt_id = snap.value().attempt_id;
    w.generation = snap.value().generation;
    w.phase = snap.value().phase;
    w.model = snap.value().model;
    w.revision = snap.value().revision;
    w.adapter = specr.value().adapter;
    w.tokens = specr.value().tokens;
    w.cost_units = static_cast<std::int64_t>(specr.value().demand.cost_units > 0 ? std::ceil(specr.value().demand.cost_units) : 1);
    w.batch_index = static_cast<std::int32_t>(i);
    w.payload = specr.value().payload;
    reqs.push_back(w);
  }
  if (reqs.empty()) return false;
  auto results = ex_->run(reqs);
  // Confirm start for every member (Reserved -> Running).
  for (const auto& w : reqs) {
    CompletionReport start;
    start.epoch = sched_.epoch();
    start.worker = out.worker_id;
    start.boot_id = wr->boot_id;
    start.request_id = w.request_id;
    start.attempt_id = w.attempt_id;
    start.generation = w.generation;
    start.worker_snapshot.available_capacity_units = avail_units_[0];
    start.worker_snapshot.total_capacity_units = avail_units_[0];
    sched_.confirm_dispatch_start(start);
  }
  for (const auto& res : results) {
    CompletionReport comp;
    comp.epoch = sched_.epoch();
    comp.worker = out.worker_id;
    comp.boot_id = wr->boot_id;
    comp.request_id = res.request_id;
    comp.attempt_id = res.attempt_id;
    comp.generation = res.generation;
    comp.status = res.status;
    comp.failure_class = res.failure_class;
    comp.output_tokens_produced = res.output_tokens_produced;
    comp.work_units = res.work_units;
    comp.duration_us = res.duration_us;
    comp.error_message = res.error_message;
    auto co = sched_.complete_attempt(comp);
    if (co.ok() && co.value().acceptance == CompletionAcceptance::Accepted) {
      ++requests_completed_;
    }
  }
  ++bcount;
  ++batches_run_;
  return true;
}

std::size_t LocalDriver::run_until_idle(std::size_t max_batches) {
  std::size_t b = 0;
  while (b < max_batches) {
    if (sched_.queued_count() == 0) break;
    if (!dispatch_one(b)) break;
  }
  return batches_run_;
}

}  // namespace inference_scheduler