#pragma once

// Inference Scheduler public API umbrella header.
//
// Include this single header to access the full public surface of the
// scheduler runtime. The library is vendor-neutral and implements scheduling
// authority only; it never performs model inference.

#include "version.hpp"
#include "clock.hpp"
#include "id_types.hpp"
#include "enums.hpp"
#include "error.hpp"
#include "result.hpp"
#include "types.hpp"
#include "scheduler.hpp"
