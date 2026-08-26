#pragma once

#include "inference_scheduler/error.hpp"
#include "inference_scheduler/result.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace inference_scheduler::detail {

constexpr std::uint32_t k_persistence_magic = 0x49534653u;
constexpr std::uint32_t k_persistence_frame_version = 1u;
// magic+ver+len precede the payload; the checksum follows it as a trailer.
constexpr std::size_t k_persistence_header_size = 16u;  // magic+ver+len
constexpr std::size_t k_persistence_trailer_size = 8u;  // checksum

Result<void> write_persistence_file(const std::string& path, std::string_view payload);

Result<std::string> read_persistence_file(const std::string& path);

}  // namespace inference_scheduler::detail
