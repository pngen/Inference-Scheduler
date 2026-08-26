#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace inference_scheduler::net {

constexpr std::uint32_t k_max_frame = 16u * 1024u * 1024u;  // 16 MiB bound

class FramedConnection {
 public:
  ~FramedConnection();
  FramedConnection(const FramedConnection&) = delete;
  FramedConnection& operator=(const FramedConnection&) = delete;

  static bool init();
  static void shutdown();
  static bool listen(const std::string& host, int port);
  static std::shared_ptr<FramedConnection> accept();
  static std::shared_ptr<FramedConnection> connect(const std::string& host, int port);

  bool send(std::string_view payload);
  bool recv(std::string& payload);
  void close();
  bool valid() const;

  explicit FramedConnection(int fd);

 private:
  int fd_ = -1;
};

}  // namespace inference_scheduler::net