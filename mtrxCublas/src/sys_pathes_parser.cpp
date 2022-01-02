#include "sys_pathes_parser.hpp"

#include <filesystem>
#include <sstream>

namespace mtrx {
Pathes loadSysPathes(const std::string &ev) {
  char *cenvs = ::getenv(ev.c_str());
  if (cenvs != nullptr) {
    std::string envs = cenvs;
    return parseSysPathes(envs);
  }
  return {};
}

Pathes parseSysPathes(const std::string &_pathes) {
  Pathes pathes;
  auto split = [&pathes](const std::string &env, char c) {
    size_t pindex = 0;
    size_t index = 0;
    while ((index = env.find(c, pindex)) != std::string::npos) {
      pathes.push_back(env.substr(pindex, index - pindex));
      pindex = index + 1;
    }
    auto path = env.substr(pindex, env.length() - pindex);
    if (!path.empty()) {
      pathes.push_back(path);
    }
  };
  split(_pathes, ':');
  return pathes;
}

std::string toString(const mtrx::Pathes &pathes) {
  std::stringstream sstream;
  for (size_t idx = 0; idx < pathes.size(); ++idx) {
    sstream << pathes[idx];
    if (idx < pathes.size() - 1) {
      sstream << ":";
    }
  }
  return sstream.str();
}
} // namespace mtrx
