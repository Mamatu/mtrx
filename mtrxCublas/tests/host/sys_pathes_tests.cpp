#include <array>
#include <gtest/gtest.h>

#include "../src/sys_pathes_parser.hpp"
#include <spdlog/spdlog.h>

namespace mtrx {
class SysPathesTests : public testing::Test {
public:
  void SetUp() override { spdlog::set_level(spdlog::level::debug); }
};

TEST_F(SysPathesTests, parse_empty) {
  EXPECT_EQ(Pathes({}), mtrx::parseSysPathes(""));
}

TEST_F(SysPathesTests, parse_singe_path_without_delimiter) {
  EXPECT_EQ(Pathes({"/tmp/example"}), mtrx::parseSysPathes("/tmp/example"));
}

TEST_F(SysPathesTests, parse_singe_path_with_delimiter) {
  EXPECT_EQ(Pathes({"/tmp/example"}), mtrx::parseSysPathes("/tmp/example:"));
}

TEST_F(SysPathesTests, parse_two_pathes_with_delimiter) {
  EXPECT_EQ(Pathes({"/tmp/example", "/tmp/example_1"}),
            mtrx::parseSysPathes("/tmp/example:/tmp/example_1:"));
}

TEST_F(SysPathesTests, parse_two_pathes_without_delimiter) {
  EXPECT_EQ(Pathes({"/tmp/example", "/tmp/example_1"}),
            mtrx::parseSysPathes("/tmp/example:/tmp/example_1"));
}
} // namespace mtrx
