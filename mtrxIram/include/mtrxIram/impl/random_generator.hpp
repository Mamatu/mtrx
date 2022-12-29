#ifndef MTRX_RANDOM_GENERATOR_HPP
#define MTRX_RANDOM_GENERATOR_HPP

#include <functional>
#include <random>

namespace mtrx {

template <typename T> class RandomGenerator final {
public:
  RandomGenerator(T min, T max);
  RandomGenerator(T min, T max, unsigned int seed);

  std::pair<T, T> setRange(T min, T max);
  void setSeed(T seed);

  T operator()();
  T operator()(T min, T max);

private:
  std::random_device m_rd;
  std::default_random_engine m_dre;
  std::uniform_real_distribution<T> m_dis;
  std::pair<T, T> m_range;
};

template <typename T>
RandomGenerator<T>::RandomGenerator(T min, T max) : m_dre(m_rd()) {
  setRange(min, max);
}

template <typename T>
RandomGenerator<T>::RandomGenerator(T min, T max, unsigned int seed)
    : RandomGenerator(min, max) {
  setSeed(seed);
}

template <typename T>
std::pair<T, T> RandomGenerator<T>::setRange(T min, T max) {
  std::pair<T, T> previousRange = m_range;

  m_range = std::make_pair(min, max);
  m_dis = std::uniform_real_distribution<T>(min, max);

  return previousRange;
}

template <typename T> void RandomGenerator<T>::setSeed(T seed) {
  m_dre.seed(seed);
}

template <typename T> T RandomGenerator<T>::operator()() {
  T v = m_dis(m_dre);
  return v;
}

template <typename T> T RandomGenerator<T>::operator()(T min, T max) {
  auto prange = setRange(min, max);
  T v = this->operator()();

  setRange(prange.first, prange.second);
  return v;
}
} // namespace mtrx
#endif
