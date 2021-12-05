#include "avg_pool_2d.hpp"

namespace cnn {

AvgPool2d::AvgPool2d() : Layer(ELayerType::AVG_POOL_2D) {}
AvgPool2d::~AvgPool2d() {}

void AvgPool2d::forward(types::float4d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

AvgPool2d::AvgPool2d(const seal::Plaintext& mul_factor,
                     const std::vector<int>& rotation_map,
                     const std::shared_ptr<helper::he::SealTool>& seal_tool)
    : Layer(ELayerType::AVG_POOL_2D, seal_tool),
      mul_factor_(mul_factor),
      rotation_map_(rotation_map) {}
AvgPool2d::AvgPool2d() {}
AvgPool2d::~AvgPool2d() {}

void AvgPool2d::forward(std::vector<seal::Ciphertext>& x_cts,
                        std::vector<seal::Ciphertext>& y_cts) {
  const size_t input_channel_size = x_cts.size();
}

}  // namespace cnn::encrypted
