#include "batch_norm.hpp"

namespace cnn {

BatchNorm::BatchNorm() : Layer(ELayerType::BATCH_NORM) {}
BatchNorm::~BatchNorm() {}

void BatchNorm::forward(types::float4d& x) const {}

void BatchNorm::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

BatchNorm::BatchNorm(const std::vector<seal::Plaintext>& weights_pts,
                     const std::vector<seal::Plaintext>& biases_pts,
                     const std::shared_ptr<helper::he::SealTool>& seal_tool)
    : Layer(ELayerType::BATCH_NORM, seal_tool),
      weights_pts_(weights_pts),
      biases_pts_(biases_pts) {}
BatchNorm::BatchNorm() {}
BatchNorm::~BatchNorm() {}

void BatchNorm::forward(std::vector<seal::Ciphertext>& x_cts,
                        std::vector<seal::Ciphertext>& y_cts) {
  const size_t input_channel_size = x_cts.size();
}

void BatchNorm::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

}  // namespace cnn::encrypted
