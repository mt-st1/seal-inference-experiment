#include "activation.hpp"

namespace cnn {

Activation::Activation() : Layer(ELayerType::ACTIVATION) {}
Activation::~Activation() {}

void Activation::forward(types::float4d& x) const {}

void Activation::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

Activation::Activation(const std::string& activation,
                       const std::shared_ptr<helper::he::SealTool>& seal_tool)
    : Layer(ELayerType::ACTIVATION, seal_tool), activation_(activation) {}
Activation::Activation() {}
Activation::~Activation() {}

void Activation::forward(std::vector<seal::Ciphertext>& x_cts,
                         std::vector<seal::Ciphertext>& y_cts) {
  const size_t input_channel_size = x_cts.size();
}

void Activation::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Activation::Activation() {}
Activation::~Activation() {}

void Activation::forward(types::Ciphertext3d& x_ct_3d) {}

void Activation::forward(std::vector<seal::Ciphertext>& x_cts) {}

}  // namespace cnn::encrypted::batch
