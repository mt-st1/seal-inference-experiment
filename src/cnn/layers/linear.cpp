#include "linear.hpp"

namespace cnn {

Linear::Linear() : Layer(ELayerType::LINEAR) {}
Linear::~Linear() {}

void Linear::forward(types::float2d& x) const {}

}  // namespace cnn

namespace cnn::encrypted {

Linear::Linear() {}
Linear::~Linear() {}

void Linear::forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) {}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Linear::Linear() {}
Linear::~Linear() {}

void Linear::forward(std::vector<seal::Ciphertext>& x_cts) {}

}  // namespace cnn::encrypted::batch
