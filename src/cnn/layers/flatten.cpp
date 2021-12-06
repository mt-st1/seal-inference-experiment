#include "flatten.hpp"

namespace cnn {

Flatten::Flatten() : Layer(ELayerType::FLATTEN) {}
Flatten::~Flatten() {}

void Flatten::forward(types::float4d& x, types::float2d& y) const {
  y.reserve(x.size());
  size_t units_size =
      x.at(0).size() * x.at(0).at(0).size() * x.at(0).at(0).at(0).size();
  std::vector<float> units;

  for (const auto& channels_3d : x) {
    units.reserve(units_size);
    for (const auto& channel_2d : channels_3d) {
      for (const auto& row : channel_2d) {
        for (const auto& e : row) {
          units.push_back(e);
        }
      }
    }
    y.push_back(units);
    units.clear();
  }
}

}  // namespace cnn

namespace cnn::encrypted {

Flatten::Flatten(const std::shared_ptr<helper::he::SealTool>& seal_tool)
    : Layer(ELayerType::FLATTEN, seal_tool) {}
Flatten::Flatten() {}
Flatten::~Flatten() {}

void Flatten::forward(std::vector<seal::Ciphertext>& x_cts,
                      seal::Ciphertext& y_ct) const {
  // NOTE: This is temporary implementation
  y_ct = std::move(x_cts[0]);
}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Flatten::Flatten() {}
Flatten::~Flatten() {}

void Flatten::forward(types::Ciphertext3d& x_ct_3d,
                      std::vector<seal::Ciphertext>& x_cts) const {}

}  // namespace cnn::encrypted::batch
