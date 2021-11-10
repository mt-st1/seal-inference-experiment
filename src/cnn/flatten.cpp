#include "flatten.hpp"

namespace cnn {

Flatten::Flatten() : Layer(ELayerType::FLATTEN) {}
Flatten::~Flatten() {}

void Flatten::forward(types::double4d& x, types::double2d& y) const {
  y.reserve(x.size());
  size_t units_size = x.at(0).size() * x.at(0).at(0).size() * x.at(0).at(0).at(0).size();
  std::vector<double> units;

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

} // cnn


namespace cnn::encrypted {

Flatten::Flatten(const std::shared_ptr<helper::SealTool>& seal_tool) : Layer(ELayerType::FLATTEN, seal_tool) {}
Flatten::~Flatten() {}

void Flatten::forward(std::vector<seal::Ciphertext>& x_cts, seal::Ciphertext& y_ct) const {
  // NOTE: This is temporary implementation
  y_ct = std::move(x_cts[0]);
}

}
