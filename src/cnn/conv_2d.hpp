#pragma once

#include "layer.hpp"

namespace cnn {

class Conv2D : public Layer {
public:
  Conv2D(
    const types::double4D& filters,
    const std::vector<double> biases,
    const std::pair<std::size_t, std::size_t> stride = {1, 1},
    const std::pair<std::string, std::pair<std::size_t, std::size_t>> padding = {"", {0, 0}}
  );
  ~Conv2D();

  void forward(types::double4D& x) const override;

private:
  types::double4D filters_;  // form of [FN, C, FH, FW]
  std::vector<double> biases_;  // form of [FN]
  std::pair<std::size_t, std::size_t> stride_;
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding_;
  std::size_t pad_top_;
  std::size_t pad_btm_;
  std::size_t pad_left_;
  std::size_t pad_right_;
};

} // cnn


namespace cnn::encrypted {

class Conv2D : public Layer {
public:
  Conv2D(
    const types::Plaintext3D& filters_pts,
    const std::vector<seal::Plaintext>& biases_pts,
    const std::vector<int>& filter_rotation_map,
    const std::shared_ptr<helper::SealTool>& seal_tool
  );
  ~Conv2D();

  void forward(std::vector<seal::Ciphertext>& x_cts, std::vector<seal::Ciphertext>& y_cts) override;

private:
  types::Plaintext3D filters_pts_;  // form of [FN, C, FH * FW]
  std::vector<seal::Plaintext> biases_pts_;  // form of [FN]
  std::vector<int> filter_rotation_map_;  // size: [FH * FW]
};

} // cnn::encrypted
