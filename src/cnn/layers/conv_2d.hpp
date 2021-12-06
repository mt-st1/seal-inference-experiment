#pragma once

#include "layer.hpp"

const std::string CONV_2D_CLASS_NAME = "Conv2d";

namespace cnn {

class Conv2d : public Layer {
public:
  Conv2d(const types::float4d& filters,
         const std::vector<float>& biases,
         const std::pair<std::size_t, std::size_t>& stride = {1, 1},
         const std::pair<std::string, std::pair<std::size_t, std::size_t>>&
             padding = {"", {0, 0}});
  ~Conv2d();

  void forward(types::float4d& x) const override;

private:
  types::float4d filters_;     // form of [FN, C, FH, FW]
  std::vector<float> biases_;  // form of [FN]
  std::pair<std::size_t, std::size_t> stride_;
  std::pair<std::string, std::pair<std::size_t, std::size_t>> padding_;
  std::size_t pad_top_;
  std::size_t pad_btm_;
  std::size_t pad_left_;
  std::size_t pad_right_;
};

}  // namespace cnn

namespace cnn::encrypted {

class Conv2d : public Layer {
public:
  Conv2d(const types::Plaintext3d& filters_pts,
         const std::vector<seal::Plaintext>& biases_pts,
         const std::vector<int>& rotation_map,
         const std::shared_ptr<helper::he::SealTool>& seal_tool);
  Conv2d();
  ~Conv2d();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

private:
  types::Plaintext3d filters_pts_;           // form of [FN, C, FH * FW]
  std::vector<seal::Plaintext> biases_pts_;  // form of [FN]
  std::vector<int> rotation_map_;            // size: [FH * FW]
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class Conv2d : public Layer {
public:
  Conv2d();
  ~Conv2d();

  void forward(types::Ciphertext3d& x_ct_3d) override;

private:
};

}  // namespace cnn::encrypted::batch
