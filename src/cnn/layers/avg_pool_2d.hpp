#pragma once

#include "layer.hpp"

const std::string AVG_POOL_2D_CLASS_NAME = "AvgPool2d";

namespace cnn {

class AvgPool2d : public Layer {
public:
  // AvgPool2d(const std::pair<std::size_t, std::size_t>& kernel,
  //           const std::pair<std::size_t, std::size_t>& stride,
  //           const std::pair<std::size_t, std::size_t>& padding);
  AvgPool2d();
  ~AvgPool2d();

  void forward(types::float4d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class AvgPool2d : public Layer {
public:
  AvgPool2d(const seal::Plaintext& mul_factor,
            const std::vector<int>& rotation_map,
            const std::shared_ptr<helper::he::SealTool>& seal_tool);
  AvgPool2d();
  ~AvgPool2d();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

private:
  seal::Plaintext mul_factor_;
  std::vector<int> rotation_map_;
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class AvgPool2d : public Layer {
public:
  AvgPool2d();
  ~AvgPool2d();

  void forward(types::Ciphertext3d& x_ct_3d) override;

private:
};

}  // namespace cnn::encrypted::batch
