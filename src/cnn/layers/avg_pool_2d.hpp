#pragma once

#include "layer.hpp"

const std::string AVG_POOL_2D_CLASS_NAME = "AvgPool2d";

namespace cnn {

class AvgPool2d : public Layer {
public:
  AvgPool2d();
  ~AvgPool2d();

  void forward(types::float4d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class AvgPool2d : public Layer {
public:
  AvgPool2d();
  ~AvgPool2d();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

private:
};

}  // namespace cnn::encrypted
