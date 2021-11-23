#pragma once

#include "layer.hpp"

const std::string ACTIVATION_CLASS_NAME = "Activation";

namespace cnn {

class Activation : public Layer {
public:
  Activation();
  ~Activation();

  void forward(types::float4d& x) const override;

  void forward(types::float2d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class Activation : public Layer {
public:
  Activation();
  ~Activation();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

  void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override;

private:
};

}  // namespace cnn::encrypted
