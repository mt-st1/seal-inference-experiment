#pragma once

#include "layer.hpp"

const std::string LINEAR_CLASS_NAME = "Linear";

namespace cnn {

class Linear : public Layer {
public:
  Linear();
  ~Linear();

  void forward(types::double2d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class Linear : public Layer {
public:
  Linear();
  ~Linear();

  void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override;

private:
};

}  // namespace cnn::encrypted
