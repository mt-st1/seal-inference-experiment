#pragma once

#include "layer.hpp"

const std::string FLATTEN_CLASS_NAME = "Flatten";

namespace cnn {

class Flatten : public Layer {
public:
  Flatten();
  ~Flatten();

  void forward(types::double4d& x, types::double2d& y) const override;
};

}  // namespace cnn

namespace cnn::encrypted {

class Flatten : public Layer {
public:
  Flatten(const std::shared_ptr<helper::he::SealTool>& seal_tool);
  ~Flatten();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               seal::Ciphertext& y_ct) const override;
};

}  // namespace cnn::encrypted
