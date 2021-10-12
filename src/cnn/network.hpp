#pragma once

#include <seal/seal.h>

#include "layer.hpp"

namespace cnn {

class Network {
public:
  Network();
  ~Network();

  types::double2D predict(types::double4D& x_4d);

  void add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
  }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

} // cnn


namespace cnn::encrypted {

class Network {
public:
  Network();
  ~Network();

  seal::Ciphertext predict(std::vector<seal::Ciphertext>& x_cts);

  void add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
  }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

} // cnn::encrypted
