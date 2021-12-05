#pragma once

#include "layer.hpp"

const std::string BATCH_NORM_CLASS_NAME = "BatchNorm";

namespace cnn {

class BatchNorm : public Layer {
public:
  BatchNorm();
  ~BatchNorm();

  void forward(types::float4d& x) const override;

  void forward(types::float2d& x) const override;

private:
};

}  // namespace cnn

namespace cnn::encrypted {

class BatchNorm : public Layer {
public:
  BatchNorm(const std::vector<seal::Plaintext>& weights_pts,
            const std::vector<seal::Plaintext>& biases_pts,
            const std::shared_ptr<helper::he::SealTool>& seal_tool);
  BatchNorm();
  ~BatchNorm();

  void forward(std::vector<seal::Ciphertext>& x_cts,
               std::vector<seal::Ciphertext>& y_cts) override;

  void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override;

private:
  std::vector<seal::Plaintext> weights_pts_;
  std::vector<seal::Plaintext> biases_pts_;
};

}  // namespace cnn::encrypted
