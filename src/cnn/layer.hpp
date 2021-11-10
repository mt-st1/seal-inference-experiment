#pragma once

#include <memory>
#include <seal/seal.h>

#include "layer_type.hpp"
#include "utils/types.hpp"
#include "utils/helper.hpp"

namespace cnn {

class Forwardable {  // Interface
public:
  virtual ~Forwardable() {}

  /**
   * @param x input in the form of [N, C, H, W]
   */
  virtual void forward(types::double4d& x) const = 0;
  /**
   * @param x input in the form of [N, C]
   */
  virtual void forward(types::double2d& x) const = 0;
  /**
   * @param x input in the form of [N, C, H, W]
   * @param[out] y flattened output in the form of [N, C]
   */
  virtual void forward(types::double4d& x, types::double2d& y) const = 0;
};

class Layer : public Forwardable {
public:
  Layer(const ELayerType& layer_type);
  virtual ~Layer();

  const ELayerType& layer_type() const { return layer_type_; };

  virtual void forward(types::double4d& x) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(types::double2d& x) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(types::double4d& x, types::double2d& y) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

private:
  ELayerType layer_type_;
};

} // cnn


namespace cnn::encrypted {

class Forwardable {  // Interface
public:
  virtual ~Forwardable() {}

  /**
   * @param x_cts input ciphertexts (size: number of input channels)
   * @param[out] y_cts output ciphertexts (size: number of output channels)
   */
  virtual void forward(std::vector<seal::Ciphertext>& x_cts, std::vector<seal::Ciphertext>& y_cts) = 0;
  /**
   * @param x_ct input ciphertext
   * @param[out] y_ct output ciphertext
   */
  virtual void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) = 0;
  /**
   * @param x_cts input ciphertexts (size: number of input channels)
   * @param[out] y_ct flattened output ciphertext
   */
  virtual void forward(std::vector<seal::Ciphertext>& x_cts, seal::Ciphertext& y_ct) const = 0;
};

class Layer : public Forwardable {
public:
  Layer(const ELayerType& layer_type, const std::shared_ptr<helper::SealTool> seal_tool);
  virtual ~Layer();

  const ELayerType& layer_type() const { return layer_type_; };

  virtual void forward(std::vector<seal::Ciphertext>& x_cts, std::vector<seal::Ciphertext>& y_cts)  override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(seal::Ciphertext& x_ct, seal::Ciphertext& y_ct) override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

  virtual void forward(std::vector<seal::Ciphertext>& x_cts, seal::Ciphertext& y_ct) const override {
    std::cerr << __PRETTY_FUNCTION__ << " is not implemented." << std::endl;
  }

protected:
  ELayerType layer_type_;
  std::shared_ptr<helper::SealTool> seal_tool_;
};

} // cnn::encrypted
