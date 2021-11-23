#include "network.hpp"

namespace cnn {

Network::Network() {}
Network::~Network() {}

/**
 * @param x_4d input images in the form of [N, C, H, W]
 * @return prediction outputs in the form of [N, CLASS_NUM]
 */
types::float2d Network::predict(types::float4d& x_4d) {
  types::float2d x_2d;
  bool is_flattened = false;

  using cnn::ELayerType;
  for (auto layer_it = layers_.begin(); layer_it != layers_.end();) {
    switch ((*layer_it)->layer_type()) {
      case CONV_2D:
      case AVG_POOL_2D:
      case ACTIVATION:
      case BATCH_NORM:
      case LINEAR:
        if (!is_flattened) {
          (*layer_it)->forward(x_4d);
        } else {
          (*layer_it)->forward(x_2d);
        }
        break;
      case FLATTEN:
        (*layer_it)->forward(x_4d, x_2d);
        x_4d.clear();
        x_4d.shrink_to_fit();
        is_flattened = true;
        break;
      default:
        break;
    }
    layer_it = layers_.erase(layer_it);
  }

  return x_2d;
}

}  // namespace cnn

namespace cnn::encrypted {

Network::Network() {}
Network::~Network() {}

/**
 * @param x_cts input ciphertexts in the form of [C]
 * @return prediction output ciphertext
 */
seal::Ciphertext Network::predict(std::vector<seal::Ciphertext>& x_cts) {
  std::vector<seal::Ciphertext> y_cts;
  seal::Ciphertext x_ct, y_ct;
  bool is_flattened = false;

  using cnn::encrypted::ELayerType;
  for (auto layer_it = layers_.begin(); layer_it != layers_.end();) {
    switch ((*layer_it)->layer_type()) {
      case CONV_2D:
      case AVG_POOL_2D:
      case ACTIVATION:
      case BATCH_NORM:
      case LINEAR:
        if (!is_flattened) {
          (*layer_it)->forward(x_cts, y_cts);
          x_cts.clear();
          x_cts.reserve(y_cts.size());
          for (auto& y_ct : y_cts) {
            x_cts.push_back(std::move(y_ct));
          }
        } else {
          (*layer_it)->forward(x_ct, y_ct);
          x_ct = std::move(y_ct);
        }
        break;
      case FLATTEN:
        (*layer_it)->forward(x_cts, x_ct);
        for (auto& ct : x_cts) {
          ct.release();
        }
        is_flattened = true;
        break;
      default:
        break;
    }
    layer_it = layers_.erase(layer_it);
  }

  return x_ct;
}

}  // namespace cnn::encrypted
