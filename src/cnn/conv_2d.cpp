#include "conv_2d.hpp"

namespace cnn {

Conv2D::Conv2D(
  const types::double4D& filters,
  const std::vector<double> biases,
  const std::size_t stride,
  const std::size_t padding
) : Layer(ELayerType::CONV_2D),
    filters_(filters),
    biases_(biases),
    stride_(stride),
    padding_(padding)
  {}
Conv2D::~Conv2D() {}

void Conv2D::forward(types::double4D& x) const {
  // NOTE: This is temporary implementation
  std::cout << __PRETTY_FUNCTION__ << " is called." << std::endl;
}

} // cnn


namespace cnn::encrypted {

Conv2D::Conv2D(
  const types::Plaintext3D& filters_pts,
  const std::vector<seal::Plaintext>& biases_pts,
  const std::vector<int>& filter_rotation_map,
  const std::shared_ptr<helper::SealTool>& seal_tool
) : Layer(ELayerType::CONV_2D, seal_tool),
    filters_pts_(filters_pts),
    biases_pts_(biases_pts),
    filter_rotation_map_(filter_rotation_map)
  {}
Conv2D::~Conv2D() {}

void Conv2D::forward(std::vector<seal::Ciphertext>& x_cts, std::vector<seal::Ciphertext>& y_cts) {
  const size_t filter_count = filters_pts_.size();
  const size_t input_channel_size = x_cts.size();
  const size_t filter_hw_size = filters_pts_.at(0).at(0).size();
  std::vector<seal::Ciphertext> mid_cts(input_channel_size * filter_hw_size);
  y_cts.resize(filter_count);

  for (size_t fi = 0; fi < filter_count; ++fi) {
    for (size_t ci = 0; ci < input_channel_size; ++ci) {
      for (size_t i = 0; i < filter_hw_size; ++i) {
        size_t mid_cts_idx = ci * filter_hw_size + i;
        seal_tool_->evaluator().rotate_vector(x_cts[ci], filter_rotation_map_[i], seal_tool_->galois_keys(), mid_cts[mid_cts_idx]);
        seal_tool_->evaluator().multiply_plain_inplace(mid_cts[mid_cts_idx], filters_pts_[fi][ci][i]);
        seal_tool_->evaluator().rescale_to_next_inplace(mid_cts[mid_cts_idx]);
      }
    }
    seal_tool_->evaluator().add_many(mid_cts, y_cts[fi]);
    y_cts[fi].scale() = seal_tool_->scale();
    seal_tool_->evaluator().mod_switch_to_inplace(biases_pts_[fi], y_cts[fi].parms_id());
    seal_tool_->evaluator().add_plain_inplace(y_cts[fi], biases_pts_[fi]);
  }
}

} // cnn::encrypted
