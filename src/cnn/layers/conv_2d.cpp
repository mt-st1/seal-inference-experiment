#include "conv_2d.hpp"
#include "util.hpp"

#include <Eigen/Dense>

using std::size_t;

namespace cnn {

Conv2d::Conv2d(
    const types::float4d& filters,
    const std::vector<float> biases,
    const std::pair<std::size_t, std::size_t> stride,
    const std::pair<std::string, std::pair<std::size_t, std::size_t>> padding)
    : Layer(ELayerType::CONV_2D),
      filters_(filters),
      biases_(biases),
      stride_(stride),
      padding_(padding) {
  if (padding.first.empty()) {
    pad_top_ = padding.second.first;
    pad_btm_ = padding.second.first;
    pad_left_ = padding.second.second;
    pad_right_ = padding.second.second;
  } else if (padding.first == "valid") {
    pad_top_ = 0;
    pad_btm_ = 0;
    pad_left_ = 0;
    pad_right_ = 0;
  } else if (padding.first == "same") {
    assert(stride_.first == 1 && stride_.second == 1);
    const size_t fh = filters_.at(0).at(0).size(),
                 fw = filters_.at(0).at(0).at(0).size();
    pad_top_ = (fh - 1) / 2;
    pad_btm_ = fh - pad_top_ - 1;
    pad_left_ = (fw - 1) / 2;
    pad_right_ = fw - pad_left_ - 1;
  }
}
Conv2d::~Conv2d() {}

void Conv2d::forward(types::float4d& x) const {
  const size_t batch_size = x.size(), ih = x.at(0).at(0).size(),
               iw = x.at(0).at(0).at(0).size();
  const size_t fn = filters_.size(), fh = filters_.at(0).at(0).size(),
               fw = filters_.at(0).at(0).at(0).size();
  const size_t oh = static_cast<int>(
                   ((ih + pad_top_ + pad_btm_ - fh) / stride_.first) + 1),
               ow = static_cast<int>(
                   ((iw + pad_left_ + pad_right_ - fw) / stride_.second) + 1);

  types::float4d padded_x =
      util::apply_zero_padding(x, pad_top_, pad_btm_, pad_left_, pad_right_);

  auto col =
      util::im2col(padded_x, fh, fw, oh, ow, stride_);  // [N*OH*OW, IC*FH*FW]
  types::float2d flattened_filters = util::flatten_4d_vector_to_2d(filters_);
  auto col_filters =
      util::convert_to_eigen_matrix(flattened_filters);  // [FN, IC*FH*FW]

  auto wx_matrix = col * col_filters.transpose();

  std::vector<float> biases_copy(biases_.size());
  std::copy(biases_.begin(), biases_.end(), biases_copy.begin());
  auto bias_vec = util::convert_to_eigen_vector(biases_copy);

  Eigen::MatrixXf y_matrix(wx_matrix.rows(),
                           wx_matrix.cols());  // [N*OH*OW, FN]
  for (size_t i = 0; i < y_matrix.rows(); ++i) {
    y_matrix.row(i) = wx_matrix.row(i) + bias_vec.transpose();
  }

  y_matrix.transposeInPlace();  // [FN, N*OH*OW]

  types::float2d y_2d_vec = util::convert_to_double_2d(y_matrix);
  x = util::reshape_2d_vector_to_4d(y_2d_vec, batch_size, fn, oh, ow);
}

}  // namespace cnn

namespace cnn::encrypted {

Conv2d::Conv2d(const types::Plaintext3d& filters_pts,
               const std::vector<seal::Plaintext>& biases_pts,
               const std::vector<int>& filter_rotation_map,
               const std::shared_ptr<helper::he::SealTool>& seal_tool)
    : Layer(ELayerType::CONV_2D, seal_tool),
      filters_pts_(filters_pts),
      biases_pts_(biases_pts),
      filter_rotation_map_(filter_rotation_map) {}
Conv2d::~Conv2d() {}

void Conv2d::forward(std::vector<seal::Ciphertext>& x_cts,
                     std::vector<seal::Ciphertext>& y_cts) {
  const size_t filter_count = filters_pts_.size();
  const size_t input_channel_size = x_cts.size();
  const size_t filter_hw_size = filters_pts_.at(0).at(0).size();
  std::vector<seal::Ciphertext> mid_cts(input_channel_size * filter_hw_size);
  y_cts.resize(filter_count);

  for (size_t fi = 0; fi < filter_count; ++fi) {
    for (size_t ci = 0; ci < input_channel_size; ++ci) {
      for (size_t i = 0; i < filter_hw_size; ++i) {
        size_t mid_cts_idx = ci * filter_hw_size + i;
        seal_tool_->evaluator().rotate_vector(
            x_cts[ci], filter_rotation_map_[i], seal_tool_->galois_keys(),
            mid_cts[mid_cts_idx]);
        seal_tool_->evaluator().multiply_plain_inplace(mid_cts[mid_cts_idx],
                                                       filters_pts_[fi][ci][i]);
        seal_tool_->evaluator().rescale_to_next_inplace(mid_cts[mid_cts_idx]);
      }
    }
    seal_tool_->evaluator().add_many(mid_cts, y_cts[fi]);
    y_cts[fi].scale() = seal_tool_->scale();
    seal_tool_->evaluator().mod_switch_to_inplace(biases_pts_[fi],
                                                  y_cts[fi].parms_id());
    seal_tool_->evaluator().add_plain_inplace(y_cts[fi], biases_pts_[fi]);
  }
}

}  // namespace cnn::encrypted
