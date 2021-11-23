#include "util.hpp"

using std::size_t;

namespace cnn::util {

/**
 * @brief expand images into a matrix for efficient convolution
 * @param x input in the form of [N, IC, IH, IW]
 * @return eigen matrix in the form of [N*OH*OW, IC*FH*FW]
 */
Eigen::MatrixXd im2col(types::double4d& x,
                       const size_t& fh,
                       const size_t& fw,
                       const size_t& oh,
                       const size_t& ow,
                       const std::pair<size_t, size_t>& stride) {
  const size_t n = x.size(), ic = x.at(0).size(), ih = x.at(0).at(0).size(),
               iw = x.at(0).at(0).at(0).size();
  const size_t stride_h = stride.first, stride_w = stride.second,
               out_hw_size = oh * ow, filter_hw_size = fh * fw;

  types::double2d matrix(n * oh * ow, std::vector<double>(ic * fh * fw));
  size_t mat_col_i, mat_row_i, x_col_i, x_row_i;
  for (size_t i = 0; i < n; ++i) {
    for (size_t oh_i = 0; oh_i < oh; ++oh_i) {
      for (size_t ow_i = 0; ow_i < ow; ++ow_i) {
        mat_col_i = (i * out_hw_size) + (oh_i * oh) + ow_i;
        for (size_t ic_i = 0; ic_i < ic; ++ic_i) {
          for (size_t fh_i = 0; fh_i < fh; ++fh_i) {
            x_col_i = fh_i + oh_i * stride_h;
            for (size_t fw_i = 0; fw_i < fw; ++fw_i) {
              mat_row_i = (ic_i * filter_hw_size) + (fh_i * fh) + fw_i;
              x_row_i = fw_i + ow_i * stride_w;
              matrix[mat_col_i][mat_row_i] = x[i][ic_i][x_col_i][x_row_i];
            }
          }
        }
      }
    }
  }

  return convertToEigenMatrix(matrix);
}

/**
 * @brief apply padding to target input 4d vector (type: double)
 * @param x input in the form of [N, IC, IH, IW]
 * @return padded 4d vector
 */
types::double4d apply_zero_padding(types::double4d& x,
                                   const std::size_t& pad_top,
                                   const std::size_t& pad_btm,
                                   const std::size_t& pad_left,
                                   const std::size_t& pad_right) {
  const size_t n = x.size(), ic = x.at(0).size(), ih = x.at(0).at(0).size(),
               iw = x.at(0).at(0).at(0).size();
  const size_t padded_height_size = ih + pad_top + pad_btm,
               padded_width_size = iw + pad_left + pad_right;
  types::double4d padded_x(
      n, types::double3d(
             ic, types::double2d(padded_height_size,
                                 std::vector<double>(padded_width_size))));

  auto is_tb_pad_idx = [&](size_t h) {
    int btm_pad_idx = h - (pad_top + ih);
    return h < pad_top || (0 <= btm_pad_idx && btm_pad_idx < pad_btm);
  };
  auto is_lr_pad_idx = [&](size_t w) {
    int right_pad_idx = w - (pad_left + iw);
    return w < pad_left || (0 <= right_pad_idx && right_pad_idx < pad_right);
  };

  size_t org_x_col_i, org_x_row_i;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < ic; ++j) {
      for (size_t k = 0; k < padded_height_size; ++k) {
        if (is_tb_pad_idx(k)) {
          continue;
        } else {
          for (size_t l = 0; l < padded_width_size; ++l) {
            if (is_lr_pad_idx(l)) {
              continue;
            } else {
              org_x_col_i = k - pad_top, org_x_row_i = l - pad_left;
              padded_x[i][j][k][l] = x[i][j][org_x_col_i][org_x_row_i];
            }
          }
        }
      }
    }
  }

  return padded_x;
}

/**
 * @brief vector<vector<double>> -> Eigen::MatrixXd
 * @param vec_2d 2d vector (type: double)
 * @return eigen matrix (type: double)
 */
Eigen::MatrixXd convertToEigenMatrix(types::double2d& vec_2d) {
  const size_t row_size = vec_2d.size();
  const size_t col_size = vec_2d.at(0).size();

  Eigen::MatrixXd matrix(row_size, col_size);
  for (size_t i = 0; i < row_size; ++i) {
    matrix.row(i) =
        Eigen::Map<Eigen::VectorXd>(vec_2d.at(i).data(), vec_2d.at(i).size());
  }

  return matrix;
}

/**
 * @brief vector<double> -> Eigen::VectorXd
 * @param vec vector (type: double)
 * @return eigen vector (type: double)
 */
Eigen::VectorXd convertToEigenVector(std::vector<double>& vec) {
  return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}

/**
 * @brief Eigen::MatrixXd -> vector<vector<double>>
 * @param matrix eigen matrix (type: double)
 * @return 2d vector (type: double)
 */
types::double2d convertToDouble2D(Eigen::MatrixXd& matrix) {
  const size_t row_size = matrix.rows(), col_size = matrix.cols();

  types::double2d vec_2d(row_size, std::vector<double>(col_size));
  for (size_t i = 0; i < row_size; ++i) {
    Eigen::Map<Eigen::VectorXd>(vec_2d.at(i).data(), col_size) = matrix.row(i);
  }

  return vec_2d;
}

}  // namespace cnn::util
