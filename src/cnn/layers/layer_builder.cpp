#include "layer_builder.hpp"
#include "activation.hpp"
#include "avg_pool_2d.hpp"
#include "batch_norm.hpp"
#include "conv_2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "picojson.h"

#include <H5Cpp.h>

namespace cnn {

std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
                                     const std::string& model_params_path);
std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
                                         const std::string& model_params_path);
std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
                                        const std::string& model_params_path);
std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
                                        const std::string& model_params_path);
std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
                                    const std::string& model_params_path);
std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
                                     const std::string& model_params_path);

const std::map<const std::string,
               std::shared_ptr<Layer> (*)(picojson::object&,
                                          const std::string&)>
    BUILD_LAYER_MAP{{CONV_2D_CLASS_NAME, build_conv_2d},
                    {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
                    {ACTIVATION_CLASS_NAME, build_activation},
                    {BATCH_NORM_CLASS_NAME, build_batch_norm},
                    {LINEAR_CLASS_NAME, build_linear},
                    {FLATTEN_CLASS_NAME, build_flatten}};

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    const std::string& model_params_path) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  if (auto map_iter = BUILD_LAYER_MAP.find(layer_class_name);
      map_iter != BUILD_LAYER_MAP.end()) {
    picojson::object layer_info = layer["info"].get<picojson::object>();
    const std::string layer_name = layer_info["name"].get<std::string>();
    std::cout << "Building " << layer_name << "..." << std::endl;

    return map_iter->second(layer_info, model_params_path);
  } else {
    throw std::runtime_error("\"" + layer_class_name +
                             "\" is not registered as layer class");
  }
}

std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
                                     const std::string& model_params_path) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  const size_t filter_size = layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const size_t filter_h = filter_hw[0].get<double>();
  const size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const size_t stride_h = stride_hw[0].get<double>();
  const size_t stride_w = stride_hw[1].get<double>();
  const picojson::array padding_hw =
      layer_info["padding"].get<picojson::array>();
  const size_t padding_h = padding_hw[0].get<double>();
  const size_t padding_w = padding_hw[1].get<double>();

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  types::float4d filters(
      weight_shape[0],
      types::float3d(weight_shape[1],
                     types::float2d(weight_shape[2],
                                    std::vector<float>(weight_shape[3]))));
  std::vector<float> biases(weight_shape[0]);

  weight_data.read(filters.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  return std::make_shared<Conv2d>(
      filters, biases, std::make_pair(stride_h, stride_w),
      std::make_pair("", std::make_pair(padding_h, padding_w)));
}

std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
                                         const std::string& model_params_path) {
  return std::make_shared<AvgPool2d>();
}

std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
                                        const std::string& model_params_path) {
  return std::make_shared<Activation>();
}

std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
                                        const std::string& model_params_path) {
  return std::make_shared<BatchNorm>();
}

std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
                                    const std::string& model_params_path) {
  return std::make_shared<Linear>();
}

std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
                                     const std::string& model_params_path) {
  return std::make_shared<Flatten>();
}

}  // namespace cnn

namespace cnn::encrypted {

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);
std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);
std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);
std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);
std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);
std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool);

const std::map<const std::string,
               std::shared_ptr<Layer> (*)(
                   picojson::object&,
                   const std::string&,
                   const std::shared_ptr<helper::he::SealTool>&)>
    BUILD_LAYER_MAP{{CONV_2D_CLASS_NAME, build_conv_2d},
                    {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
                    {ACTIVATION_CLASS_NAME, build_activation},
                    {BATCH_NORM_CLASS_NAME, build_batch_norm},
                    {LINEAR_CLASS_NAME, build_linear},
                    {FLATTEN_CLASS_NAME, build_flatten}};

std::shared_ptr<Layer> LayerBuilder::build(
    picojson::object& layer,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  const std::string layer_class_name = layer["class_name"].get<std::string>();
  if (auto map_iter = BUILD_LAYER_MAP.find(layer_class_name);
      map_iter != BUILD_LAYER_MAP.end()) {
    picojson::object layer_info = layer["info"].get<picojson::object>();
    const std::string layer_name = layer_info["name"].get<std::string>();
    std::cout << "Building " << layer_name << "..." << std::endl;

    return map_iter->second(layer_info, model_params_path, seal_tool);
  } else {
    throw std::runtime_error("\"" + layer_class_name +
                             "\" is not registered as layer class");
  }
}

std::shared_ptr<Layer> build_conv_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  /* Read structure info */
  const std::string layer_name = layer_info["name"].get<std::string>();
  const size_t filter_size = layer_info["filters"].get<double>();
  const picojson::array filter_hw =
      layer_info["kernel_size"].get<picojson::array>();
  const size_t filter_h = filter_hw[0].get<double>();
  const size_t filter_w = filter_hw[1].get<double>();
  const picojson::array stride_hw = layer_info["stride"].get<picojson::array>();
  const size_t stride_h = stride_hw[0].get<double>();
  const size_t stride_w = stride_hw[1].get<double>();
  const picojson::array padding_hw =
      layer_info["padding"].get<picojson::array>();
  const size_t padding_h = padding_hw[0].get<double>();
  const size_t padding_w = padding_hw[1].get<double>();

  /* Read params data */
  H5::H5File params_file(model_params_path, H5F_ACC_RDONLY);
  H5::Group group = params_file.openGroup("/" + layer_name);
  H5::DataSet weight_data = group.openDataSet("weight");
  H5::DataSet bias_data = group.openDataSet("bias");

  H5::DataSpace weight_space = weight_data.getSpace();
  int weight_rank = weight_space.getSimpleExtentNdims();
  hsize_t weight_shape[weight_rank];
  int ndims = weight_space.getSimpleExtentDims(weight_shape);

  types::float4d filters(
      weight_shape[0],
      types::float3d(weight_shape[1],
                     types::float2d(weight_shape[2],
                                    std::vector<float>(weight_shape[3]))));
  std::vector<float> biases(weight_shape[0]);

  weight_data.read(filters.data(), H5::PredType::NATIVE_FLOAT);
  bias_data.read(biases.data(), H5::PredType::NATIVE_FLOAT);

  return std::make_shared<Conv2d>();
}

std::shared_ptr<Layer> build_avg_pool_2d(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  return std::make_shared<AvgPool2d>();
}

std::shared_ptr<Layer> build_activation(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  return std::make_shared<Activation>();
}

std::shared_ptr<Layer> build_batch_norm(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  return std::make_shared<BatchNorm>();
}

std::shared_ptr<Layer> build_linear(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  return std::make_shared<Linear>();
}

std::shared_ptr<Layer> build_flatten(
    picojson::object& layer_info,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  return std::make_shared<Flatten>();
}

}  // namespace cnn::encrypted
