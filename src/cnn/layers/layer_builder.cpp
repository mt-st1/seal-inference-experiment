// #include "layer_builder.hpp"
// #include "activation.hpp"
// #include "avg_pool_2d.hpp"
// #include "batch_norm.hpp"
// #include "conv_2d.hpp"
// #include "flatten.hpp"
// #include "linear.hpp"
// #include "picojson.h"

// #include <H5Cpp.h>

// namespace cnn {

// std::shared_ptr<Layer> LayerBuilder::build(
//     picojson::object& layer,
//     const std::string& model_params_path) {
//   const std::map<const std::string, std::shared_ptr<Layer> (*)(
//                                         picojson::object&, const
//                                         std::string&)>
//       build_layer_map{{CONV_2D_CLASS_NAME, build_conv_2d},
//                       {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
//                       {ACTIVATION_CLASS_NAME, build_activation},
//                       {BATCH_NORM_CLASS_NAME, build_batch_norm},
//                       {LINEAR_CLASS_NAME, build_linear},
//                       {FLATTEN_CLASS_NAME, build_flatten}};

//   const std::string layer_class_name =
//   layer["class_name"].get<std::string>(); if (auto map_iter =
//   build_layer_map.find(layer_class_name);
//       map_iter != build_layer_map.end()) {
//     picojson::object layer_info = layer["info"].get<picojson::object>();
//     return map_iter->second(layer_info, model_params_path);
//   } else {
//     throw runtime_error("\"" + layer_class_name +
//                         "\" is not registered as layer class");
//   }
// }

// std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
//                                      const string& model_params_path) {
//   return std::make_shared<Conv2d>();
// }

// std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
//                                          const string& model_params_path) {
//   return std::make_shared<AvgPool2d>();
// }

// std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
//                                         const string& model_params_path) {
//   return std::make_shared<Activation>();
// }

// std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
//                                         const string& model_params_path) {
//   return std::make_shared<BatchNorm>();
// }

// std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
//                                     const string& model_params_path) {
//   return std::make_shared<Linear>();
// }

// std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
//                                      const string& model_params_path) {
//   return std::make_shared<Flatten>();
// }

// }  // namespace cnn

// namespace cnn::encrypted {

// std::shared_ptr<Layer> LayerBuilder::build(
//     picojson::object& layer,
//     const std::string& model_params_path) {
//   const std::map<const std::string, std::shared_ptr<Layer> (*)(
//                                         picojson::object&, const
//                                         std::string&)>
//       build_layer_map{{CONV_2D_CLASS_NAME, build_conv_2d},
//                       {AVG_POOL_2D_CLASS_NAME, build_avg_pool_2d},
//                       {ACTIVATION_CLASS_NAME, build_activation},
//                       {BATCH_NORM_CLASS_NAME, build_batch_norm},
//                       {LINEAR_CLASS_NAME, build_linear},
//                       {FLATTEN_CLASS_NAME, build_flatten}};

//   const std::string layer_class_name =
//   layer["class_name"].get<std::string>(); if (auto map_iter =
//   build_layer_map.find(layer_class_name);
//       map_iter != build_layer_map.end()) {
//     picojson::object layer_info = layer["info"].get<picojson::object>();
//     return map_iter->second(layer_info, model_params_path);
//   }
// }

// std::shared_ptr<Layer> build_conv_2d(picojson::object& layer_info,
//                                      const string& model_params_path) {
//   return std::make_shared<Conv2d>();
// }

// std::shared_ptr<Layer> build_avg_pool_2d(picojson::object& layer_info,
//                                          const string& model_params_path) {
//   return std::make_shared<AvgPool2d>();
// }

// std::shared_ptr<Layer> build_activation(picojson::object& layer_info,
//                                         const string& model_params_path) {
//   return std::make_shared<Activation>();
// }

// std::shared_ptr<Layer> build_batch_norm(picojson::object& layer_info,
//                                         const string& model_params_path) {
//   return std::make_shared<BatchNorm>();
// }

// std::shared_ptr<Layer> build_linear(picojson::object& layer_info,
//                                     const string& model_params_path) {
//   return std::make_shared<Linear>();
// }

// std::shared_ptr<Layer> build_flatten(picojson::object& layer_info,
//                                      const string& model_params_path) {
//   return std::make_shared<Flatten>();
// }

// }  // namespace cnn::encrypted
