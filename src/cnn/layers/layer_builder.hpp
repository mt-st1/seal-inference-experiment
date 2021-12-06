#pragma once

#include "layer.hpp"

namespace cnn {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(picojson::object& layer,
                                      const std::string& model_params_path);
};

}  // namespace cnn

namespace cnn::encrypted {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool>& seal_tool);
};

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

class LayerBuilder {
public:
  static std::shared_ptr<Layer> build(
      picojson::object& layer,
      const std::string& model_params_path,
      const std::shared_ptr<helper::he::SealTool>& seal_tool);
};

}  // namespace cnn::encrypted::batch
