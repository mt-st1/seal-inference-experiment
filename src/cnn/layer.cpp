#include "layer.hpp"

namespace cnn {

Layer::Layer(const ELayerType& layer_type) : layer_type_(layer_type) {}
Layer::~Layer() {}

}  // namespace cnn

namespace cnn::encrypted {

Layer::Layer(const ELayerType& layer_type,
             const std::shared_ptr<helper::SealTool> seal_tool)
    : layer_type_(layer_type), seal_tool_(seal_tool) {}
Layer::~Layer() {}

}  // namespace cnn::encrypted
