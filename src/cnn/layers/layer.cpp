#include "layer.hpp"

namespace cnn {

Layer::Layer(const ELayerType& layer_type) : layer_type_(layer_type) {}
Layer::~Layer() {}

}  // namespace cnn

namespace cnn::encrypted {

Layer::Layer(const ELayerType& layer_type,
             const std::shared_ptr<helper::he::SealTool> seal_tool)
    : layer_type_(layer_type), seal_tool_(seal_tool) {}
Layer::Layer() {}
Layer::~Layer() {}

}  // namespace cnn::encrypted

namespace cnn::encrypted::batch {

Layer::Layer(const ELayerType& layer_type,
             const std::shared_ptr<helper::he::SealTool> seal_tool)
    : layer_type_(layer_type), seal_tool_(seal_tool) {}
Layer::Layer() {}
Layer::~Layer() {}

}  // namespace cnn::encrypted::batch
