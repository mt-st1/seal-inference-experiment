#include "network_builder.hpp"
#include "layers/layer_builder.hpp"
#include "utils/helper.hpp"

picojson::array load_layers(const std::string& model_structure_path) {
  picojson::object json_obj = helper::json::read_json(model_structure_path);
  return json_obj["structure"].get<picojson::array>();
}

namespace cnn {

Network NetworkBuilder::build(const std::string& model_structure_path,
                              const std::string& model_params_path) {
  Network network;

  picojson::array layers = load_layers(model_structure_path);
  for (picojson::array::const_iterator it = layers.cbegin(),
                                       layers_end = layers.cend();
       it != layers_end; ++it) {
    picojson::object layer = (*it).get<picojson::object>();
    network.add_layer(LayerBuilder::build(layer, model_params_path));
  }

  return network;
}

}  // namespace cnn

namespace cnn::encrypted {

Network NetworkBuilder::build(
    const std::string& model_structure_path,
    const std::string& model_params_path,
    const std::shared_ptr<helper::he::SealTool>& seal_tool) {
  Network network;

  picojson::array layers = load_layers(model_structure_path);
  for (picojson::array::const_iterator it = layers.cbegin(),
                                       layers_end = layers.cend();
       it != layers_end; ++it) {
    picojson::object layer = (*it).get<picojson::object>();
    network.add_layer(LayerBuilder::build(layer, model_params_path, seal_tool));
  }

  return network;
}

}  // namespace cnn::encrypted
