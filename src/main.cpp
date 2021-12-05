#include <fstream>
#include <map>

#include "cmdline.h"
#include "cnn/network.hpp"
#include "utils/constants.hpp"
#include "utils/helper.hpp"
#include "utils/load_dataset.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

std::map<string, std::map<char, size_t>> CHW_MAP = {
    {constants::dataset::MNIST, {{'C', 1}, {'H', 28}, {'W', 28}}},
    {constants::dataset::CIFAR10, {{'C', 3}, {'H', 32}, {'W', 32}}}};

template <typename T>
types::vector3d<double> flatten_images_per_channel(
    const types::vector4d<T>& images) {
  size_t n = images.size(), c = images.at(0).size(),
         h = images.at(0).at(0).size(), w = images.at(0).at(0).at(0).size();
  types::double3d flattened(n, types::double2d(c, vector<double>(h * w)));

  for (size_t n_i = 0; n_i < n; ++n_i) {
    for (size_t c_i = 0; c_i < c; ++c_i) {
      for (size_t h_i = 0; h_i < h; ++h_i) {
        for (size_t w_i = 0; w_i < w; ++w_i) {
          flattened[n_i][c_i][h_i * h + w_i] = images[n_i][c_i][h_i][w_i];
        }
      }
    }
  }

  return flattened;
}

int main(int argc, char* argv[]) {
  cmdline::parser parser;

  parser.add<string>("secret-prefix", 'P', "Prefix of filename of keys");
  parser.add<string>("dataset", 'D', "Dataset name", false,
                     constants::dataset::MNIST,
                     cmdline::oneof<string>(constants::dataset::MNIST,
                                            constants::dataset::CIFAR10));
  parser.add<string>("model", 'M', "Model name trained with PyTorch");
  parser.add("optimize", 0, "Whether to apply level reduction optimizations");
  parser.add<string>(
      "mode", 0,
      "Inference mode "
      "(single: channel-wise packing, batch: fixed-pixel packing)",
      false, constants::mode::SINGLE,
      cmdline::oneof<string>(constants::mode::SINGLE, constants::mode::BATCH));
  parser.parse_check(argc, argv);
  const string secret_fname_prefix = parser.get<string>("secret-prefix");
  const string dataset_name = parser.get<string>("dataset");
  const string model_name = parser.get<string>("model");
  const bool apply_opt = parser.exist("optimize");
  const string inference_mode = parser.get<string>("mode");

  const string base_model_path = constants::fname::TRAIN_MODEL_DIR + "/" +
                                 dataset_name + "/saved_models/" + model_name;
  const string model_structure_path = base_model_path + "-structure.json";
  const string model_params_path = base_model_path + "-params.h5";

  unique_ptr<ifstream> ifs_ptr;
  auto secrets_ifs =
      [&](const string& fname_suffix) -> const unique_ptr<ifstream>& {
    ifs_ptr.reset(new ifstream(constants::fname::SECRETS_DIR + "/" +
                                   secret_fname_prefix + fname_suffix,
                               ios::binary));
    return ifs_ptr;
  };

  /* Load seal params */
  seal::EncryptionParameters params;
  params.load(*secrets_ifs(constants::fname::PARAMS_SUFFIX));

  shared_ptr<seal::SEALContext> context(new seal::SEALContext(params));
  helper::he::print_parameters(context);
  cout << endl;

  /* Load seal keys */
  shared_ptr<seal::SecretKey> secret_key(new seal::SecretKey);
  secret_key->load(*context, *secrets_ifs(constants::fname::SECRET_KEY_SUFFIX));
  shared_ptr<seal::PublicKey> public_key(new seal::PublicKey);
  public_key->load(*context, *secrets_ifs(constants::fname::PUBLIC_KEY_SUFFIX));
  shared_ptr<seal::RelinKeys> relin_keys(new seal::RelinKeys);
  relin_keys->load(*context, *secrets_ifs(constants::fname::RELIN_KEYS_SUFFIX));
  shared_ptr<seal::GaloisKeys> galois_keys(new seal::GaloisKeys);
  galois_keys->load(*context,
                    *secrets_ifs(constants::fname::GALOIS_KEYS_SUFFIX));

  seal::CKKSEncoder encoder(*context);
  seal::Encryptor encryptor(*context, *public_key);
  seal::Decryptor decryptor(*context, *secret_key);

  const size_t log2_f = std::log2(params.coeff_modulus()[1].value() - 1) + 1;
  const double scale = static_cast<double>(static_cast<uint64_t>(1) << log2_f);
  cout << "scale: 2^" << log2_f << "(" << scale << ")" << endl;
  const size_t slot_count = encoder.slot_count();
  cout << "# of slots: " << slot_count << endl;
  cout << endl;

  /* Load test dataset for inference */
  cout << "Loading test images & labels..." << endl;
  types::float2d test_images = utils::load_test_images(dataset_name);
  vector<unsigned char> test_labels = utils::load_test_labels(dataset_name);

  const size_t input_n = test_images.size(),
               input_c = CHW_MAP[dataset_name]['C'],
               input_h = CHW_MAP[dataset_name]['H'],
               input_w = CHW_MAP[dataset_name]['W'],
               label_size = test_labels.size();

  cout << "Finish loading!" << endl;
  cout << "Number of test images: " << input_n << endl;
  cout << endl;

  return 0;
}
