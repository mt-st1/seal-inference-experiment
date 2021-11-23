#include <fstream>

#include "cmdline.h"
#include "cnn/layers/conv_2d.hpp"
#include "cnn/layers/flatten.hpp"
#include "cnn/network.hpp"
#include "utils/constants.hpp"
#include "utils/helper.hpp"

using std::ifstream;
using std::ios;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

constexpr size_t INPUT_N = 1;
constexpr size_t INPUT_C = 1;
constexpr size_t INPUT_H = 4;
constexpr size_t INPUT_W = 4;
constexpr size_t FILTER_N = 1;
constexpr size_t FILTER_H = 2;
constexpr size_t FILTER_W = 2;
constexpr size_t STRIDE = 1;
constexpr size_t PADDING = 0;
constexpr size_t OUTPUT_H = std::floor(
    (static_cast<double>(INPUT_H + 2 * PADDING - FILTER_H) / STRIDE) + 1);
constexpr size_t OUTPUT_W = std::floor(
    (static_cast<double>(INPUT_W + 2 * PADDING - FILTER_W) / STRIDE) + 1);

types::double4d generate_inputs(const size_t& n,
                                const size_t& c,
                                const size_t& h,
                                const size_t w) {
  types::double4d inputs(
      n, types::double3d(c, types::double2d(h, vector<double>(w))));

  size_t input_hw_size = h * w;
  double input_cur = 0.0;
  double step_size = 1.0 / (input_hw_size - 1);

  std::cout << "<INPUT> " << n << " x " << c << " x " << h << " x " << w
            << std::endl;
  for (size_t h_i = 0; h_i < h; ++h_i) {
    for (size_t w_i = 0; w_i < w; ++w_i) {
      for (size_t n_i = 0; n_i < n; ++n_i) {
        for (size_t c_i = 0; c_i < c; ++c_i) {
          inputs[n_i][c_i][h_i][w_i] = input_cur;
        }
      }
      std::cout << "inputs[x][x][" << h_i << "][" << w_i << "]: " << input_cur
                << std::endl;
      input_cur += step_size;
    }
  }
  std::cout << std::endl;

  return inputs;
}

types::double4d generate_filters(const size_t& fn,
                                 const size_t& c,
                                 const size_t& fh,
                                 const size_t fw) {
  types::double4d filters(
      fn, types::double3d(c, types::double2d(fh, vector<double>(fw))));

  size_t filter_cur = 1;

  std::cout << "<FILTER> " << fn << " x " << c << " x " << fh << " x " << fw
            << std::endl;
  for (size_t c_i = 0; c_i < c; ++c_i) {
    for (size_t fh_i = 0; fh_i < fh; ++fh_i) {
      for (size_t fw_i = 0; fw_i < fw; ++fw_i) {
        for (size_t fn_i = 0; fn_i < fn; ++fn_i) {
          filters[fn_i][c_i][fh_i][fw_i] = filter_cur;
        }
        std::cout << "filter[x][" << c_i << "][" << fh_i << "][" << fw_i
                  << "]: " << filter_cur << std::endl;
        filter_cur++;
      }
    }
  }
  std::cout << std::endl;

  return filters;
}

template <typename T>
types::vector3d<T> flatten_images_per_channel(
    const types::vector4d<T>& images) {
  size_t n = images.size(), c = images.at(0).size(),
         h = images.at(0).at(0).size(), w = images.at(0).at(0).at(0).size();
  types::vector3d<T> result(n, types::double2d(c, vector<double>(h * w)));

  for (size_t n_i = 0; n_i < n; ++n_i) {
    for (size_t c_i = 0; c_i < c; ++c_i) {
      for (size_t h_i = 0; h_i < h; ++h_i) {
        for (size_t w_i = 0; w_i < w; ++w_i) {
          result[n_i][c_i][h_i * h + w_i] = images[n_i][c_i][h_i][w_i];
        }
      }
    }
  }

  return result;
}

int main(int argc, char* argv[]) {
  cmdline::parser parser;

  parser.add<string>("prefix", 0, "Prefix of filename of keys");
  parser.add<string>("model-structure", 0,
                     "Path to PyTorch trained model structure (json)");
  parser.add<string>("model-params", 0,
                     "Path to PyTorch trained model parameters (hdf5)");
  parser.parse_check(argc, argv);
  const string fname_prefix = parser.get<string>("prefix");
  const string model_structure_path = parser.get<string>("model-structure");
  const string model_params_path = parser.get<string>("model-params");

  unique_ptr<ifstream> ifs_ptr;
  auto secrets_ifs =
      [&](const string& fname_suffix) -> const unique_ptr<ifstream>& {
    ifs_ptr.reset(new ifstream(
        constants::fname::SECRETS_DIR + "/" + fname_prefix + fname_suffix,
        ios::binary));
    return ifs_ptr;
  };

  seal::EncryptionParameters params;
  params.load(*secrets_ifs(constants::fname::PARAMS_SUFFIX));

  shared_ptr<seal::SEALContext> context(new seal::SEALContext(params));
  helper::he::print_parameters(context);
  std::cout << std::endl;

  // load keys
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

  size_t log2_f = std::log2(params.coeff_modulus()[1].value() - 1) + 1;
  double scale = static_cast<double>(static_cast<uint64_t>(1) << log2_f);
  std::cout << "scale: 2^" << log2_f << "(" << scale << ")" << std::endl;
  size_t slot_count = encoder.slot_count();
  std::cout << "# of slots: " << slot_count << std::endl;
  std::cout << std::endl;

  // Normal prediction
  {
    types::double4d images =
        generate_inputs(INPUT_N, INPUT_C, INPUT_H, INPUT_W);
    types::double4d filters =
        generate_filters(FILTER_N, INPUT_C, FILTER_H, FILTER_W);
    vector<double> biases(FILTER_N, 0);

    cnn::Network network;
    network.add_layer(std::make_shared<cnn::Conv2d>(filters, biases));
    network.add_layer(std::make_shared<cnn::Flatten>());
    types::double2d pred_results = network.predict(images);

    for (size_t i = 0; i < pred_results.size(); ++i) {
      for (size_t j = 0; j < pred_results.at(0).size(); ++j) {
        std::cout << "pred_results[" << i << "][" << j
                  << "]: " << pred_results.at(i).at(j) << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // Encrypted prediction
  {
    types::double4d images =
        generate_inputs(INPUT_N, INPUT_C, INPUT_H, INPUT_W);
    types::double4d filters =
        generate_filters(FILTER_N, INPUT_C, FILTER_H, FILTER_W);
    vector<double> biases(FILTER_N, 0);

    types::double3d flattened_hw_inputs =
        flatten_images_per_channel<double>(images);
    seal::Plaintext pt;
    types::Ciphertext2d inputs_cts(INPUT_N, vector<seal::Ciphertext>(INPUT_C));
    for (size_t in; in < INPUT_N; ++in) {
      for (size_t ic; ic < INPUT_C; ++ic) {
        encoder.encode(flattened_hw_inputs[in][ic], scale, pt);
        encryptor.encrypt(pt, inputs_cts[in][ic]);
      }
    }

    size_t filter_hw_size = FILTER_H * FILTER_W;
    types::double4d slot_filters_values(
        FILTER_N, types::double3d(
                      INPUT_C, types::double2d(filter_hw_size,
                                               vector<double>(slot_count, 0))));
    for (size_t fn = 0; fn < FILTER_N; ++fn) {
      for (size_t ic = 0; ic < INPUT_C; ++ic) {
        for (size_t oh = 0; oh < OUTPUT_H; ++oh) {
          for (size_t ow = 0; ow < OUTPUT_W; ++ow) {
            for (size_t fh = 0; fh < FILTER_H; ++fh) {
              for (size_t fw = 0; fw < FILTER_W; ++fw) {
                slot_filters_values[fn][ic][fh * FILTER_H + fw]
                                   [oh * INPUT_H + ow] =
                                       filters[fn][ic][fh][fw];
              }
            }
          }
        }
      }
    }
    types::Plaintext3d filters_pts(
        FILTER_N,
        types::Plaintext2d(INPUT_C, vector<seal::Plaintext>(filter_hw_size)));
    for (size_t fn = 0; fn < FILTER_N; ++fn) {
      for (size_t ic = 0; ic < INPUT_C; ++ic) {
        for (size_t i = 0; i < filter_hw_size; ++i) {
          encoder.encode(slot_filters_values[fn][ic][i], scale,
                         filters_pts[fn][ic][i]);
        }
      }
    }

    types::double2d slot_biases_values(FILTER_N, vector<double>(slot_count, 0));
    for (size_t fn = 0; fn < FILTER_N; ++fn) {
      for (size_t oh = 0; oh < OUTPUT_H; ++oh) {
        for (size_t ow = 0; ow < OUTPUT_W; ++ow) {
          slot_biases_values[fn][oh * INPUT_H + ow] = biases[fn];
        }
      }
    }
    vector<seal::Plaintext> biases_pts(FILTER_N);
    for (size_t fn = 0; fn < FILTER_N; ++fn) {
      encoder.encode(slot_biases_values[fn], scale, biases_pts[fn]);
    }

    vector<int> filter_rotation_map(filter_hw_size);
    for (size_t i = 0; i < FILTER_H; ++i) {
      for (size_t j = 0; j < FILTER_W; ++j) {
        filter_rotation_map[i * FILTER_H + j] = i * INPUT_H + j;
      }
    }

    seal::Evaluator evaluator(*context);
    shared_ptr<helper::he::SealTool> seal_tool =
        std::make_shared<helper::he::SealTool>(evaluator, *relin_keys,
                                               *galois_keys, slot_count, scale);

    cnn::encrypted::Network enc_network;
    enc_network.add_layer(std::make_shared<cnn::encrypted::Conv2d>(
        filters_pts, biases_pts, filter_rotation_map, seal_tool));
    enc_network.add_layer(std::make_shared<cnn::encrypted::Flatten>(seal_tool));
    for (size_t i = 0; i < INPUT_N; ++i) {
      seal::Ciphertext enc_pred_result = enc_network.predict(inputs_cts[i]);

      seal::Plaintext result_pt;
      vector<double> decrypted_result;
      decryptor.decrypt(enc_pred_result, result_pt);
      encoder.decode(result_pt, decrypted_result);

      for (size_t hw = 0; hw < INPUT_H * INPUT_W; ++hw) {
        std::cout << "decrypted_result[" << hw << "]: " << decrypted_result[hw]
                  << std::endl;
      }
    }
  }

  return 0;
}
