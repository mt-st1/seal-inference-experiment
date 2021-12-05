#include "constants.hpp"

namespace constants::fname {

const std::string DATASETS_DIR = "datasets";
const std::string TRAIN_MODEL_DIR = "train_model";
const std::string SECRETS_DIR = "secrets";
const std::string PARAMS_SUFFIX = "_params";
const std::string SECRET_KEY_SUFFIX = "_secretKey";
const std::string PUBLIC_KEY_SUFFIX = "_publicKey";
const std::string RELIN_KEYS_SUFFIX = "_relinKeys";
const std::string GALOIS_KEYS_SUFFIX = "_galoisKeys";

}  // namespace constants::fname

namespace constants::dataset {

const std::string MNIST = "mnist";
const std::string CIFAR10 = "cifar-10";

}  // namespace constants::dataset

namespace constants::mode {

const std::string SINGLE = "single";
const std::string BATCH = "batch";

}  // namespace constants::mode
