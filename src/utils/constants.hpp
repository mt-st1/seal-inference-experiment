#pragma once

#include <string>

namespace constants::fname {

extern const std::string DATASETS_DIR;
extern const std::string TRAIN_MODEL_DIR;
extern const std::string SECRETS_DIR;
extern const std::string PARAMS_SUFFIX;
extern const std::string SECRET_KEY_SUFFIX;
extern const std::string PUBLIC_KEY_SUFFIX;
extern const std::string RELIN_KEYS_SUFFIX;
extern const std::string GALOIS_KEYS_SUFFIX;

}  // namespace constants::fname

namespace constants::dataset {

extern const std::string MNIST;
extern const std::string CIFAR10;

}  // namespace constants::dataset

namespace constants::mode {

extern const std::string SINGLE;

extern const std::string BATCH;

}  // namespace constants::mode
