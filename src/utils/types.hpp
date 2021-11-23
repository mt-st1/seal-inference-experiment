#pragma once

#include <seal/seal.h>
#include <vector>

namespace types {

template <typename T>
using vector2d = std::vector<std::vector<T>>;

template <typename T>
using vector3d = std::vector<std::vector<std::vector<T>>>;

template <typename T>
using vector4d = std::vector<std::vector<std::vector<std::vector<T>>>>;

template <typename T>
using vector5d =
    std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>;

template <typename T>
using vector6d = std::vector<
    std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>>;

using double2d = vector2d<double>;
using double3d = vector3d<double>;
using double4d = vector4d<double>;
using double5d = vector5d<double>;
using double6d = vector6d<double>;

using Plaintext2d = vector2d<seal::Plaintext>;
using Plaintext3d = vector3d<seal::Plaintext>;

using Ciphertext2d = vector2d<seal::Ciphertext>;
using Ciphertext3d = vector3d<seal::Ciphertext>;

}  // namespace types
