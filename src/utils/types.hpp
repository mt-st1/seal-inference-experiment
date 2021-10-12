#pragma once

#include <vector>
#include <seal/seal.h>

namespace types {

template <typename T>
using vector2D = std::vector<std::vector<T>>;
template <typename T>
using vector3D = std::vector<std::vector<std::vector<T>>>;
template <typename T>
using vector4D = std::vector<std::vector<std::vector<std::vector<T>>>>;

using double2D = vector2D<double>;
using double3D = vector3D<double>;
using double4D = vector4D<double>;

using Plaintext2D = vector2D<seal::Plaintext>;
using Plaintext3D = vector3D<seal::Plaintext>;

using Ciphertext2D = vector2D<seal::Ciphertext>;
using Ciphertext3D = vector3D<seal::Ciphertext>;

} // types
