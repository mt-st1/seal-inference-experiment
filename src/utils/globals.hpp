#pragma once

#include <utils/opt_option.hpp>
#include <vector>

extern OptOption OPT_OPTION;

/* memo of consumed multiplicative level that is used when load trained model */
extern std::size_t CONSUMED_LEVEL;

/* coefficients of polynomial activation */
extern std::vector<double> POLY_ACT_COEFFS;

/* highest degree coefficient of polynomial activation */
extern double POLY_ACT_HIGHEST_DEG_COEFF;

/* pooling multiply coefficient */
extern double CURRENT_POOL_MUL_COEFF;

/* if 'opt-act' option is true,
   multiply highest degree coeff of polynomial activation function to weight
   parameter of next linear layer of activation layer */
extern bool SHOULD_MUL_ACT_COEFF;

/* if 'opt-pool' option is true,
   multiply 1/(pool_height * pool_width) to parameter of next linear layer of
   average pooling layer */
extern bool SHOULD_MUL_POOL_COEFF;

/* supported types of activation function for inference on ciphertext */
enum EActivationType { SQUARE, DEG2_POLY_APPROX, DEG4_POLY_APPROX };
extern EActivationType ACTIVATION_TYPE;

// Rounding value for when encode too small value (depending on SEAL parameter)
// if fabs(target_encode_value) < ROUND_THRESHOLD, we change target_encode_value
// = ROUND_THRESHOLD * (target_encode_value/fabs(target_encode_value))
constexpr double ROUND_THRESHOLD = 1e-7;