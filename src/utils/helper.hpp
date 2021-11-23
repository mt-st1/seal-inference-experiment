#pragma once

#include <seal/seal.h>

namespace helper::seal {

class SealTool {
public:
  SealTool(seal::Evaluator& evaluator,
           seal::RelinKeys& relin_keys,
           seal::GaloisKeys& galois_keys,
           const std::size_t slot_count,
           const double scale);
  ~SealTool() = default;

  const seal::Evaluator& evaluator() const { return evaluator_; };
  const seal::RelinKeys& relin_keys() const { return relin_keys_; };
  const seal::GaloisKeys& galois_keys() const { return galois_keys_; };
  size_t slot_count() const { return slot_count_; };
  size_t scale() const { return scale_; };

private:
  seal::Evaluator& evaluator_;
  seal::RelinKeys& relin_keys_;
  seal::GaloisKeys& galois_keys_;
  std::size_t slot_count_;
  double scale_;
};

/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const std::shared_ptr<seal::SEALContext>& context);

}  // namespace helper::seal
