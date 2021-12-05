#include "helper.hpp"

#include <fstream>

namespace helper::he {

SealTool::SealTool(seal::Evaluator& evaluator,
                   seal::RelinKeys& relin_keys,
                   seal::GaloisKeys& galois_keys,
                   const std::size_t slot_count,
                   const double scale)
    : evaluator_(evaluator),
      relin_keys_(relin_keys),
      galois_keys_(galois_keys),
      slot_count_(slot_count),
      scale_(scale) {}

void print_parameters(const std::shared_ptr<seal::SEALContext>& context) {
  auto& context_data = *context->key_context_data();

  /*
  Which scheme are we using?
  */
  std::string scheme_name;
  switch (context_data.parms().scheme()) {
    case seal::scheme_type::bfv:
      scheme_name = "BFV";
      break;
    case seal::scheme_type::ckks:
      scheme_name = "CKKS";
      break;
    default:
      throw std::invalid_argument("unsupported scheme");
  }
  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: " << scheme_name << std::endl;
  std::cout << "|   poly_modulus_degree: "
            << context_data.parms().poly_modulus_degree() << std::endl;

  /*
  Print the size of the true (product) coefficient modulus.
  */
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_modulus_size = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_modulus_size - 1; ++i) {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;

  /*
  For the BFV scheme print the plain_modulus parameter.
  */
  if (context_data.parms().scheme() == seal::scheme_type::bfv) {
    std::cout << "|   plain_modulus: "
              << context_data.parms().plain_modulus().value() << std::endl;
  }

  std::cout << "\\" << std::endl;
}

}  // namespace helper::he

namespace helper::json {

/**
 * @brief Get picojson object from JSON file
 * @param file_path JSON file path
 * @return picojson::object
 * @throws std::runtime_error if fail to read JSON file or fail to parse
 */
picojson::object read_json(const std::string& file_path) {
  std::ifstream ifs(file_path, std::ios::in);
  if (ifs.fail()) {
    throw std::runtime_error("Failed to read JSON file (" + file_path + ")");
  }

  std::istreambuf_iterator<char> it(ifs);
  std::istreambuf_iterator<char> last;
  const std::string json_str(it, last);
  ifs.close();

  picojson::value val;
  const std::string err = picojson::parse(val, json_str);
  if (const std::string err = picojson::parse(val, json_str); !err.empty()) {
    throw std::runtime_error("Failed to parse JSON (" + err + ")");
  }

  return val.get<picojson::object>();
}

}  // namespace helper::json
