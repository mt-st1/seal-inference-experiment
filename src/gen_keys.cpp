#include <seal/seal.h>
#include <experimental/filesystem>
#include <fstream>

#include "cmdline.h"
#include "utils/constants.hpp"

using std::ios;
using std::ofstream;
using std::size_t;
using std::string;
using std::unique_ptr;
using std::vector;
namespace fs = std::experimental::filesystem;

vector<int> generate_log2_modulus(size_t level,
                                  size_t log2_q0,
                                  size_t log2_qi,
                                  size_t log2_ql) {
  vector<int> modulus(level + 2);
  modulus[0] = log2_q0;
  for (size_t i = 1; i <= level; ++i) {
    modulus[i] = log2_qi;
  }
  modulus[level + 1] = log2_ql;

  return modulus;
}

int main(int argc, char* argv[]) {
  cmdline::parser parser;

  parser.add<size_t>("poly-deg", 'N', "Degree of polynomial ring");
  parser.add<size_t>("level", 'L',
                     "Initial level of ciphertext (Multiplicative depth)");
  parser.add<size_t>("q0", 0, "Bit number of the first prime in coeff_modulus");
  parser.add<size_t>("qi", 0,
                     "Bit number of intermediate primes in coeff_modulus");
  parser.add<size_t>("ql", 0, "Bit number of the last prime in coeff_modulus");
  parser.add<string>("prefix", 0, "Prefix of the generating file name");

  parser.parse_check(argc, argv);
  const size_t poly_modulus_degree = parser.get<size_t>("poly-deg");
  const size_t level = parser.get<size_t>("level");
  const size_t log2_q0 = parser.get<size_t>("q0");
  const size_t log2_qi = parser.get<size_t>("qi");
  const size_t log2_ql = parser.get<size_t>("ql");
  const string fname_prefix = parser.get<string>("prefix");

  vector<int> prime_bit_sizes =
      generate_log2_modulus(level, log2_q0, log2_qi, log2_ql);

  seal::EncryptionParameters params(seal::scheme_type::ckks);
  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(
      seal::CoeffModulus::Create(poly_modulus_degree, prime_bit_sizes));

  seal::SEALContext context(params);
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  seal::RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  seal::GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  unique_ptr<ofstream> ofs_ptr;
  auto secrets_ofs =
      [&](const string& fname_suffix) -> const unique_ptr<ofstream>& {
    ofs_ptr.reset(new ofstream(
        constants::fname::SECRETS_DIR + "/" + fname_prefix + fname_suffix,
        ios::binary));
    return ofs_ptr;
  };

  if (!fs::exists(constants::fname::SECRETS_DIR))
    fs::create_directory(constants::fname::SECRETS_DIR);

  params.save(*secrets_ofs(constants::fname::PARAMS_SUFFIX));
  secret_key.save(*secrets_ofs(constants::fname::SECRET_KEY_SUFFIX));
  public_key.save(*secrets_ofs(constants::fname::PUBLIC_KEY_SUFFIX));
  relin_keys.save(*secrets_ofs(constants::fname::RELIN_KEYS_SUFFIX));
  galois_keys.save(*secrets_ofs(constants::fname::GALOIS_KEYS_SUFFIX));

  return 0;
}
