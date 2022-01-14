#/usr/bin/env bash
set -eu

commands=(
  # MNIST: Square [Level: 5]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L5_50-30 -D mnist -M 3layer_cnn-square-BN --model-params 3layer_cnn-square-BN-99.35_200epoch-1223_1032-params.h5 -A square --fuse-layer --mode batch -N 2)"
  # MNIST: ReLU approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 5 or 7]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L5_50-30 -D mnist -M 3layer_cnn-relu_rg5_deg2-BN --model-params 3layer_cnn-relu_rg5_deg2-BN-99.36_200epoch-1223_1247-params.h5 -A relu_rg5_deg2 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L5_50-30 -D mnist -M 3layer_cnn-relu_rg7_deg2-BN --model-params 3layer_cnn-relu_rg7_deg2-BN-99.49_200epoch-1223_1508-params.h5 -A relu_rg7_deg2 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L7_50-30 -D mnist -M 3layer_cnn-relu_rg5_deg4-BN --model-params 3layer_cnn-relu_rg5_deg4-BN-99.39_200epoch-1223_1730-params.h5 -A relu_rg5_deg4 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L7_50-30 -D mnist -M 3layer_cnn-relu_rg7_deg4-BN --model-params 3layer_cnn-relu_rg7_deg4-BN-99.43_200epoch-1223_1951-params.h5 -A relu_rg7_deg4 --fuse-layer --opt-act --mode batch -N 2)"
  # MNIST: Swish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 5 or 7]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L5_50-30 -D mnist -M 3layer_cnn-swish_rg5_deg2-BN --model-params 3layer_cnn-swish_rg5_deg2-BN-99.48_200epoch-1223_2212-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L5_50-30 -D mnist -M 3layer_cnn-swish_rg7_deg2-BN --model-params 3layer_cnn-swish_rg7_deg2-BN-99.42_200epoch-1224_0038-params.h5 -A swish_rg7_deg2 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L7_50-30 -D mnist -M 3layer_cnn-swish_rg5_deg4-BN --model-params 3layer_cnn-swish_rg5_deg4-BN-99.48_200epoch-1224_0307-params.h5 -A swish_rg5_deg4 --fuse-layer --opt-act --mode batch -N 2)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 ./bin/main -P N16384_L7_50-30 -D mnist -M 3layer_cnn-swish_rg7_deg4-BN --model-params 3layer_cnn-swish_rg7_deg4-BN-99.52_200epoch-1224_0536-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --mode batch -N 2)"

  # CIFAR-10: Square [Level: 8]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L8_50-30 -D cifar-10 -M 5layer_cnn-square-BN --model-params 5layer_cnn-square-BN-81.42_200epoch-1225_0027-params.h5 -A square --fuse-layer --opt-pool --mode batch -N 2)"
  # CIFAR-10: ReLU approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 8 or 11]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L8_50-30 -D cifar-10 -M 5layer_cnn-relu_rg5_deg2-BN --model-params 5layer_cnn-relu_rg5_deg2-BN-81.82_200epoch-1225_0233-params.h5 -A relu_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L8_50-30 -D cifar-10 -M 5layer_cnn-relu_rg7_deg2-BN --model-params 5layer_cnn-relu_rg7_deg2-BN-81.21_200epoch-1225_0435-params.h5 -A relu_rg7_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L11_50-30 -D cifar-10 -M 5layer_cnn-relu_rg5_deg4-BN --model-params 5layer_cnn-relu_rg5_deg4-BN-83.35_200epoch-1225_0642-params.h5 -A relu_rg5_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L11_50-30 -D cifar-10 -M 5layer_cnn-relu_rg7_deg4-BN --model-params 5layer_cnn-relu_rg7_deg4-BN-83.30_200epoch-1225_0846-params.h5 -A relu_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  # CIFAR-10: Swish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 8 or 11]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L8_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN --model-params 5layer_cnn-swish_rg5_deg2-BN-81.84_200epoch-1225_1046-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L8_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg2-BN --model-params 5layer_cnn-swish_rg7_deg2-BN-81.70_200epoch-1225_1246-params.h5 -A swish_rg7_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L11_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg4-BN --model-params 5layer_cnn-swish_rg5_deg4-BN-83.77_200epoch-1225_1456-params.h5 -A swish_rg5_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L11_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN --model-params 5layer_cnn-swish_rg7_deg4-BN-83.81_200epoch-1225_1703-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"

  # CIFAR-10(GAP): Square [Level: 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-params 5layer_cnn-square-BN-GAP-79.17_200epoch-0106_1855-params.h5 -A square --fuse-layer --opt-pool --mode batch -N 2)"
  # CIFAR-10(GAP): ReLU approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-relu_rg5_deg2-BN-GAP --model-params 5layer_cnn-relu_rg5_deg2-BN-GAP-78.79_200epoch-0106_2012-params.h5 -A relu_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-relu_rg7_deg2-BN-GAP --model-params 5layer_cnn-relu_rg7_deg2-BN-GAP-78.29_200epoch-0106_2112-params.h5 -A relu_rg7_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-relu_rg5_deg4-BN-GAP --model-params 5layer_cnn-relu_rg5_deg4-BN-GAP-79.48_200epoch-0106_2214-params.h5 -A relu_rg5_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-relu_rg7_deg4-BN-GAP --model-params 5layer_cnn-relu_rg7_deg4-BN-GAP-79.95_200epoch-0106_2316-params.h5 -A relu_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  # CIFAR-10(GAP): Swish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-79.53_200epoch-0107_0014-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg2-BN-GAP --model-params 5layer_cnn-swish_rg7_deg2-BN-GAP-78.40_200epoch-0107_0120-params.h5 -A swish_rg7_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg4-BN-GAP --model-params 5layer_cnn-swish_rg5_deg4-BN-GAP-80.78_200epoch-0107_0217-params.h5 -A swish_rg5_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-80.91_200epoch-0107_0317-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
)

if [ $# -ne 1 ]; then
  echo "Please specify log file name"
  echo "usage: ./execute.sh ${LOG_FILE_NAME}"
  exit 1
fi

log_path="./logs/"
log_file_name=$1
# log_file_path="${log_path}main_log.txt"
log_file_path="${log_path}${log_file_name}"

[[ -d "${log_path}" ]] || mkdir "${log_path}"
[[ -f "${log_file_path}" ]] || touch "${log_file_path}"

for command in "${commands[@]}"
do
  echo "====================================================================================================================================================================================" >> "${log_file_path}"
  echo "{" >> "${log_file_path}"
  /bin/echo -n "[$(date)]" >> "${log_file_path}"
  echo ": START \"${command}\"" >> "${log_file_path}"
  eval "${command}" &>> "${log_file_path}"
  /bin/echo -n "[$(date)]" >> "${log_file_path}"
  echo ": FINISH \"${command}\"" >> "${log_file_path}"
  echo "}" >> "${log_file_path}"
  echo "====================================================================================================================================================================================" >> "${log_file_path}"
  echo -e "\n" >> "${log_file_path}"
done
