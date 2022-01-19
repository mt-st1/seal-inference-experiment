
#/usr/bin/env bash
set -eu

commands=(
  # MNIST: Square [Level: 5]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-square-BN --model-params 3layer_cnn-square-BN-99.35_200epoch-1223_1032-params.h5 -A square --fuse-layer --mode single --images 20)"
  # MNIST: Swish approx. (rg5_deg2, rg7_deg4) [Level: 5 or 7]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-swish_rg5_deg2-BN --model-params 3layer_cnn-swish_rg5_deg2-BN-99.48_200epoch-1223_2212-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L7_50-30-60 -D mnist -M 3layer_cnn-swish_rg7_deg4-BN --model-params 3layer_cnn-swish_rg7_deg4-BN-99.52_200epoch-1224_0536-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --mode single --images 20)" # Transparent error has occurred

  # CIFAR-10(GAP): Square [Level: 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-params 5layer_cnn-square-BN-GAP-79.17_200epoch-0106_1855-params.h5 -A square --fuse-layer --opt-pool --mode single --images 20)"
  # CIFAR-10(GAP): Swish approx. (rg5_deg2, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-79.53_200epoch-0107_0014-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-80.91_200epoch-0107_0317-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
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
