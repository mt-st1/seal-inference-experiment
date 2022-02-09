#/usr/bin/env bash
set -eu

commands=(
  ##############
  # Channel-wise
  ##############
  # MNIST: Square [Level: 5]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-square-BN --model-params 3layer_cnn-square-BN-99.35_200epoch-1223_1032-params.h5 -A square --fuse-layer --mode single --images 10000)"
  # MNIST: Swish approx. (rg5_deg2, rg7_deg4) [Level: 5 or 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-swish_rg5_deg2-BN --model-params 3layer_cnn-swish_rg5_deg2-BN-99.48_200epoch-1223_2212-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --mode single --images 10000)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D mnist -M 3layer_cnn-swish_rg7_deg4-BN --model-params 3layer_cnn-swish_rg7_deg4-BN-99.52_200epoch-1224_0536-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --mode single --images 10000)"

  ##############
  # Channel-wise (Pruned model)
  ##############
  # MNIST: Square [Level: 5]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-square-BN --model-structure 3layer_cnn-square-BN-round1-98.61-structure.json --model-params 3layer_cnn-square-BN-round1-98.61-params.h5 -A square --fuse-layer --mode single --images 10000)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-square-BN --model-structure 3layer_cnn-square-BN-round2-96.68-structure.json --model-params 3layer_cnn-square-BN-round2-96.68-params.h5 -A square --fuse-layer --mode single --images 10000)"
  # MNIST: Swish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4) [Level: 5 or 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-swish_rg5_deg2-BN --model-structure 3layer_cnn-swish_rg5_deg2-BN-round1-98.83-structure.json --model-params 3layer_cnn-swish_rg5_deg2-BN-round1-98.83-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --mode single --images 10000)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-swish_rg5_deg2-BN --model-structure 3layer_cnn-swish_rg5_deg2-BN-round2-95.52-structure.json --model-params 3layer_cnn-swish_rg5_deg2-BN-round2-95.52-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --mode single --images 10000)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D mnist -M 3layer_cnn-swish_rg7_deg4-BN --model-structure 3layer_cnn-swish_rg7_deg4-BN-round1-97.77-structure.json --model-params 3layer_cnn-swish_rg7_deg4-BN-round1-97.77-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --mode single --images 10000)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D mnist -M 3layer_cnn-swish_rg7_deg4-BN --model-structure 3layer_cnn-swish_rg7_deg4-BN-round2-96.78-structure.json --model-params 3layer_cnn-swish_rg7_deg4-BN-round2-96.78-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --mode single --images 10000)"
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
