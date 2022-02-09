#/usr/bin/env bash
set -eu

commands=(
  ##############
  # Batch-axis (Pruned model)
  ##############
  # CIFAR-10(GAP): Square [Level: 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-params.h5 -A square --fuse-layer --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-params.h5 -A square --fuse-layer --opt-pool --mode batch -N 2)"
  # CIFAR-10(GAP): Swish approx. (rg5_deg2, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode batch -N 2)"

  ##############
  # Channel-wise (Pruned model)
  ##############
  # CIFAR-10(GAP): Square [Level: 7]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-params.h5 -A square --fuse-layer --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-params.h5 -A square --fuse-layer --opt-pool --mode single --images 20)"
  # CIFAR-10(GAP): Swish approx. (rg5_deg2, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=72 /usr/bin/time -v ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"

  ##############
  # Channel-wise (Pruned model) (18 threads)
  ##############
  # CIFAR-10(GAP): Square [Level: 7]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.1-round1-72.73-params.h5 -A square --fuse-layer --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-square-BN-GAP --model-structure 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-structure.json --model-params 5layer_cnn-square-BN-GAP-prune_conv2_0.2-round1-66.70-params.h5 -A square --fuse-layer --opt-pool --mode single --images 20)"
  # CIFAR-10(GAP): Swish approx. (rg5_deg2, rg7_deg4) [Level: 7 or 10]
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.1-round1-73.93-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L7_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg5_deg2-BN-GAP --model-structure 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-structure.json --model-params 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round1-64.94-params.h5 -A swish_rg5_deg2 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.1-round1-76.69-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.2-round1-73.18-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
  "(OMP_NUM_THREADS=18 /usr/bin/time -v numactl -N 3 -m 3 ./bin/main -P N16384_L10_50-30-60 -D cifar-10 -M 5layer_cnn-swish_rg7_deg4-BN-GAP --model-structure 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-structure.json --model-params 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.3-round1-65.20-params.h5 -A swish_rg7_deg4 --fuse-layer --opt-act --opt-pool --mode single --images 20)"
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
