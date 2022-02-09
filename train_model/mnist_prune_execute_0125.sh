#/usr/bin/env bash
set -eu

yyyy=`date "+%Y"`
mm=`date "+%m"`
dd=`date "+%d"`
HH=`date "+%H"`
MM=`date "+%M"`

mnist_log_dir="./mnist/logs/${yyyy}/${mm}${dd}"
log_dir="./logs/${yyyy}/${mm}${dd}"

mnist_train_commands=(
  #########################################################
  # Square & Swish approx. (rg5_deg2, rg7_deg4)
  #########################################################
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act square --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-square-BN-prune_conv2_0.4-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg5_deg2 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg5_deg2-BN-prune_conv2_0.4-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg7_deg4 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg7_deg4-BN-prune_conv2_0.4-${HH}${MM}.txt)"
)

[[ -d "${mnist_log_dir}" ]] || mkdir -p "${mnist_log_dir}"

for command in "${mnist_train_commands[@]}"
do
  eval "${command}"
done
