#/usr/bin/env bash
set -eu

yyyy=`date "+%Y"`
mm=`date "+%m"`
dd=`date "+%d"`
HH=`date "+%H"`
MM=`date "+%M"`

cifar_log_dir="./cifar-10/logs/${yyyy}/${mm}${dd}"
log_dir="./logs/${yyyy}/${mm}${dd}"

cifar_train_commands=(
  #########################################################
  # (GAP apply) Square & Swish approx. (rg5_deg2, rg7_deg4)
  #########################################################
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-square-BN-GAP-prune_conv2_0.3-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.25-${HH}${MM}.txt)"
)

[[ -d "${cifar_log_dir}" ]] || mkdir -p "${cifar_log_dir}"

for command in "${cifar_train_commands[@]}"
do
  eval "${command}"
done
