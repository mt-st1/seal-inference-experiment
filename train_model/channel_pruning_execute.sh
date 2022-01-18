#/usr/bin/env bash
set -eu

yyyy=`date "+%Y"`
mm=`date "+%m"`
dd=`date "+%d"`
HH=`date "+%H"`
MM=`date "+%M"`

mnist_log_dir="./mnist/logs/${yyyy}/${mm}${dd}"
cifar_log_dir="./cifar-10/logs/${yyyy}/${mm}${dd}"
log_dir="./logs/${yyyy}/${mm}${dd}"

mnist_train_commands=(
  #########################################################
  # Square & Swish approx. (rg5_deg2, rg7_deg4)
  #########################################################
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act square --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-square-BN-prune-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act square --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/3layer_cnn-square-BN-prune-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg5_deg2 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg5_deg2-BN-prune-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg5_deg2 --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg5_deg2-BN-prune-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg7_deg4 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg7_deg4-BN-prune-${HH}${MM}.txt)"
  "(cd mnist && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./3layer_cnn.py --act swish_rg7_deg4 --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/3layer_cnn-swish_rg7_deg4-BN-prune-${HH}${MM}.txt)"
)

cifar_train_commands=(
  #########################################################
  # Square & Swish approx. (rg5_deg2, rg7_deg4)
  #########################################################
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-square-BN-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-square-BN-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-prune-${HH}${MM}.txt)"

  #########################################################
  # (GAP apply) Square & Swish approx. (rg5_deg2, rg7_deg4)
  #########################################################
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-square-BN-GAP-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --gap --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-square-BN-GAP-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-GAP-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --gap --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-GAP-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --gap --mode prune --round 1 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-GAP-prune-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --gap --mode prune --round 2 --epochs 30 >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-GAP-prune-${HH}${MM}.txt)"
)

[[ -d "${mnist_log_dir}" ]] || mkdir -p "${mnist_log_dir}"
[[ -d "${cifar_log_dir}" ]] || mkdir -p "${cifar_log_dir}"

for command in "${mnist_train_commands[@]}"
do
  eval "${command}"
done

for command in "${cifar_train_commands[@]}"
do
  eval "${command}"
done
