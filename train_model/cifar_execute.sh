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
  # ReLU, Swish, Mish, Square
  #########################################################
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu --bn >> ${log_dir}/5layer_cnn-relu-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu --bn --gap >> ${log_dir}/5layer_cnn-relu-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish --bn >> ${log_dir}/5layer_cnn-swish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish --bn --gap >> ${log_dir}/5layer_cnn-swish-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish --bn >> ${log_dir}/5layer_cnn-mish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish --bn --gap >> ${log_dir}/5layer_cnn-mish-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn >> ${log_dir}/5layer_cnn-square-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act square --bn --gap >> ${log_dir}/5layer_cnn-square-BN-GAP-${HH}${MM}.txt)"

  #########################################################
  # ReLU approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4)
  #########################################################
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg5_deg2 --bn >> ${log_dir}/5layer_cnn-relu_rg5_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg5_deg2 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg5_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg7_deg2 --bn >> ${log_dir}/5layer_cnn-relu_rg7_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg7_deg2 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg7_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg5_deg4 --bn >> ${log_dir}/5layer_cnn-relu_rg5_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg5_deg4 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg5_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg7_deg4 --bn >> ${log_dir}/5layer_cnn-relu_rg7_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act relu_rg7_deg4 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg7_deg4-BN-GAP-${HH}${MM}.txt)"

  #########################################################
  # Swish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4)
  #########################################################
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg2 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg5_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg2 --bn >> ${log_dir}/5layer_cnn-swish_rg7_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg2 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg7_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg4 --bn >> ${log_dir}/5layer_cnn-swish_rg5_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg5_deg4 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg5_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act swish_rg7_deg4 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg7_deg4-BN-GAP-${HH}${MM}.txt)"

  #########################################################
  # Mish approx. (rg5_deg2, rg7_deg2, rg5_deg5, rg7_deg4)
  #########################################################
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg5_deg2 --bn >> ${log_dir}/5layer_cnn-mish_rg5_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg5_deg2 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg5_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg7_deg2 --bn >> ${log_dir}/5layer_cnn-mish_rg7_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg7_deg2 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg7_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg5_deg4 --bn >> ${log_dir}/5layer_cnn-mish_rg5_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg5_deg4 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg5_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg7_deg4 --bn >> ${log_dir}/5layer_cnn-mish_rg7_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./5layer_cnn.py --act mish_rg7_deg4 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg7_deg4-BN-GAP-${HH}${MM}.txt)"


  #########################################################
  # PreActResNet10
  #########################################################
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act relu >> ${log_dir}/preact_resnet10-relu-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act swish >> ${log_dir}/preact_resnet10-swish-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act mish >> ${log_dir}/preact_resnet10-mish-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act square >> ${log_dir}/preact_resnet10-square-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act relu_rg5_deg2 >> ${log_dir}/preact_resnet10-relu_rg5_deg2-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act relu_rg7_deg2 >> ${log_dir}/preact_resnet10-relu_rg7_deg2-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act swish_rg5_deg2 >> ${log_dir}/preact_resnet10-swish_rg5_deg2-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act swish_rg7_deg2 >> ${log_dir}/preact_resnet10-swish_rg7_deg2-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act mish_rg5_deg2 >> ${log_dir}/preact_resnet10-mish_rg5_deg2-${HH}${MM}.txt)"
  "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act mish_rg7_deg2 >> ${log_dir}/preact_resnet10-mish_rg7_deg2-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act relu_rg5_deg4 >> ${log_dir}/preact_resnet10-relu_rg5_deg4-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act relu_rg7_deg4 >> ${log_dir}/preact_resnet10-relu_rg7_deg4-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act swish_rg5_deg4 >> ${log_dir}/preact_resnet10-swish_rg5_deg4-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act swish_rg7_deg4 >> ${log_dir}/preact_resnet10-swish_rg7_deg4-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act mish_rg5_deg4 >> ${log_dir}/preact_resnet10-mish_rg5_deg4-${HH}${MM}.txt)"
  # "(cd cifar-10 && CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ./preact_resnet.py --model preact_resnet10 --act mish_rg7_deg4 >> ${log_dir}/preact_resnet10-mish_rg7_deg4-${HH}${MM}.txt)"
)

[[ -d "${cifar_log_dir}" ]] || mkdir -p "${cifar_log_dir}"

for command in "${cifar_train_commands[@]}"
do
  eval "${command}"
done
