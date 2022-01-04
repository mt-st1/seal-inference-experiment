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
  # ReLU, Swish, Mish, Square
  #########################################################
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu >> ${log_dir}/3layer_cnn-relu-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu --bn >> ${log_dir}/3layer_cnn-relu-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish >> ${log_dir}/3layer_cnn-swish-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish --bn >> ${log_dir}/3layer_cnn-swish-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish >> ${log_dir}/3layer_cnn-mish-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish --bn >> ${log_dir}/3layer_cnn-mish-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square >> ${log_dir}/3layer_cnn-square-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square --bn >> ${log_dir}/3layer_cnn-square-BN-${HH}${MM}.txt)"

  # Ghost Module
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu --ghost --bn >> ${log_dir}/3layer_cnn_ghost-relu-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish --ghost --bn >> ${log_dir}/3layer_cnn_ghost-swish-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish --ghost --bn >> ${log_dir}/3layer_cnn_ghost-mish-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square --ghost --bn >> ${log_dir}/3layer_cnn_ghost-square-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu --bn --do >> ${log_dir}/3layer_cnn-relu-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish --bn --do >> ${log_dir}/3layer_cnn-swish-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish --bn --do >> ${log_dir}/3layer_cnn-mish-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square --bn --do >> ${log_dir}/3layer_cnn-square-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu --bn --gap >> ${log_dir}/3layer_cnn-relu-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish --bn --gap >> ${log_dir}/3layer_cnn-swish-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish --bn --gap >> ${log_dir}/3layer_cnn-mish-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square --bn --gap >> ${log_dir}/3layer_cnn-square-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu --bn --do --gap >> ${log_dir}/3layer_cnn-relu-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish --bn --do --gap >> ${log_dir}/3layer_cnn-swish-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish --bn --do --gap >> ${log_dir}/3layer_cnn-mish-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act square --bn --do --gap >> ${log_dir}/3layer_cnn-square-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # ReLU approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg4 --bn >> ${log_dir}/3layer_cnn-relu_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg4 --bn >> ${log_dir}/3layer_cnn-relu_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg2 --bn >> ${log_dir}/3layer_cnn-relu_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg2 --bn >> ${log_dir}/3layer_cnn-relu_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg4 --bn --do >> ${log_dir}/3layer_cnn-relu_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg4 --bn --do >> ${log_dir}/3layer_cnn-relu_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg2 --bn --do >> ${log_dir}/3layer_cnn-relu_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg2 --bn --do >> ${log_dir}/3layer_cnn-relu_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg4 --bn --gap >> ${log_dir}/3layer_cnn-relu_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg4 --bn --gap >> ${log_dir}/3layer_cnn-relu_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg2 --bn --gap >> ${log_dir}/3layer_cnn-relu_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg2 --bn --gap >> ${log_dir}/3layer_cnn-relu_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-relu_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-relu_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg4_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-relu_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act relu_rg6_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-relu_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # Swish approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg4 --bn >> ${log_dir}/3layer_cnn-swish_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg4 --bn >> ${log_dir}/3layer_cnn-swish_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg2 --bn >> ${log_dir}/3layer_cnn-swish_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg2 --bn >> ${log_dir}/3layer_cnn-swish_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg4 --bn --do >> ${log_dir}/3layer_cnn-swish_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg4 --bn --do >> ${log_dir}/3layer_cnn-swish_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg2 --bn --do >> ${log_dir}/3layer_cnn-swish_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg2 --bn --do >> ${log_dir}/3layer_cnn-swish_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg4 --bn --gap >> ${log_dir}/3layer_cnn-swish_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg4 --bn --gap >> ${log_dir}/3layer_cnn-swish_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg2 --bn --gap >> ${log_dir}/3layer_cnn-swish_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg2 --bn --gap >> ${log_dir}/3layer_cnn-swish_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-swish_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-swish_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg4_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-swish_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act swish_rg6_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-swish_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # Mish approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg4 --bn >> ${log_dir}/3layer_cnn-mish_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg4 --bn >> ${log_dir}/3layer_cnn-mish_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg2 --bn >> ${log_dir}/3layer_cnn-mish_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg2 --bn >> ${log_dir}/3layer_cnn-mish_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg4 --bn --do >> ${log_dir}/3layer_cnn-mish_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg4 --bn --do >> ${log_dir}/3layer_cnn-mish_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg2 --bn --do >> ${log_dir}/3layer_cnn-mish_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg2 --bn --do >> ${log_dir}/3layer_cnn-mish_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg4 --bn --gap >> ${log_dir}/3layer_cnn-mish_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg4 --bn --gap >> ${log_dir}/3layer_cnn-mish_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg2 --bn --gap >> ${log_dir}/3layer_cnn-mish_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg2 --bn --gap >> ${log_dir}/3layer_cnn-mish_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-mish_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg4 --bn --do --gap >> ${log_dir}/3layer_cnn-mish_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg4_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-mish_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  "(cd mnist && ../.venv/bin/python ./3layer_cnn.py --act mish_rg6_deg2 --bn --do --gap >> ${log_dir}/3layer_cnn-mish_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"
)

cifar_train_commands=(
  #########################################################
  # ReLU, Swish, Mish, Square
  #########################################################
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu >> ${log_dir}/5layer_cnn-relu-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu --bn >> ${log_dir}/5layer_cnn-relu-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish >> ${log_dir}/5layer_cnn-swish-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish --bn >> ${log_dir}/5layer_cnn-swish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish >> ${log_dir}/5layer_cnn-mish-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish --bn >> ${log_dir}/5layer_cnn-mish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square >> ${log_dir}/5layer_cnn-square-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square --bn >> ${log_dir}/5layer_cnn-square-BN-${HH}${MM}.txt)"

  # Ghost Module
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu --ghost --bn >> ${log_dir}/5layer_cnn_ghost-relu-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish --ghost --bn >> ${log_dir}/5layer_cnn_ghost-swish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish --ghost --bn >> ${log_dir}/5layer_cnn_ghost-mish-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square --ghost --bn >> ${log_dir}/5layer_cnn_ghost-square-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu --bn --do >> ${log_dir}/5layer_cnn-relu-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish --bn --do >> ${log_dir}/5layer_cnn-swish-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish --bn --do >> ${log_dir}/5layer_cnn-mish-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square --bn --do >> ${log_dir}/5layer_cnn-square-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu --bn --gap >> ${log_dir}/5layer_cnn-relu-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish --bn --gap >> ${log_dir}/5layer_cnn-swish-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish --bn --gap >> ${log_dir}/5layer_cnn-mish-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square --bn --gap >> ${log_dir}/5layer_cnn-square-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu --bn --do --gap >> ${log_dir}/5layer_cnn-relu-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish --bn --do --gap >> ${log_dir}/5layer_cnn-swish-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish --bn --do --gap >> ${log_dir}/5layer_cnn-mish-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act square --bn --do --gap >> ${log_dir}/5layer_cnn-square-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # ReLU approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg4 --bn >> ${log_dir}/5layer_cnn-relu_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg4 --bn >> ${log_dir}/5layer_cnn-relu_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg2 --bn >> ${log_dir}/5layer_cnn-relu_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg2 --bn >> ${log_dir}/5layer_cnn-relu_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg4 --bn --do >> ${log_dir}/5layer_cnn-relu_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg4 --bn --do >> ${log_dir}/5layer_cnn-relu_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg2 --bn --do >> ${log_dir}/5layer_cnn-relu_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg2 --bn --do >> ${log_dir}/5layer_cnn-relu_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg4 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg4 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg2 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg2 --bn --gap >> ${log_dir}/5layer_cnn-relu_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-relu_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-relu_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg4_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-relu_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act relu_rg6_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-relu_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # Swish approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg4 --bn >> ${log_dir}/5layer_cnn-swish_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg4 --bn >> ${log_dir}/5layer_cnn-swish_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg2 --bn >> ${log_dir}/5layer_cnn-swish_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg2 --bn >> ${log_dir}/5layer_cnn-swish_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg4 --bn --do >> ${log_dir}/5layer_cnn-swish_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg4 --bn --do >> ${log_dir}/5layer_cnn-swish_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg2 --bn --do >> ${log_dir}/5layer_cnn-swish_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg2 --bn --do >> ${log_dir}/5layer_cnn-swish_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg4 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg4 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg2 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg2 --bn --gap >> ${log_dir}/5layer_cnn-swish_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-swish_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-swish_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg4_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-swish_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act swish_rg6_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-swish_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"

  #########################################################
  # Mish approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg4 --bn >> ${log_dir}/5layer_cnn-mish_rg4_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg4 --bn >> ${log_dir}/5layer_cnn-mish_rg6_deg4-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg2 --bn >> ${log_dir}/5layer_cnn-mish_rg4_deg2-BN-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg2 --bn >> ${log_dir}/5layer_cnn-mish_rg6_deg2-BN-${HH}${MM}.txt)"

  # Dropout
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg4 --bn --do >> ${log_dir}/5layer_cnn-mish_rg4_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg4 --bn --do >> ${log_dir}/5layer_cnn-mish_rg6_deg4-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg2 --bn --do >> ${log_dir}/5layer_cnn-mish_rg4_deg2-BN-DO-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg2 --bn --do >> ${log_dir}/5layer_cnn-mish_rg6_deg2-BN-DO-${HH}${MM}.txt)"

  # GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg4 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg4_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg4 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg6_deg4-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg2 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg4_deg2-BN-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg2 --bn --gap >> ${log_dir}/5layer_cnn-mish_rg6_deg2-BN-GAP-${HH}${MM}.txt)"

  # Dropout + GlobalAveragePooling
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-mish_rg4_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg4 --bn --do --gap >> ${log_dir}/5layer_cnn-mish_rg6_deg4-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg4_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-mish_rg4_deg2-BN-DO-GAP-${HH}${MM}.txt)"
  # "(cd cifar-10 && ../.venv/bin/python ./5layer_cnn.py --act mish_rg6_deg2 --bn --do --gap >> ${log_dir}/5layer_cnn-mish_rg6_deg2-BN-DO-GAP-${HH}${MM}.txt)"
)

[[ -d "${mnist_log_dir}" ]] || mkdir -p "${mnist_log_dir}"
[[ -d "${cifar_log_dir}" ]] || mkdir -p "${cifar_log_dir}"

# for command in "${cifar_train_commands[@]}"
# do
#   eval "${command}"
# done

for command in "${mnist_train_commands[@]}"
do
  eval "${command}"
done
