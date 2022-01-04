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

  # GlobalAveragePooling


  #########################################################
  # ReLU approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################

  # GlobalAveragePooling

  #########################################################
  # Swish approx. (rg4_deg4, rg6_deg4, rg4_deg2, rg6_deg2)
  #########################################################
)

cifar_train_commands=(
)


[[ -d "${mnist_log_dir}" ]] || mkdir -p "${mnist_log_dir}"
[[ -d "${cifar_log_dir}" ]] || mkdir -p "${cifar_log_dir}"

for command in "${cifar_train_commands[@]}"
do
  eval "${command}"
done

for command in "${mnist_train_commands[@]}"
do
  eval "${command}"
done
