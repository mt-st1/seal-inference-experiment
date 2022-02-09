#/usr/bin/env bash

cd cifar-10/saved_models &&
cp 5layer_cnn-square-BN-GAP-0106-best.pt 5layer_cnn-square-BN-GAP-prune_conv2_0.3-round0.pt &&
cp 5layer_cnn-swish_rg5_deg2-BN-GAP-0106-best.pt 5layer_cnn-swish_rg5_deg2-BN-GAP-prune_conv2_0.2-round0.pt &&
cp 5layer_cnn-swish_rg7_deg4-BN-GAP-0107-best.pt 5layer_cnn-swish_rg7_deg4-BN-GAP-prune_conv2_0.25-round0.pt

# cd mnist/saved_models &&
# cp 3layer_cnn-square-BN-1223-best.pt 3layer_cnn-square-BN-prune_conv2_0.4-round0.pt &&
# cp 3layer_cnn-swish_rg5_deg2-BN-1223-best.pt 3layer_cnn-swish_rg5_deg2-BN-prune_conv2_0.4-round0.pt &&
# cp 3layer_cnn-swish_rg7_deg4-BN-1224-best.pt 3layer_cnn-swish_rg7_deg4-BN-prune_conv2_0.4-round0.pt
