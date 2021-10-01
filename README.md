
<b>How to run:</b>

```shell
# cd to the your workspace.
# Specify the directory where dataset is located using the data_path flag.
# Note: User can split samples from training set into the eval set by changing train_size and validation_size.

# For example, to train the Wide-ResNet-28-10 model on a GPU.
python train_cifar.py --model_name=wrn 
```