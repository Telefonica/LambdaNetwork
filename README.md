# Efficient Keyword Spotting through long-range interactions with Temporal Lambda Networks

## Abstract
Recent models based on attention mechanisms have shown unprecedented performance in the speech recognition domain. These are computational expensive and unnecessarily complex for the keyword spotting task where its main usage is in small-footprint devices. This work explores the application of the Lambda networks, a framework for capturing long-range interactions, within this spotting task. The proposed architecture is inspired by current state-of-the-art models for keyword spotting built on residual connections. Our main contribution consists on swapping the residual blocks by temporal Lambda layers thus bypassing the expensive computation of attention maps, largely reducing the model complexity. Furthermore, the proposed Lambda network is built upon uni-dimensional convolutions which also dramatically decreases the number of floating point operations performed along the inference stage. This architecture does not only reach state-of-the-art accuracies on the Google Speech Commands dataset, but it is 85% and 65% lighter than its multi headed attention (MHAtt-RNN) and residual convolutional (Res15) counterparts, while being up to 100x faster than them. To the best of our knowledge, this is the first attempt to examine the Lambda framework within the speech domain and therefore, we unravel further research and development of future speech interfaces based on this architecture.

## Requirements
* torch
* torchscan
* torchaudio

## Configurating the model
The model configurations of the 3 tested models for the *Google Speech Commands dataset* are found in `configs/google_commands/`.
* LambdaResnet18
* TC Resnet 14
* Resnet 15

To change the subtask, change the `num_lables` variable in the config `.yml` file. More parameters can be fine tuned just changing the configuration variables.

## Training and evaluating the model

the training of the model can be performed by running the `train.py` file through the following command: (Choose the available GPU on your PC)

```bash
python -u train.py --config_env configs/env.yml --config_exp configs/google_commands/'desired_config'.yml --gpu X
```

If gpu is not set, the model will be trained in `DataParallel` mode, using all the available GPUs and multipliying its batch size for the number of available GPUs.

**Note**: The first time running the training script, the *Google Speech Commands dataset* will be downloaded to the `datasets/` folder. Once downloaded, the run may fail as the dataset was not loaded in memory. Run again the training script and everything will be fine.

To evaluate the trained model, the path to the `pth.tar` saved model must be given:

```bash
python -u eval.py --config_env configs/env.yml --config_exp configs/google_commands/'desired_config'.yml --gpu X --model output/google_commands/'desired_model'/'model'.pth.tar
```

## Plotting the results
Additional scripts are attached in the `scripts/` folder in order to get plots, and debug custom modules used in the training setup.