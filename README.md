# Efficient Keyword Spotting through long-range interactions with Temporal Lambda Networks

## Abstract
Models based on attention mechanisms have shown unprecedented speech recognition performance. However, they are computationally expensive and unnecessarily complex for keyword spotting, a task targeted to small-footprint devices.

This work explores the application of Lambda networks, an alternative framework for capturing long-range interactions without attention, for the keyword spotting task. We propose a novel ResNet-based model by swapping the residual blocks by temporal Lambda layers. Furthermore, the proposed architecture is built upon uni-dimensional temporal convolutions that further reduce its complexity.

The presented model does not only reach state-of-the-art accuracies on the Google Speech Commands dataset, but it is 85% and 65% lighter than its Transformer-based (KWT) and convolutional (Res15) counterparts while being up to 100x faster. To the best of our knowledge, this is the first attempt to explore the Lambda framework within the speech domain and therefore, we unravel further research of new interfaces based on this architecture. 

## Requirements
Here we list the requirements needed to run the project (and the version we used). It is recomendded to install `pytorch` and `torchaudio` following the official installation instructions
* easydict (1.9)
* einops (0.3.0)
* librosa (0.8.0)
* pyyaml (5.4.1)

## Configurating the model
The model configurations of the 3 tested models for the *Google Speech Commands dataset* are found in `configs/google_commands/`.
* LambdaResnet18
* TC Resnet 14
* Resnet 15

To change the subtask, change the `num_lables` variable in the config `.yml` file. More parameters can be fine tuned just changing the configuration variables.

## Training and evaluating the model

The training of the model can be performed by running the `train.py` file through the following command: (Choose the available GPU on your PC)

```bash
python -u train.py --config_exp configs/google_commands/'desired_config'.yml --gpu X
```

If gpu is not set, the model will be trained in `DataParallel` mode, using all the available GPUs and multipliying its batch size for the number of available GPUs.

To evaluate the trained model, the path to the `pth.tar` saved model must be given:

```bash
python -u eval.py --config_exp configs/google_commands/'desired_config'.yml --gpu X --model output/google_commands/'desired_model'/'model'.pth.tar
```

## Plotting the results
Additional scripts are attached in the `scripts/` folder in order to get plots, and debug custom modules used in the training setup.