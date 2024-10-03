# UNet

This repository contains an implementation of the U-Net architecture.

## Setup

To set up the development environment and configure the `sys.path`:

1. Run the `init_dev_env.sh` script:

   ```
   $ source init_dev_env.sh
   ```

   This script will create an IPython startup file that adds the project directory to `sys.path`, allowing you to import modules from this project in Jupyter Notebook.


   ## Install PyTorch with CUDA support

   The `init_dev_env.sh` script includes a command to install PyTorch with CUDA 11.7 support:

   ```
   $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

   This command installs PyTorch with CUDA 11.7 support, which is optimized for GPU acceleration.

   Note: If your CUDA version is different from 11.7, you should modify this command in the
   `init_dev_env.sh` script to match your CUDA version. You can find the appropriate command for
   your CUDA version on the official PyTorch website: https://pytorch.org/get-started/locally/


2. Restart Jupyter Notebook for the changes to take effect.

## Usage

For usage examples and demonstrations of the UNet implementation, please refer to the `unet-example.ipynb` notebook in this repository.



## Acknowledgements

The UNet implementation used in this project is based on the code shared in the following article:

[DACON - U-Net 구현하기](https://dacon.io/codeshare/4245)

We appreciate the author's contribution to the community by sharing their implementation.

