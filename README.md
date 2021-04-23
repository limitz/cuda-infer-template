# CUDA Inference Template
A clean starter project for running image classification through CNN inference on CUDA, using [TensorRT](https://developer.nvidia.com/tensorrt).


## Usage
Compile for your platform after cloning this repo:
``` sh
git clone https://github.com/limitz/cuda-infer-template/
cd cuda-infer-template
make
```

Generate intermediary `ONNX` network, and derive the `TensorRT` engine from that:
``` sh
sudo chmod +x ./models/generate.py
cd models && ./generate.py; cd -
```

Run the end result to see the classification in action:
``` sh
./program
```


## Installation and Requirements
### Nvidia Jetson

TensorRT is included with the [latest JetPack releases](https://developer.nvidia.com/embedded/jetpack).

In order to create the model it might be needed to create a BIG swapfile. (Building the engine seems to require at least 12 GB of RAM, adjust your swapfile size according to the amount of RAM your jetson has)
``` sh
sudo fallocate -l 12G /mnt/swapfile
sudo chmod 600 /mnt/swapfile
sudo mkswap /mnt/swapfile
sudo swapon /mnt/swapfile
```

To make the swapfile persistent across reboots add the following line to `/etc/fstab`
```
/mnt/swapfile swap swap defaults 0 0
```
__Invalid output on Jetson after deserialization of the engine__

There is an issue with the prebuilt tensorrt package shipped in JetPack it seems. The solution is to build TensorRT yourself from tags/7.1.3 like so:
```
cd TensorRT
rm -rf build
rm -rf lib
rm -rf bin
git checkout tags/7.1.3
git reset --hard
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCUDA_VERSION=10.2 -DGPU_ARCHS="72"
make -j2
make install
sudo cp -P ../lib/libnv* /usr/lib/aarch64-linux-gnu/
```
Then make sure to remove any serialized engine in the models directory (ssd.engine) and rebuild the application
```
cd cuda-infer-template
rm models/ssd.engine
cd examples/basic
make clean
make
```

__Nvidia Jetson and NVJPEG__

nvjpeg is not supported (yet) on the Jetson platform, partly because of it's Ubuntu 18.04 limit and GPU limits tied to that. See [this forum post](https://forums.developer.nvidia.com/t/installing-cuda-11-x-on-jetson-nano/169109/3) for more information on this topic. For jetson the libjpeg library is used, which can be substituted by jpeg-turbo if more performance is needed.

### Ubuntu
Install TensorRT through [Nvidia's instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
Make sure to export `CUDA_BIN_PATH` for easy access to `nvcc` and other tools.

### Arch
TensorRT is only compatible with dependency CUDA `11.0` and `11.1`. The default CUDA version in Pacman is `11.2`. Make sure this version is not installed to prevent conflicts and install the previous version through AUR:

```sh
pamac install cuda11.1
```
Export bin path to CUDA tools (preferably from `.profile` or similar):
```sh
export CUDA_BIN_PATH=/opt/cuda/bin
```

Download the correct `tar.gz` archive from [the TensorRT download page](https://developer.nvidia.com/nvidia-tensorrt-download). You will need an Nvidia Developers account.
At the time of writing you need the `7.2.3.4` version with CUDA `11.1`. 
The exact filename is mentioned in the `TensorRT` AUR package's `PKGBUILD` file.

Create build dir for TensorRT AUR package:
```sh
cd /tmp
git clone https://aur.archlinux.org/tensorrt.git
cd tensorrt
```
Now move the downloaded TensorRT `.tar.gz` into this folder:
```sh
mv ~/my-downloads/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz .
```
Compile the AUR package:
``` sh
makepkg -si
```
