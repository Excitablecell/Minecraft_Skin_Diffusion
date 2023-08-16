# Minecraft Skin Generation using Diffusion

## Dependencies
* Ubuntu 20.04 LTS
* CUDA 11.7
* python>=3.8
* pytorch
* diffusers
* minepi
* matplotlib

(中文翻译：我是在ubuntu 20.04系统下运行这个工程的，你需要安装CUDA11.7和pytorch，以及上面提到的库)

## Install
We recommend using anaconda environment

Or you can just use the requiements.txt

（中文翻译：我建议你使用anaconda，然后根据requiements.txt安装依赖库）

## Usage
1.You need to download and extract model from my [Google drive](https://drive.google.com/file/d/1nv-3oEUSCvrBqdQjcytxNbAF4qwNXDbS/view?usp=sharing)

(中文翻译：你需要从我的Google云或者百度网盘中下载模型，并解压到工程目录中，百度网盘链接: https://pan.baidu.com/s/1PG3jrcqcHyUrTiOsDX6ECA?pwd=zr7s 提取码: zr7s)

2.Just run the huggingface_diffuser_inference.py

(中文翻译：确保模型被正确解压到工程目录下后(即unet文件夹在次级目录下)，运行huggingface_diffuser_inference.py即可生成MC皮肤)

## Contribution
You are welcome contributing to the package by opening a pull-request

We are following: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s2.2-imports)

## License
MIT License
