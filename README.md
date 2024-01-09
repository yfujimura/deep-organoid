# Machine learning-based estimation of spatial gene expression pattern during ESC-derived retinal organoid development

This is the code of "[Machine learning-based estimation of spatial gene expression pattern during ESC-derived retinal organoid development](https://link.springer.com/article/10.1038/s41598-023-49758-y?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20231220&utm_content=10.1038/s41598-023-49758-y)," Scientific Reports, 2023.
<br>

```
@article{fujimura23,
	author = {Fujimura, Yuki and Sakai, Itsuki and Shioka, Itsuki and Takata, Nozomu and Hashimoto, Atsushi and Funatomi, Takuya and Okuda, Satoru},
	title = {Machine learning-based estimation of spatial gene expression pattern during ESC-derived retinal organoid development},
	journal = {Scientific Reports},
	volume={13},
	number={1},
	pages={22781},
	year = 2023
}
```

## Requirements
Our experimental environment is
```
Ubuntu 20.04
CUDA 11.3.1
pytorch 1.11.0
```
The requirements can be installed by
```
conda install --name organoid --file requirements.yml
conda activate organoid
```

## Pre-trained models
We provide our pre-trained model, which can be downloaded as
```
wget 'https://onedrive.live.com/download?cid=22AA8A9F0CDA7E59&resid=22AA8A9F0CDA7E59%2124986&authkey=ALoL8L1euqJQNJs' -O ckpts.tar.gz
tar -zxvf ckpts.tar.gz
```

## Sample result
We provide a sample code for the results in Fig. 2 in the paper.
```
python sample.py ckpts resnet
python fig2.py
```

![fig2](figures/fig2.png)
