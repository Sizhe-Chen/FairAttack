# Decription
* The code is the official implementation of [Measuring the Transferability of Linf Attacks by the L2 Norm](https://arxiv.org/abs/2102.10343)
* Authors: [Sizhe Chen](https://sizhechen.top), Qinghua Tao, Zhixing Ye, [Xiaolin Huang](http://www.pami.sjtu.edu.cn/en/xiaolin)
* To run, prepare [ImageNet validation set (2012)](http://www.image-net.org), place in folder 'ILSVRC2012_img_val', and run

```
conda env create -f fairattack.yaml
```

# Validation
* Reproduce results in Section 3.1, e.g., by

```
python attack_const_loss.py VGG16 Adam 0 1000 gpu_id
```
* Reproduce results in Section 3.2, e.g., by

```
python attack_rmse_loss.py VGG16 Adam 0 1000 gpu_id
```
* Reproduce results in Section 3.3, e.g., by

```
python attack_const_rmse.py VGG16 Adam 0 1000 gpu_id
```

# Test
* Prepare [adv_inception_v3.ckpt](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz), [ens_adv_inception_resnet_v2.ckpt](http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz), [X101-DenoiseAll.npz](https://github.com/facebookresearch/ImageNet-Adversarial-Training/releases/download/v0.2/X101-DenoiseAll.npz)
* Test by

```
python test.py auto gpu_id
```
* or e.g.,

```
python test.py 2021-07-24-12-52-47_VGG16_DI-TI-MI-PGD_SIGN_Iter10_Eps16_Index0to1000_GPU0 gpu_id
```
