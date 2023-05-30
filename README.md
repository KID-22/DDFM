Code for CIKM 2023 Submission: Dually Enhanced Delayed Feedback Modeling for Streaming Conversion Rate Prediction

## Quick Start

1. Please run the shell file ```run_pretrain.sh``` to get pretrain model.

2. Please run the shell file ```run_stream.sh``` to test our method DDFM in stream training setting.

## Environment

Our experimental environment is shown below:

```
numpy version: 1.19.2
pandas version: 1.1.5
scikit-learn version: 0.24.2
torch version: 1.7.0+cu110
torchvision version: 0.8.1+cu110
```
## Reference

Our experiments follow the previous studies, which are shown below:

```
@inproceedings{yang2021capturing,
  title={Capturing delayed feedback in conversion rate prediction via elapsed-time sampling},
  author={Yang, Jia-Qi and Li, Xiang and Han, Shuguang and Zhuang, Tao and Zhan, De-Chuan and Zeng, Xiaoyi and Tong, Bin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4582--4589},
  year={2021}
}
```
```
@inproceedings{chen2022asymptotically,
  title={Asymptotically unbiased estimation for delayed feedback modeling via label correction},
  author={Chen, Yu and Jin, Jiaqi and Zhao, Hui and Wang, Pengjie and Liu, Guojun and Xu, Jian and Zheng, Bo},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={369--379},
  year={2022}
}
