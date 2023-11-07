Code for paper of CIKM 2023: Dually Enhanced Delayed Feedback Modeling for Streaming Conversion Rate Prediction [[PDF](https://dl.acm.org/doi/10.1145/3583780.3614856)]

# Quick Start

1. Please run the shell file ```run_pretrain.sh``` to get the pretrain model.

2. Please run the shell file ```run_stream.sh``` to evaluate our method DDFM in the streaming protocol.

# Environment

Our experimental environment is shown below:

```
numpy version: 1.19.2
pandas version: 1.1.5
scikit-learn version: 0.24.2
torch version: 1.7.0+cu110
torchvision version: 0.8.1+cu110
```
# Reference

Our experiments follow the previous studies: [[ES-DFM](https://github.com/ThyrixYang/es_dfm)], [[DEFER](https://github.com/gusuperstar/defer)], [[DEFUSE](https://github.com/ychen216/DEFUSE)]

# Citation
If you find our code or work useful for your research, please cite our work.

```
@inproceedings{dai2023dually,
  title={Dually Enhanced Delayed Feedback Modeling for Streaming Conversion Rate Prediction},
  author={Dai, Sunhao and Zhou, Yuqi and Xu, Jun and Wen, Ji-Rong},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={390--399},
  year={2023}
}
```


