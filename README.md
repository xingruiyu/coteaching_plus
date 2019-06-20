# How does Disagreement Help Generalization against Label Corruption? 
ICML'19: How does Disagreement Help Generalization against Label Corruption? (Pytorch implementation).

========

This is the code for the paper:
[How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)  
Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor W. Tsang, Masashi Sugiyama.  

## Run Co-teaching+ on benchmark datasets (MNIST, CIFAR-10, CIFAR-100, and Tiny-Imagenet) with Pytorch >= 0.4.1
```bash
sh script/mnist.sh
sh script/cifar10.sh
sh script/cifar100.sh
sh script/news.sh 
sh script/imagenet_tiny.sh
```

If you find this code useful in your research then please cite  
```bash
@inproceedings{yu2019does,
  title={How does Disagreement Help Generalization against Label Corruption?},
  author={Yu, Xingrui and Han, Bo and Yao, Jiangchao and Niu, Gang and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={International Conference on Machine Learning},
  pages={7164--7173},
  year={2019}
}
```  

