# Simplicial-neural-network-benchmark
Simplicial neural network benchmarking software as part of my final year dissertation project
Paper on SAT can be found here: https://arxiv.org/abs/2204.09455

To create the correct environments, run the following commands
```
conda create -n myenv python=3.8
conda activate myenv
./requirements.sh
```

This software features the implementation of 6 models:
- Graph convolutional network (GCN) by Kipf and Welling
- Graph attention network (GAT) by Veličković et al.
- Simplicial neural network (SCN) by Ebli, Defferrard, and Spreemann
- Simplicial 2-complex convolutional neural network (SCConv) by Bunch et al
- Simplicial attention network (SAT) by Goh and Bodnar
- Simplicial attention neural network (SAN) by Giusti et al.

This software features four benchmarking tests
- Superpixel graph classification of MNIST and CIFAR10 images (superpixel_benchmark.py)
- Trajectory classification (orientation_flow_benchmark.py)
- Adversarial resistance (adversarial_superpixel_benchmark.py)
- Unsupervised representation learning (planetoid_dgi_benchmark.py)
