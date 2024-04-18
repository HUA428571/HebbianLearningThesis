使用Hebbian学习算法来训练深度卷积神经网络的Pytorch实现。一个神经网络模型在CIFAR10上被训练，同时使用Hebbian算法和SGD进行结果比较。虽然Hebbian学习是无监督的，我也实现了一种技术，使用Hebbian算法以监督方式训练最终的线性分类层。这是通过在最后一层应用一个教师信号来完成的，该信号提供了期望的输出；然后强制神经元更新它们的权重，以便跟随那个信号。

你可能也想看看新的代码库！  
HebbianPCA: https://github.com/GabrieleLagani/HebbianPCA/blob/master/README.md  
最新更新: https://github.com/GabrieleLagani/HebbianLearning  

要启动一个训练会话，请输入：  
` python <项目根目录>/train.py --config <配置族>/<配置名称>`  
其中`<配置族>`是`gdes`或`hebb`，取决于你想要运行梯度下降还是Hebbian训练，  
`<配置名称>`是`config.py`文件中的一个训练配置的名称。  
示例：  
` python <项目根目录>/train.py --config gdes/config_base`  
要在CIFAR10测试集上评估网络，请输入：  
` python <项目根目录>/evaluate.py --config <配置族>/<配置名称>`

有关更多详细信息，请参考我的论文工作：  
_"Hebbian Learning Algorithms for Training Convolutional Neural Networks; G. Lagani"_  
可以在 https://etd.adm.unipi.it/theses/available/etd-03292019-220853/unrestricted/hebbian_learning_algorithms_for_training_convolutional_neural_networks_gabriele_lagani.pdf  
以及相关论文：  
_"Hebbian Learning Meets Deep Convolutional Neural Networks; G. Amato, F. Carrara, F. Falchi, C. Gennaro and G. Lagani"_  
可以在：http://www.nmis.isti.cnr.it/falchi/Draft/2019-ICIAP-HLMSD.pdf  

作者：Gabriele Lagani - gabriele.lagani@gmail.com