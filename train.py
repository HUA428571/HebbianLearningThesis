import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn
from torchvision import transforms

import params as P
import utils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from evaluate import launch_experiment, load_models, eval_pass, eval_batch


# 定义一个模型的训练过程函数，该函数会遍历数据集进行训练并计算训练误差。
def train_pass(net, dataset, config, pre_net=None, criterion=None, optimizer=None):
	net.train()  # 将模型设置为训练模式

	# 用于追踪训练进度的变量
	total = 0  # 到目前为止处理的样本总数
	acc = None  # 训练准确率

	# 遍历数据集中的每个批次
	for batch in dataset:
		# 处理批次数据，计算命中数、批次总数和损失
		batch_hits, batch_count, loss = eval_batch(net, batch, config, pre_net, criterion)

		# 更新统计数据
		total += batch_count
		batch_acc = batch_hits / batch_count  # 计算批次准确率
		if acc is None:
			acc = batch_acc
		else:
			acc = 0.1 * batch_acc + 0.9 * acc  # 使用指数加权平均计算总体准确率

		if optimizer is not None:
			optimizer.zero_grad()  # 清空梯度
			loss.backward()  # 反向传播计算梯度
			optimizer.step()  # 更新模型权重

		# 每处理5000个样本大约输出一次训练进度（或者是最后一个批次）
		if total % 5000 < config.BATCH_SIZE or total == config.VAL_SET_SPLIT:
			print("Epoch progress: " + str(total) + "/" + str(config.VAL_SET_SPLIT) + " processed samples")

	return acc  # 返回训练准确率


# 执行一次网络的训练迭代
def run_train_iter(config, iter_id):
	if config.CONFIG_FAMILY == P.CONFIG_FAMILY_HEBB: torch.set_grad_enabled(False)  # 如果配置指定，则禁用梯度计算

	torch.manual_seed(iter_id)  # 设置随机数生成器种子
	torch.backends.cudnn.deterministic = True  # 设置确定性的卷积算法
	torch.backends.cudnn.benchmark = False  # 禁用卷积性能基准测试，保证结果可复现

	# 加载数据集
	print("Preparing training dataset...")
	# 定义转换
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 使用CIFAR10数据集的均值和标准差
	train_set = CIFAR10(root='F:\DataSet\CIFAR10', train=True, download=False, transform=transform)
	train_set = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
	print("Training dataset ready!")
	print("Preparing validation dataset...")
	val_set = CIFAR10(root='F:\DataSet\CIFAR10', train=False, download=False, transform=transform)
	val_set = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
	print("Validation dataset ready!")

	# 加载数据集
	# print("Preparing dataset manager...")
	# dataManager = data.DataManager(config)  # 初始化数据集管理器
	# print("Dataset manager ready!")
	# print("Preparing training dataset...")
	# train_set = dataManager.get_train()  # 加载训练集
	# print("Training dataset ready!")
	# print("Preparing validation dataset...")
	# val_set = dataManager.get_val()  # 加载验证集
	# print("Validation dataset ready!")

	# 准备训练模型
	print("Preparing network...")
	pre_net, net = load_models(config, iter_id, testing=False)  # 加载模型
	criterion = None  # 损失函数
	optimizer = None  # 优化器
	scheduler = None  # 学习率调度器
	if config.CONFIG_FAMILY == P.CONFIG_FAMILY_GDES:
		criterion = nn.CrossEntropyLoss()  # 实例化交叉熵损失函数
		optimizer = optim.SGD(net.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.L2_PENALTY,
							  nesterov=True)  # 实例化SGD优化器
		scheduler = sched.MultiStepLR(optimizer, gamma=config.LR_DECAY, milestones=config.MILESTONES)  # 实例化学习率调度器
	print("Network ready!")

	# 开始训练模型
	print("Starting training...")
	train_acc_data = []
	val_acc_data = []
	best_acc = 0.0
	best_epoch = 0
	start_time = time.time()  # 记录开始时间
	for epoch in range(1, config.NUM_EPOCHS + 1):
		# 每个周期开始时输出训练进度
		utils.print_train_progress(epoch, config.NUM_EPOCHS, time.time() - start_time, best_acc, best_epoch)

		print("Training...")
		train_acc = train_pass(net, train_set, config, pre_net, criterion, optimizer)  # 执行训练
		print("Training accuracy: {:.2f}%".format(100 * train_acc))

		print("Validating...")
		val_acc = eval_pass(net, val_set, config, pre_net)  # 执行验证
		print("Validation accuracy: {:.2f}%".format(100 * val_acc))

		# 更新训练统计和保存图表
		train_acc_data += [train_acc]
		val_acc_data += [val_acc]
		utils.save_figure(train_acc_data, val_acc_data, config.ACC_PLT_PATH[iter_id])

		# 如果验证准确率提高，更新最佳模型
		if val_acc > best_acc:
			print("Top accuracy improved! Saving new best model...")
			best_acc = val_acc
			best_epoch = epoch
			utils.save_dict(net.state_dict(), config.MDL_PATH[iter_id])
			if hasattr(net, 'conv1') and net.input_shape == P.INPUT_SHAPE: utils.plot_grid(net.conv1.weight, config.KNL_PLT_PATH[iter_id])
			if hasattr(net, 'fc') and net.input_shape == P.INPUT_SHAPE: utils.plot_grid(net.fc.weight.view(-1, *P.INPUT_SHAPE),
																						config.KNL_PLT_PATH[iter_id])
			print("Model saved!")

		if scheduler is not None: scheduler.step()  # 更新学习率调度器


if __name__ == '__main__':
	launch_experiment(run_train_iter)  # 启动实验
