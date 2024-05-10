import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import params as P
import utils


# 计算CIFAR10数据集的均值、标准差和ZCA白化矩阵的函数
def get_dataset_stats(limit):
	DATA_STATS_FILE = P.STATS_FOLDER + '/cifar10_' + str(limit) + '.pt'  # 定义存储统计数据的文件路径
	MEAN_KEY = 'mean'  # 均值关键字
	STD_KEY = 'std'  # 标准差关键字
	ZCA_KEY = 'zca'  # ZCA白化矩阵关键字

	# 尝试从文件加载统计数据
	stats = utils.load_dict(DATA_STATS_FILE)
	if stats is None:  # 如果统计文件不存在，则计算统计数据
		print("Computing statistics on dataset[0:" + str(limit) + "] (this might take a while)")

		# 加载数据集
		cifar10 = CIFAR10(root=P.DATA_FOLDER, train=True, download=True)
		X = cifar10.data[0:limit]  # 获取前limit个样本

		# 将数据归一化到[0, 1]范围
		X = X / 255.
		# 计算均值和标准差，并将数据标准化至均值为0，标准差为1
		mean = X.mean(axis=(0, 1, 2), keepdims=True)
		std = X.std(axis=(0, 1, 2), keepdims=True)
		X = (X - mean) / std
		# 调整数据维度，以适配PyTorch期望的维度顺序
		X = X.transpose(0, 3, 1, 2)
		# 将图片张量从32x32x3调整为向量，长度为3072
		X = X.reshape(limit, -1)
		# 计算ZCA白化矩阵
		cov = np.cov(X, rowvar=False)
		U, S, V = np.linalg.svd(cov)
		SMOOTHING_CONST = 1e-1
		zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + SMOOTHING_CONST)), U.T))

		# 保存计算出的统计数据
		stats = {MEAN_KEY: mean.squeeze().tolist(), STD_KEY: std.squeeze().tolist(), ZCA_KEY: torch.from_numpy(zca).float()}
		utils.save_dict(stats, DATA_STATS_FILE)
		print("Statistics computed and saved")

	return stats[MEAN_KEY], stats[STD_KEY], stats[ZCA_KEY]


class DataManager:
	def __init__(self, config):
		# 数据加载的常数设置
		self.VAL_SET_SPLIT = config.VAL_SET_SPLIT  # 验证集分割点
		self.BATCH_SIZE = config.BATCH_SIZE  # 批量大小

		# 计算数据集统计信息
		mean, std, zca = get_dataset_stats(self.VAL_SET_SPLIT)

		# 定义要应用于数据的转换操作
		T = transforms.Compose([transforms.ToTensor(),  # 将原始CIFAR10数据转换为张量，同时将像素值映射到[0, 1]范围
			transforms.Normalize(mean, std)  # 使用均值和标准差对每个通道进行归一化，使其具有零均值和单位方差
		])
		# 如果配置要求白化数据，添加白化转换
		if config.WHITEN_DATA:
			T = transforms.Compose([T, transforms.LinearTransformation(zca, torch.zeros(zca.size(1)))])

		self.T_train = T
		self.T_test = T

		# 如果配置要求数据增强，则定义增强转换
		if config.AUGMENT_DATA:
			T_augm = transforms.Compose(
				[transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=20 / 360)], p=0.5),
					transforms.RandomHorizontalFlip(), transforms.Pad(8), transforms.RandomCrop(32),
					transforms.RandomApply([transforms.RandomAffine(degrees=10, shear=10)], p=0.3)])
			self.T_train = transforms.Compose([T_augm, self.T_train])  # 应用增强转换

	# 方法定义获取训练、验证和测试集
	def get_train(self):
		# 下载数据集（如果需要），并应用指定的转换
		cifar10 = CIFAR10(root=P.DATA_FOLDER, train=True, download=True, transform=self.T_train)
		sampler = SubsetRandomSampler(range(self.VAL_SET_SPLIT))
		return DataLoader(cifar10, batch_size=self.BATCH_SIZE, sampler=sampler, num_workers=P.NUM_WORKERS)

	def get_val(self):
		if self.VAL_SET_SPLIT >= P.CIFAR10_NUM_TRN_SAMPLES:
			return self.get_test()
		cifar10 = CIFAR10(root=P.DATA_FOLDER, train=True, download=True, transform=self.T_test)
		sampler = SubsetRandomSampler(range(self.VAL_SET_SPLIT, P.CIFAR10_NUM_TRN_SAMPLES))
		return DataLoader(cifar10, batch_size=self.BATCH_SIZE, sampler=sampler, num_workers=P.NUM_WORKERS)

	def get_test(self):
		cifar10 = CIFAR10(root=P.DATA_FOLDER, train=False, download=True, transform=self.T_test)
		return DataLoader(cifar10, batch_size=self.BATCH_SIZE, num_workers=P.NUM_WORKERS)

