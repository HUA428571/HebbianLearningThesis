import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


# 对输入应用展开操作，以便将其准备好以便对滑动内核进行处理，该滑动内核的形状作为参数传递。
def unfold_map2d(input, kernel_height, kernel_width):
	# 在对输入和滑动内核进行操作之前，我们需要展开输入，即提取出将要应用内核的窗口并将其分开。为此，将内核形状作为操作的参数。通过展开操作将单个提取的窗口重塑为秩为1的向量。F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2)的输出是一个张量，其结构如下：第一维是批次维度；第二维是滑动维度，即每个元素都是在不同偏移处提取的窗口（并重塑为秩为1的向量）；第三维是该向量内的标量。
	inp_unf = F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2)
	# 现在我们需要将张量重塑为我们想要的实际输出形状，该形状如下：第一维是批次维度，第二维是输出通道维度，第三和第四维是高度和宽度维度（通过将前三维，即表示输入图中线性偏移的滑动维度，分割为两个新的维度，分别表示高度和宽度），第五维是窗口组件维度，对应于从输入中提取的窗口的元素（重塑为秩为1的向量）。然后返回结果张量。
	inp_unf = inp_unf.view(input.size(0),  # 批次维度
						   1,  # 输出通道维度
						   input.size(2) - kernel_height + 1,  # 高度维度
						   input.size(3) - kernel_width + 1,  # 宽度维度
						   -1  # 过滤器/窗口维度
						   )
	return inp_unf


# 自定义向量函数，表示输入与滑动内核的和，就像卷积是通过滑动内核的乘法（类比为卷积为kernel_mult2d）
def kernel_sum2d(input, kernel):
	# 为了与滑动内核进行求和，我们首先需要展开输入。结果张量将具有以下结构：第一维是批次维度，第二维是输出通道维度，第三和第四维是高度和宽度维度，第五维是过滤器/窗口组件维度，对应于从输入中提取的窗口的元素以及与过滤器的元素等效（重塑为秩为1的向量）
	inp_unf = unfold_map2d(input, kernel.size(2), kernel.size(3))
	# 此时，这两个张量可以相加。通过在批次维度和高度和宽度维度上增加单例维度来重塑内核。通过利用广播，发生的是inp_unf张量在输出通道维度上进行广播（因为其在此维度上的形状为1），因此它会自动处理内核的不同过滤器。同样，内核在第一维度（因此自动处理批次维度上的不同输入）以及第三和第四维度（因此自动处理在不同高度和宽度偏移处从图像中提取的不同窗口）上进行广播。
	out = inp_unf + kernel.view(1, kernel.size(0), 1, 1, -1)
	return out


# 测试kernel_sum2d函数的实现
def test_kernelsum():
	x = torch.randn(8,  # 批次维度
					3,  # 输入通道维度
					10,  # 高度维度
					12  # 宽度维度
					)
	w = torch.randn(6,  # 输出通道维度
					3,  # 输入通道维度
					4,  # 高度维度
					5  # 宽度维度
					)
	output = torch.empty(x.shape[0],  # 批次维度
						 w.shape[0],  # 输出通道维度
						 x.shape[2] - w.shape[2] + 1,  # 高度维度
						 x.shape[3] - w.shape[3] + 1,  # 宽度维度
						 w.shape[1] * w.shape[2] * w.shape[3]  # 过滤器维度
						 )

	# 通过for循环实现与向量化实现进行交叉验证
	for batch in range(0, x.shape[0]):  # 循环批次维度
		for outchn in range(0, w.shape[0]):  # 循环输出通道维度
			for i in range(0, x.shape[2] - w.shape[2] + 1):  # 循环高度维度
				for j in range(0, x.shape[3] - w.shape[3] + 1):  # 循环宽度维度
					output[batch, outchn, i, j, :] = (x[batch, :, i:i + w.shape[2], j:j + w.shape[3]] + w[outchn, :, :, :]).view(-1)

	out = kernel_sum2d(x, w)

	print((output.equal(out)))  # 应打印出True


# 计算输入和滑动内核之间的乘积
def kernel_mult2d(x, w, b=None):
	return F.conv2d(x, w, b)


# 输入在权重向量上的投影
def vector_proj2d(x, w, bias=None):
	# 计算与滑动内核的标量积
	prod = kernel_mult2d(x, w)
	# 除以权重向量的范数以获得投影
	norm_w = torch.norm(w.view(w.size(0), -1), p=2, dim=1).view(1, -1, 1, 1)
	norm_w += (norm_w == 0).float()  # 防止除以零
	if bias is None: return prod / norm_w
	return prod / norm_w + bias.view(1, -1, 1, 1)


# 输入在权重向量上的投影，裁剪在0和+inf之间
def clp_vector_proj2d(x, w, bias=None):
	return vector_proj2d(x, w, bias).clamp(0)


# Sigmoid相似性
def sig_sim2d(x, w, bias=None):
	proj = vector_proj2d(x, w, bias)
	# return torch.sigmoid((proj - proj.mean())/proj.std())
	return torch.sigmoid(proj)


# 输入图和滑动内核之间的余弦相似性
def cos_sim2d(x, w, bias=None):
	proj = vector_proj2d(x, w)
	# 除以输入的范数以获得余弦相似性
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	norm_x = torch.norm(x_unf, p=2, dim=4)
	norm_x += (norm_x == 0).float()  # 防止除以零
	if bias is None: return proj / norm_x
	return (proj / norm_x + bias.view(1, -1, 1, 1)).clamp(-1, 1)


# 余弦相似性裁剪在0和1之间
def clp_cos_sim2d(x, w, bias=None):
	return cos_sim2d(x, w, bias).clamp(0)


# 余弦相似性重新映射到0，1
def raised_cos2d(x, w, bias=None):
	return (cos_sim2d(x, w, bias) + 1) / 2


# 返回计算提升余弦幂p的函数
def raised_cos2d_pow(p=2):
	def raised_cos2d_pow_p(x, w, bias=None):
		if bias is None: return raised_cos2d(x, w).pow(p)
		return (raised_cos2d(x, w).pow(p) + bias.view(1, -1, 1, 1)).clamp(0, 1)

	return raised_cos2d_pow_p


# Softmax在权重向量投影激活函数上
def proj_smax2d(x, w, bias=None):
	# 计算投影的指数
	e_pow_y = torch.exp(vector_proj2d(x, w, bias))
	# 返回softmax函数的结果，即每个元素的指数除以所有元素指数的和
	return e_pow_y / e_pow_y.sum(1, keepdims=True)


# 高斯激活函数的响应
def gauss(x, w, sigma=None):
	# 计算输入x和权重w的差的范数
	d = torch.norm(kernel_sum2d(x, -w), p=2, dim=4)
	# 如果没有给出sigma，使用启发式方法计算方差，即使用维度的数量
	if sigma is None: return torch.exp(-d.pow(2) / (2 * utils.shape2size(tuple(w[0].size()))))
	# 如果给出了sigma，使用高斯函数计算结果
	return torch.exp(-d.pow(2) / (2 * (sigma.view(1, -1, 1, 1).pow(2))))


# 返回一个用于指数衰减学习率调度的lambda函数
def sched_exp(tau=1000, eta_min=0.01):
	# 计算衰减因子gamma
	gamma = torch.exp(torch.tensor(-1. / tau)).item()
	# 返回一个lambda函数，该函数将学习率乘以衰减因子，并将结果限制在eta_min以上
	return lambda eta: (eta * gamma).clamp(eta_min)


# 这个模块代表了一层使用Hebbian-WTA规则训练的卷积神经元
class HebbianMap2d(nn.Module):
	# 学习规则的类型
	RULE_BASE = 'base'  # delta_w = eta * lfb * (x - w)
	RULE_HEBB = 'hebb'  # delta_w = eta * y * lfb * (x - w)

	# LFB核的类型
	LFB_GAUSS = 'gauss'
	LFB_DoG = 'DoG'
	LFB_EXP = 'exp'
	LFB_DoE = 'DoE'

	def __init__(self, in_channels, out_size, kernel_size, competitive=True, random_abstention=False, lfb_value=0,
				 similarity=raised_cos2d_pow(2), out=vector_proj2d, weight_upd_rule=RULE_BASE, eta=0.1, lr_schedule=None, tau=1000):
		super(HebbianMap2d, self).__init__()

		# 初始化权重
		out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
		self.out_size = torch.tensor(out_size_list[0:min(len(out_size_list), 3)])
		out_channels = self.out_size.prod().item()
		if hasattr(kernel_size, '__len__') and len(kernel_size) == 1: kernel_size = kernel_size[0]
		if not hasattr(kernel_size, '__len__'): kernel_size = [kernel_size, kernel_size]
		stdv = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
		self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
		nn.init.uniform_(self.weight, -stdv, stdv)  # 使用默认的pytorch卷积模块的初始化方式（来自论文"Efficient Backprop, LeCun"）

		# 启用/禁用随机弃权，竞争学习，侧向反馈等特性
		self.competitive = competitive
		self.random_abstention = competitive and random_abstention
		self.lfb_on = competitive and isinstance(lfb_value, str)
		self.lfb_value = lfb_value

		# 设置输出函数，相似性函数和学习规则
		self.similarity = similarity
		self.out = out
		self.teacher_signal = None  # 用于监督训练的教师信号
		self.weight_upd_rule = weight_upd_rule

		# 初始学习率和学习率调度策略。学习率包装在一个注册的缓冲区中，以便我们可以保存/加载它
		self.register_buffer('eta', torch.tensor(eta))
		self.lr_schedule = lr_schedule  # 学习率调度策略

		# 设置与侧向反馈特性相关的参数
		if self.lfb_on:
			# 准备生成将用于应用侧向反馈的核的变量
			map_radius = (self.out_size - 1) // 2
			sigma_lfb = map_radius.max().item()
			x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
			for i in range(1, self.out_size.size(0)):
				x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
				for j in range(i): x_new = x_new.unsqueeze(j)
				x = torch.max(x.unsqueeze(-1), x_new)  # max给出L_infinity距离，sum会给出L_1距离，root_p(sum x^p)用于L_p
			# 在一个注册的缓冲区中存储将用于应用侧向反馈的核
			if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE:
				self.register_buffer('lfb_kernel', torch.exp(-x.float() / sigma_lfb))
			else:
				self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (sigma_lfb ** 2))))
			# 在应用lfb核之前将对输入进行填充的填充
			pad_pre = map_radius.unsqueeze(1)
			pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
			self.pad = tuple(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
			# LFB核收缩参数
			self.alpha = torch.exp(torch.log(torch.tensor(sigma_lfb).float()) / tau).item()
			if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG: self.alpha = self.alpha ** 2
		else:
			self.register_buffer('lfb_kernel', None)

		# 初始化用于统计收集的变量
		if self.random_abstention:
			self.register_buffer('victories_count', torch.zeros(out_channels))
		else:
			self.register_buffer('victories_count', None)

	def set_teacher_signal(self, y):
		self.teacher_signal = y

	def forward(self, x):
		y = self.out(x, self.weight)
		if self.training: self.update(x)
		return y

	def update(self, x):
		# 准备输入
		y = self.similarity(x, self.weight)
		t = self.teacher_signal
		if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(y, device=y.device)
		y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
		x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(y.size(0), 1, -1)

		# 随机弃权
		if self.random_abstention:
			abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
			scores = y * (torch.rand_like(abst_prob, device=y.device) >= abst_prob).float().unsqueeze(0)
		else:
			scores = y

		# 竞争。返回的winner_mask是一个位图，告诉我们哪个神经元赢了，哪个神经元输了。
		if self.competitive:
			if t is not None: scores *= t
			winner_mask = (scores == scores.max(1, keepdim=True)[0]).float()
			if self.random_abstention:  # 如果使用随机弃权，更新统计
				winner_mask_sum = winner_mask.sum(0)  # 一个神经元赢得的输入数量
				self.victories_count += winner_mask_sum
				self.victories_count -= self.victories_count.min().item()
		else:
			winner_mask = torch.ones_like(y, device=y.device)

		# 侧向反馈
		if self.lfb_on:
			lfb_kernel = self.lfb_kernel
			if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE: lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(
				0.5)  # 高斯/指数的差（墨西哥帽形状的函数）
			lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
			if self.out_size.size(0) == 1:
				lfb_out = torch.conv1d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			elif self.out_size.size(0) == 2:
				lfb_out = torch.conv2d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			else:
				lfb_out = torch.conv3d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			lfb_out = lfb_out.clamp(-1, 1).view_as(y)
		else:
			lfb_out = winner_mask
			if self.competitive:
				lfb_out[lfb_out == 0] = self.lfb_value
			elif t is not None:
				lfb_out = t

		# 计算步长调制系数
		r = lfb_out  # RULE_BASE，基本规则
		if self.weight_upd_rule == self.RULE_HEBB: r *= y  # 如果使用的是Hebbian规则，则r需要乘以y

		# 计算权重更新的增量
		r_abs = r.abs()  # 计算r的绝对值
		r_sign = r.sign()  # 计算r的符号
		# 计算权重更新的增量，这里使用了广播机制，r_abs和r_sign会自动扩展到和x_unf的维度一致
		delta_w = r_abs.unsqueeze(2) * (r_sign.unsqueeze(2) * x_unf - self.weight.view(1, self.weight.size(0), -1))

		# 由于我们使用的是批量输入，我们需要将每个内核的不同更新步骤聚合到一个唯一的更新中。
		# 我们通过取r系数（决定每个步骤长度的权重）的加权平均来实现这一点
		r_sum = r_abs.sum(0)  # 计算r_abs的和
		r_sum += (r_sum == 0).float()  # 防止除以零
		# 计算加权平均的权重更新增量
		delta_w_avg = (delta_w * r_abs.unsqueeze(2)).sum(0) / r_sum.unsqueeze(1)

		# 应用权重更新增量
		self.weight += self.eta * delta_w_avg.view_as(self.weight)

		# 如果启用了侧向反馈（LFB），则进行LFB核的收缩
		if self.lfb_on: self.lfb_kernel = self.lfb_kernel.pow(self.alpha)
		# 如果设置了学习率调度策略，则更新学习率
		if self.lr_schedule is not None: self.eta = self.lr_schedule(self.eta)


# 生成一批用于测试的随机输入
def gen_batch(centers, batch_size, win_height, win_width):
	# 首先在集群中心周围生成随机扰动的补丁，然后
	# 在水平和垂直维度上连接它们。重复以生成一批数据。
	batch = torch.empty(0)
	for j in range(batch_size):  # 循环生成批次
		image = torch.empty(0)
		for k in range(win_height):  # 循环垂直连接图像行
			row = torch.empty(0)
			for l in range(win_width):  # 循环水平连接补丁
				# 通过扰动集群中心生成输入补丁
				index = int(torch.floor(torch.rand(1) * centers.size(0)).item())
				patch = centers[index] + 0.1 * torch.randn_like(centers[index])
				# 将补丁水平连接到图像行
				row = torch.cat((row, patch), 2)
			# 将行垂直连接到图像
			image = torch.cat((image, row), 1)
		# 将图像连接到批次
		batch = torch.cat((batch, image.unsqueeze(0)), 0)
	return batch


# 测试批量生成函数
def test_genbatch():
	# 生成围绕其构建集群的中心
	centers = torch.randn(6, 3, 4, 5)
	# 生成围绕中心的输入批次
	batch = gen_batch(centers, 10, 2, 2)
	# 检查批次大小是否正确（以确保）
	print(batch.size())  # 应打印10x3x8x10


# 测试HebbianMap2d的实现
def test_hebbianmap():
	# 用于打印摘要信息的函数
	def print_results(model, centers):
		print('\n' + '#' * 79 + '\n')
		responses = model(centers).squeeze()
		top_act, closest_neurons = responses.max(1)
		# 打印每个中心最接近的神经元及其输出
		for i in range(responses.size(0)): print(
			"中心 " + str(i) + " 最接近的神经元: " + str(closest_neurons[i].item()) + ", 输出: " + str(top_act[i].item()))
		print()
		top_act, closest_centers = responses.max(0)
		# 打印每个神经元最接近的中心及其输出
		for i in range(responses.size(1)): print(
			"神经元 " + str(i) + " 最接近的中心: " + str(closest_centers[i].item()) + ", 输出: " + str(top_act[i].item()))
		print('\n' + '#' * 79 + '\n')

	# 设置随机种子
	torch.random.manual_seed(3)
	# 定义核形状，中心数量，迭代次数，批次大小，窗口高度和宽度
	kernel_shape = (6, 3, 4, 5)
	num_centers = 6
	num_iter = 2000
	batch_size = 10
	win_height = 2
	win_width = 2
	# 定义设备，如果有cuda则使用，否则使用cpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# 初始化模型
	model = HebbianMap2d(in_channels=kernel_shape[1], out_size=kernel_shape[0], kernel_size=[kernel_shape[2], kernel_shape[3]],
						 competitive=True, random_abstention=False, lfb_value=0, similarity=raised_cos2d_pow(2), out=cos_sim2d,
						 weight_upd_rule=HebbianMap2d.RULE_BASE, eta=0.1, lr_schedule=sched_exp(1000, 0.01), tau=1000)
	model.eval()
	model.to(device)

	# 生成围绕其构建集群的中心
	centers = torch.randn(num_centers, *kernel_shape[1:4])
	# 检查中心与随机初始化的权重向量之间的距离
	print_results(model, centers)

	# 训练模型：生成一批输入并将其提供给模型，重复所需的迭代次数
	model.train()
	for i in range(num_iter):
		batch = gen_batch(centers, batch_size, win_height, win_width)
		batch = batch.to(device)
		model(batch)
	model.eval()

	# 验证模型的权重向量是否已收敛到集群中心
	print_results(model, centers)


if __name__ == '__main__':
	test_kernelsum()
	test_genbatch()
	test_hebbianmap()
