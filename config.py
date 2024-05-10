import params as P
import basemodel.model,	basemodel.top1, basemodel.top2, basemodel.top3, basemodel.top4, basemodel.fc
import hebbmodel.model


class Configuration:
	def __init__(self,
				 config_family,
				 config_name,
				 net_class,
				 batch_size,
				 num_epochs,
				 iteration_ids,
				 val_set_split,
				 augment_data=False,
				 whiten_data=False,
				 learning_rate=None,
				 lr_decay=None,
				 milestones=None,
				 momentum=None,
				 l2_penalty=None,
				 pre_net_class=None,
				 pre_net_mdl_path=None,
				 pre_net_out=None):
		self.CONFIG_FAMILY = config_family
		self.CONFIG_NAME = config_name
		self.CONFIG_ID = self.CONFIG_FAMILY + '/' + self.CONFIG_NAME

		self.Net = net_class

		self.BATCH_SIZE = batch_size
		self.NUM_EPOCHS = num_epochs
		self.ITERATION_IDS = iteration_ids

		# Paths where to save the model
		self.MDL_PATH = {}
		# Path where to save accuracy plot
		self.ACC_PLT_PATH = {}
		# Path where to save kernel images
		self.KNL_PLT_PATH = {}
		for iter_id in self.ITERATION_IDS:
			# Path where to save the model
			self.MDL_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/save/model' + str(iter_id) + '.pt'
			# Path where to save accuracy plot
			self.ACC_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/accuracy' + str(iter_id) + '.png'
			# Path where to save kernel images
			self.KNL_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/kernels' + str(iter_id) + '.png'
		# Path to the CSV where test results are saved
		self.CSV_PATH = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/test_results.csv'

		# Define the splitting point of the training batches between training and validation datasets
		self.VAL_SET_SPLIT = val_set_split

		# Define whether to apply data augmentation or whitening
		self.AUGMENT_DATA = augment_data
		self.WHITEN_DATA = whiten_data

		self.LEARNING_RATE = learning_rate # Initial learning rate, periodically decreased by a lr_scheduler
		self.LR_DECAY = lr_decay # LR decreased periodically by a factor of 10
		self.MILESTONES = milestones # Epochs at which LR is decreased
		self.MOMENTUM = momentum
		self.L2_PENALTY = l2_penalty

		self.PreNet = pre_net_class
		self.PRE_NET_MDL_PATH = pre_net_mdl_path
		self.PRE_NET_OUT = pre_net_out


CONFIG_LIST = [


	# SGD

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='sgd_base', # Val: 87.28
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
	),

	# Hebb

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='hebb_base', # Val: 24.28
		net_class=hebbmodel.model.Net,
		batch_size=64,
		num_epochs=5,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
	),

	# Multi

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='hebb1_sgd', # Val: 85.15
		net_class=basemodel.top1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=7e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='hebb2_sgd', # Val: 79.15
		net_class=basemodel.top2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='hebb3_sgd', # Val: 68.34
		net_class=basemodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='hebb4_sgd', # Val: 57.77
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN4
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_fc5', # Val: 41.49
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN5
	),
]


CONFIGURATIONS = {}
for c in CONFIG_LIST: CONFIGURATIONS[c.CONFIG_ID] = c
