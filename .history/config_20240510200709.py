import params as P
import basemodel.model, basemodel.model_no_bias, basemodel.model_bn_before_relu, \
	basemodel.model1, basemodel.model1_no_bias, basemodel.model2, basemodel.model3, basemodel.model4, \
	basemodel.top1, basemodel.top2, basemodel.top3, basemodel.top4, basemodel.fc, basemodel.fc_no_bias, \
	basemodel.triplefc
import hebbmodel.model, hebbmodel.model1, hebbmodel.model1_som, \
	hebbmodel.top1, hebbmodel.top2, hebbmodel.top3, hebbmodel.top4, hebbmodel.fc, \
	hebbmodel.top4_bw, hebbmodel.triplefc, hebbmodel.triplefc_bw, \
	hebbmodel.g1h2_6, hebbmodel.g1_2h3_6, hebbmodel.g1_3h4_6, hebbmodel.g1_4h5_6


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


	################################################################################################################
	####										GDES CONFIGURATIONS												####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='sgd_base20', # Train:97.95 Val:87.22, Test:
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

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='sgd_base5', # Train:79.10 Val:72.51, Test:
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=5,
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

	################################################################################################################
	####					CONFIGS: GDES CLASSIFIER ON FEATURES EXTRACTED FROM HEBB LAYERS						####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='hebb5_1_gdes20', # Val: 64.76, Test: 63.92
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/hebb/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),

	################################################################################################################
	####										HEBB CONFIGURATIONS												####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='hebb_base', # Val: 41.65, Test: 41.84
		net_class=hebbmodel.model1.Net,
		batch_size=64,
		num_epochs=5,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='hebb_base5', # Val: 27.52, Test: 28.59
		net_class=hebbmodel.model.Net,
		batch_size=64,
		num_epochs=5,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
	),

	################################################################################################################
	####					CONFIGS: HEBB CLASSIFIER ON FEATURES EXTRACTED FROM GDES LAYERS 					####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='gdes20_5_hebb5', # Train:98.10 Val:87.38, Test:
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=5,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/sgd_base20/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='gdes5_5_hebb5', # Train:75.60 Val:72.66, Test:
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=5,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/sgd_base5/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='gdes_4_hebb',
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net, # Val: 82.73, Test: 82.18
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/sgd_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
	),

	################################################################################################################
	####									CONFIGS BACKWARD INTERACTION										####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='3fc_on_hebb_conv1', # Val: 70.57, Test: 69.67
		net_class=basemodel.triplefc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='3fc_on_hebb_conv1', # Val: 28.95, Test: 29.11
		net_class=hebbmodel.triplefc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='3fcbw_on_hebb_conv1', # Val: 23.21, Test: 23.22
		net_class=hebbmodel.triplefc_bw.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=True,
		pre_net_class=hebbmodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_1l/save/model0.pt',
		pre_net_out=hebbmodel.model1.Net.BN1
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top4_bw',
		net_class=hebbmodel.top4_bw.Net,  # Val: 83.54, Test: 83.04
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='3fc_on_gdes_conv4', # Val: 84.10, Test: 83.82
		net_class=basemodel.triplefc.Net,
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
		l2_penalty=5e-2,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='3fc_on_gdes_conv4', # Val: 82.48, Test: 82.20
		net_class=hebbmodel.triplefc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='3fcbw_on_gdes_conv4', # Val: 80.63, Test: 80.56
		net_class=hebbmodel.triplefc_bw.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4
	),

]


CONFIGURATIONS = {}
for c in CONFIG_LIST: CONFIGURATIONS[c.CONFIG_ID] = c
