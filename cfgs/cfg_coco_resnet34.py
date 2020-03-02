'''config file for resnet34-coco'''


# anchors
ANCHOR_RATIOS = [0.5, 1, 2]
ANCHOR_SCALES = [1, 2**(1.0/3.0), 2**(2.0/3.0)]
ANCHOR_BASE_SIZES = [32, 64, 128, 256, 512]
BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# backbone
BACKBONE_TYPE = 'resnet34'
PRETRAINED_MODEL_PATH = ''
IS_MULTI_GPUS = True
ADDED_MODULES_WEIGHT_INIT_METHOD = None
# dataset
DATASET_ROOT_DIR = ''
MAX_NUM_GT_BOXES = 50
NUM_CLASSES = 81
NUM_WORKERS = 8
PIN_MEMORY = True
BATCHSIZE = 16
CLSNAMESPATH = 'names/coco.names'
USE_COLOR_JITTER = False
IMAGE_NORMALIZE_INFO = {'mean_rgb': (0.485, 0.456, 0.406), 'std_rgb': (0.229, 0.224, 0.225)}
# loss function
CLS_LOSS_SET = {'type': 'focal_loss', 'focal_loss': {'size_average': True, 'weight': 1., 'alpha': 0.25, 'gamma': 2.0}}
REG_LOSS_SET = {'type': 'betaSmoothL1Loss', 'betaSmoothL1Loss': {'size_average': True, 'weight': 1., 'beta': 0.11}}
# optimizer
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
LR_ADJUST_EPOCHS = [9, 12]
MAX_EPOCHS = 12
IS_USE_WARMUP = True
NUM_WARMUP_STEPS = 500
# image size
IMAGESIZE_DICT = {'LONG_SIDE': 800, 'SHORT_SIDE': 800}
# record
TRAIN_BACKUPDIR = 'retinanet_res34_trainbackup_coco'
TRAIN_LOGFILE = 'retinanet_res34_trainbackup_coco/train.log'
TEST_BACKUPDIR = 'retinanet_res34_testbackup_coco'
TEST_LOGFILE = 'retinanet_res34_testbackup_coco/test.log'
TEST_BBOXES_SAVE_PATH = 'retinanet_res34_testbackup_coco/retinanet_res34_detection_results_coco.json'
SAVE_INTERVAL = 1