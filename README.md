# majorprojectbatch24
#installing anomalib
pip install anomalib

#with anaconda environment
yes | conda create -n anomalib_env python=3.8
conda activate anomalib_env
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e .

#training
python tools/train.py    # Train PADIM on MVTec AD leather
python tools/train.py --config <path/to/model/config.yaml>
python tools/train.py --config src/anomalib/models/padim/config.yaml
python tools/train.py --model padim

#feature extraction
model:
  name: cflow
  backbone: wide_resnet50_2
  pre_trained: true

#custom dataset
dataset:
  name: <name-of-the-dataset>
  format: folder
  path: <path/to/folder/dataset>
  normal_dir: normal # name of the folder containing normal images.
  abnormal_dir: abnormal # name of the folder containing abnormal images.
  normal_test_dir: null # name of the folder containing normal test images.
  task: segmentation # classification or segmentation
  mask: <path/to/mask/annotations> #optional
 
extensions: null
  split_ratio: 0.2 # ratio of the normal images that will be used to create a test split
  image_size: 256
  train_batch_size: 32
  test_batch_size: 32
  num_workers: 8
  transform_config:
    train: null
    val: null
  create_validation_set: true
  tiling:
    apply: false
    tile_size: null
    stride: null
    remove_border_count: 0
    use_random_tiling: False
    random_tile_count: 16

#inference
python tools/inference/lightning_inference.py -h
optimization:
  export_mode: "openvino" # options: openvino, onnx

#Hyperparameter optimization
python tools/hpo/sweep.py \
    --model padim --model_config ./path_to_config.yaml \
    --sweep_config tools/hpo/sweep.yaml

#benchmarking
python tools/benchmarking/benchmark.py \
    --config <relative/absolute path>/<paramfile>.yaml


 
#experiment management
visualization:
  log_images: True # log images to the available loggers (if any)
  mode: full # options: ["full", "simple"]

 logging:
  logger: [comet, tensorboard, wandb]
  log_graph: True
