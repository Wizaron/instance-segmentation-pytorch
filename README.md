
# Semantic Instance Segmentation with a Discriminative Loss Function

This repository implements [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551). However, it contains some enhancements.

* Reference paper does not predict semantic segmentation mask, instead it uses ground-truth semantic segmentation mask. This code predicts semantic segmentation mask, similar to [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591).
* Reference paper predicts the number of instances implicity. It predicts embeddings for instances and predicts the number of instances as a result of clustering. Instead, this code predicts the number of instances as an output of network.
* This code uses [Spectral Clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html); however, reference paper uses "a fast variant of the mean-shift algorithm".
* Reference paper uses a segmentation network based on [ResNet-38](https://arxiv.org/abs/1512.03385). Instead, this code uses [ReSeg](https://arxiv.org/abs/1511.07053) with skip-connections based on first seven convolutional layers of [VGG16](https://arxiv.org/abs/1409.1556) as segmentation network.

----------------------------

In prediction phase, network inputs an image and outputs a semantic segmentation mask, the number of instances and embeddings for all pixels in the image. Then, foreground embeddings (which correspond to instances) are selected using semantic segmentation mask and foreground embeddings are clustered into "the number of instances" groups using spectral clustering.

# Installation

* Clone this repository : `git clone --recursive https://github.com/Wizaron/instance-segmentation-pytorch.git`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Create a conda environment :  `conda env create -f instance-segmentation-pytorch/code/conda_environment.yml`

## Data

### CVPPP

* Download [CVPPP dataset](https://www.plant-phenotyping.org/datasets-download) and extract downloaded zip file (`CVPPP2017_LSC_training.zip`) to `instance-segmentation-pytorch/data/raw/CVPPP/`
*  This work uses *A1* subset of the dataset.

### Cityscapes

* Download [Cityscapes dataset](https://www.cityscapes-dataset.com/) and extract downloaded zip files (`gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`) to `instance-segmentation-pytorch/data/raw/cityscapes/`

## Code Structure

* **code**: Codes for training and evaluation.
	* **lib**
		* **lib/cvppp_arch.py**: Defines network architecture for CVPPP dataset.
		* **lib/cityscapes_arch.py**: Defines network architecture for cityscapes dataset.
		* **lib/model.py**: Defines model (optimization, criterion, fit, predict, test, etc.).
		* **lib/dataset.py**: Data loading, augmentation, minibatching procedures.
		* **lib/preprocess.py**, **lib/utils**: Augmentation methods.
		* **lib/prediction.py**: Prediction module.
		* **lib/losses/dice.py**: Dice loss for foreground semantic segmentation.
		* **lib/losses/discriminative.py**: [Discriminative loss](https://arxiv.org/pdf/1708.02551.pdf) for instance segmentation.
	* **settings**
		* **settings/CVPPP/data_settings.py**, **settings/cityscapes/data_settings.py**: Defines settings about data.
		* **settings/CVPPP/model_settings.py**, **settings/cityspcaes/model_settings.py**: Defines settings about model (hyper-parameters). 
		* **settings/CVPPP/training_settings.py**, **settings/cityscapes/training_settings.py**: Defines settings for training (optimization method, weight decay, augmentation, etc.).
	* **train.py**: Training script.
	* **pred.py**: Prediction script.
	* **pred_cvppp_validation.sh**: Runs predictions on validation subset of CVPPP dataset.
	* **pred_cityscapes_validation.sh**: Runs predictions on validation subset of cityscapes dataset.
	* **evaluate.py**: Evaluation script. Calculates SBD (symmetric best dice), |DiC| (absolute difference in count) and Foreground Dice (Dice score for semantic segmentation) as defined in the [paper](http://eprints.nottingham.ac.uk/34197/1/MVAP-D-15-00134_Revised_manuscript.pdf).
* **data**:  Stores data and scripts to prepare dataset for training and evaluation.
	* **metadata/CVPPP**, **metadata/cityscapes**: Stores metadata; such as, training, validation and test splits, image shapes etc.
	* **processed/CVPPP**, **metadata/cityscapes**: Stores processed form of the data.
	* **raw/CVPPP**, **raw/cityscapes**: Stores raw form of the data.
	* **scripts**: Stores scripts to prepare dataset.
		* **scripts/CVPPP**: For CVPPP dataset.
			* **scripts/CVPPP/1-create_annotations.py**: Saves annotations as a numpy array to `processed/CVPPP/semantic-annotations/` and `processed/CVPPP/instance-annotations`.
			* **scripts/CVPPP/1-remove_alpha.sh**: Removes alpha channels from images. (In order to run this script, `imagemagick` should be installed.).
			* **scripts/CVPPP/2-get_image_means-stds.py**: Calculates and prints channel-wise means and standard deviations from training subset.
			* **scripts/CVPPP/2-get_image_shapes.py**:  Saves image shapes to `metadata/CVPPP/image_shapes.txt`.
			* **scripts/CVPPP/2-get_number_of_instances.py**: Saves the number of instances in each image to `metadata/CVPPP/number_of_instances.txt`.
			* **scripts/CVPPP/3-create_dataset.py**: Creates an lmdb dataset to `processed/CVPPP/lmdb/`.
		* **scripts/cityscapes**: For cityscapes dataset.
			* **scripts/cityscapes/1-create-annotations.py**: Saves annotations as a numpy array to `processed/cityscapes/semantic-annotations` and `processed/cityscapes/instance-annotations`, saves the number of instances in each image to `metadata/cityscapes/number_of_instances.txt` and save subset lists to `metadata/cityscapes/training.lst` and `metadata/cityscapes/validation.lst`.
			* **scrits/cityscapes/3-create_dataset.py**: Creates an lmdb dataset to `processed/cityscapes/lmdb/`.
* **models/CVPPP**, **models/cityscapes**: Stores checkpoints of the trained models.
* **outputs/CVPPP**, **outputs/cityscapes**: Stores predictions of the trained models.

## Data Preparation

Data should be prepared prior to training and evaluation.

* Activate previously created conda environment : `source activate ins-seg-pytorch`

### CVPPP

* Place the extracted dataset to `instance-segmentation-pytorch/data/raw/CVPPP/`. Hence, raw dataset should be found at `instance-segmentation-pytorch/data/raw/CVPPP/CVPPP2017_LSC_training/`.
* In order to prepare the data go to `instance-segmentation-pytorch/data/CVPP/scripts` and 
	* `python 1-create_annotations.py`
	* `sh 1-remove_alpha.sh`
	* `python 3-create_dataset.py`

### Cityscapes

* Place the extracted datasets to `instance-segmentation-pytorch/data/raw/cityscapes/`. Hence, raw dataset should be found at `instance-segmentation-pytorch/data/raw/cityscapes/gtFine/` and `instance-segmentation-pytorch/data/raw/cityscapes/leftImg8bit/`.
* In order to prepare the data go to `instance-segmentation-pytorch/data/cityscapes/scripts` and
	* `python 1-create_annotations.py`
	* `python 3-create_dataset.py`

## Visdom Server

Start a [Visdom](https://github.com/facebookresearch/visdom) server in a `screen` or `tmux`.

* Activate previously created conda environment : `source activate ins-seg-pytorch`

* Start visdom server : `python -m visdom.server`

* Access visdom server using `http://localhost:8097`

## Training

* Activate previously created conda environment : `source activate ins-seg-pytorch`

* Go to `instance-segmentation-pytorch/code/` and run `train.py`

```
usage: train.py [-h] [--model MODEL] [--usegpu] [--nepochs NEPOCHS]
                [--batchsize BATCHSIZE] [--debug] [--nworkers NWORKERS]
                --dataset DATASET

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Filepath of trained model (to continue training)
                        [Default: '']
  --usegpu              Enables cuda to train on gpu [Default: False]
  --nepochs NEPOCHS     Number of epochs to train for [Default: 600]
  --batchsize BATCHSIZE
                        Batch size [Default: 2]
  --debug               Activates debug mode [Default: False]
  --nworkers NWORKERS   Number of workers for data loading (0 to do it using
                        main process) [Default : 2]
  --dataset DATASET     Name of the dataset: "cityscapes" or "CVPPP"
```

Debug mode plots pixel embeddings to visdom, it reduces size of the embeddings to two-dimensions using TSNE. Hence, it slows down training.

As training continues, models will be saved to `instance-segmentation-pytorch/models/CVPPP` or `instance-segmentation-pytorch/models/cityscapes` according to the dataset argument.

## Evaluation

After training is complete, we can make predictions.

* Activate previously created conda environment : `source activate ins-seg-pytorch`

* Go to `instance-segmentation-pytorch/code/`.
* Run `./pred_cvppp_validation.sh [model_path]` to make predictions on validation subset of CVPPP dataset or run `./pred_cityscapes_validation.sh [model_path]` to make predictions on validation subset of cityscapes dataset. It will save predictions to `instance-segmentation-pytorch/outputs/CVPPP` or `instance-segmentation-pytorch/outputs/cityscapes`.

For example: `./pred_cvppp_validation.sh ../models/CVPPP/2018-2-24_16-1_jcmaxwell_6-459906/model_155_0.125209495425.pth`

These shell scripts calls `pred.py` for a list of images.

```
usage: pred.py [-h] --image IMAGE --model MODEL [--usegpu] --output OUTPUT
               [--n_workers N_WORKERS] --dataset DATASET

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         path of the image
  --model MODEL         path of the model
  --usegpu              enables cuda to predict on gpu
  --output OUTPUT       path of the output directory
  --n_workers N_WORKERS
                        number of workers for clustering
  --dataset DATASET     Name of the dataset: "cityscapes" or "CVPPP"
```

* After prediction is complete we can run `python evaluate.py --pred_dir [prediction_directory] --dataset [dataset_name]`. It will print metrics to the stdout.

For example: `python evaluate.py --pred_dir ../outputs/CVPPP/2018-2-25_15-21_jcmaxwell_41-782743-model_257_0.124301478267 --dataset CVPPP`

```
usage: evaluate.py [-h] --pred_dir PRED_DIR --dataset DATASET

optional arguments:
  -h, --help           show this help message and exit
  --pred_dir PRED_DIR  prediction directory
  --dataset DATASET    Name of the dataset: "cityscapes" or "CVPPP"
```

## Results

### CVPPP

#### Scores on validation subset (28 images)

| SBD           | \|DiC\|       | Foreground Dice  |
|:-------------:|:-------------:|:----------------:|
| 87.4          | 0.6           | 96.8             |

#### Sample Predictions

![plant007 image](samples/CVPPP/plant007_rgb.png) ![plant007 image](samples/CVPPP/plant007_rgb-ins_mask_color.png) ![plant007 image](samples/CVPPP/plant007_rgb-fg_mask.png)
![plant031 image](samples/CVPPP/plant031_rgb.png?raw=true "plant031 image") ![plant031 image](samples/CVPPP/plant031_rgb-ins_mask_color.png?raw=true "plant031 instance segmentation") ![plant031 image](samples/CVPPP/plant031_rgb-fg_mask.png?raw=true "plant031 foreground segmentation")
![plant033 image](samples/CVPPP/plant033_rgb.png?raw=true "plant033 image") ![plant033 image](samples/CVPPP/plant033_rgb-ins_mask_color.png?raw=true "plant033 instance segmentation") ![plant033 image](samples/CVPPP/plant033_rgb-fg_mask.png?raw=true "plant033 foreground segmentation")

### Cityscapes

#### Scores on validation subset

TODO

#### Sample Predictions

TODO
