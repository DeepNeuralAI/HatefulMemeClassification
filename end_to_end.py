# -*- coding: utf-8 -*-
"""End_to_End.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vk1pZIGgoHIk-8KZlnIbH2rbnxPm0rcC

## GPU Info & Mounting
"""

## GPU info

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

## Mount Drive

# from google.colab import drive

# drive.mount('/content/drive')
# root_path = 'drive/My Colab Notebooks/'  #change dir to your project folder

"""## Install & Imports"""

import os
home = "/content"
os.chdir(home)
os.getcwd()

!pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

"""### Install MMF from Source"""

!git clone --branch no_feats --config core.symlinks=true https://github.com/rizavelioglu/mmf.git

import os
os.chdir(os.path.join(home, "mmf"))

!pip install --editable .

"""## Convert to MMF format"""
zip_file_path="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/data/hateful_meme_data.zip"
!mmf_convert_hm --zip_file=$zip_file_path --password="" --bypass_checksum 1
!ls /root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/ | wc -l

""" Add Memotion """
import pandas as pd
!pip install -q kaggle
!mkdir -p ~/.kaggle
!mv $home/kaggle.json ~/.kaggle/
!chmod 660 /root/.kaggle/kaggle.json
!kaggle datasets download -d williamscott701/memotion-dataset-7k
!unzip -qq memotion-dataset-7k.zip -d $home/
memo_samples = pd.read_json("/content/hateful_memes-hate_detectron/utils/label_memotion.jsonl", lines=True)['img']
memo_samples = [i.split('/')[1] for i in list(memo_samples)]
img_dir = "/content/memotion_dataset_7k/images/")
for img in memo_samples:
    os.rename(f"{img_dir+img}", f"/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/{img}")

!ls /root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/ | wc -l

""" Add MMHS150K """
labeled_mmhs150k_gt = pd.read_json(os.path.join(home, "MMHS/MMHS150K_GT.json"), orient='index', convert_axes=False)
imgs_mmhs150k_gt = labeled_mmhs150k_gt.index
mmhs_dir = "/content/MMHS/img_resized/"
mmhs_df = {}
iter = 0
for x in imgs_mmhs150k_gt:
  img = x+".jpg"
  df = {}
  df['id'] = x
  df['img'] = "img/" + x + ".jpg"
  text = labeled_mmhs150k_gt.loc[x,'tweet_text']
  df['text'] = text.split("http",1)[0] 
  labels = labeled_mmhs150k_gt.loc[x,'labels_str']
  i = 0
  for x in labels: 
    if x == "NotHate": pass
    else: i+=1
  if i>2: label = 1
  elif i==0: label = 0
  else: label = -1
  df['label'] = label
  mmhs_df[iter] = df
  iter = iter+1
mmhs_df = pd.DataFrame.from_dict(mmhs_df, orient='index')
mmhs_df = mmhs_df[mmhs_df.label>=0]
mmhs_df_0 = mmhs_df[mmhs_df.label==0].sample(n=500, random_state=1)
mmhs_df_1 = mmhs_df[mmhs_df.label==1].sample(n=500, random_state=1)
mmhs_df_2 = mmhs_df_0.append(mmhs_df_1).reset_index(drop=True)
mmhs_json = mmhs_df_2.to_json(orient='records', lines=True)
with open(os.path.join(home, "mmhs_op.jsonl"), "w", encoding='utf-8') as f:
    f.write(mmhs_json)
for x in mmhs_df_2['id']:
  img = x+".jpg"
  os.rename(f"{mmhs_dir+img}", f"/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/{img}")

!ls /root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/ | wc -l

"""Add Misogynstic Meme Dataset """
labeled_miso_gt = pd.read_json(os.path.join(home, "Miso/Misogynistic-MEME/train_miso.jsonl"), lines=True)
labeled_miso_gt = labeled_miso_gt[['id', 'img', 'text', 'label']]
miso_json = labeled_miso_gt.to_json(orient='records', lines=True)
with open(os.path.join(home, "miso_op.jsonl"), "w", encoding='utf-8') as f:
    f.write(miso_json)
miso_dir = os.path.join(home, "Miso/Misogynistic-MEME/")
for img1 in labeled_miso_gt['img']:
    x = img1.split('/')[1]
    os.rename(f"{miso_dir+img1}", f"/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/{x}")

""" Merging the datasets """
!python $home/concat_memo_mmhs_miso_hm.py --home $home

"""Check class imbalance """
check1 = pd.read_json(os.path.join(home, "train_miso.jsonl"), lines=True)
check1.groupby('label').count()
per_cls_weights = []
beta = 0.9999
cls_num_list = [3419, 5881]
for n in cls_num_list:
        term =  (1-beta**n)/(1-beta)
        per_cls_weights.append(term)
print(per_cls_weights)

"""## Feature Extraction

### VQA Mask-RCNN
"""
import os
os.chdir(home)
!git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark

!pip install ninja yacs cython matplotlib

os.chdir(os.path.join(home, "vqa-maskrcnn-benchmark"))
!rm -rf build
!python setup.py build develop

"""### Extract"""

os.chdir(os.path.join(home, "mmf/tools/scripts/features/"))
out_folder = os.path.join(home, "features/")

!python extract_features_vmb.py --config_file "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model_x152.yaml" \
                                --model_name "X-152" \
                                --output_folder $out_folder \
                                --image_dir "/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/" \
                                --num_features 100 \

os.chdir(home)
# !zip -r "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/features.zip" features

"""## Fine-tuning w/ VisualBERT"""

# os.chdir("/content/drive/MyDrive/Colab Notebooks/hateful_memes/")
# os.getcwd()
# # !unzip "features.zip"

os.chdir(home)
os.getcwd()

log_dir ="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/baseline/visual_bert"
save_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/submissions/baseline/visual_bert/submission_1/"
feats_dir = "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/features/"
train_dir = "hateful_memes/defaults/annotations/train.jsonl"

"""### Run Model with Default Config"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir $log_dir

# # Define where train.jsonl is
# train_dir = os.path.join(home, "train_v9.jsonl")

!mmf_run \
        config="projects/visual_bert/configs/hateful_memes/from_coco.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=100 \
        training.evaluation_interval=100 \
        training.max_updates=3000 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        training.lr_ratio=0.3 \
        training.use_warmup=True \
        training.batch_size=32 \
        optimizer.params.lr=5.0e-05 \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$log_dir \




"""### Run VisualBERT Model with Hyper-Parameter Sweep x27"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/hyperparameter_sweep/logs/"

hyper_params_path="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/hyperparameter_sweep"

os.chdir(hyper_params_path)
# Give rights to bash script to be executable
!chmod +x sweep.sh
# Define where image features are
feats_dir = "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/features/"
# Define where train.jsonl is
train_dir = "hateful_memes/defaults/annotations/train.jsonl"

# Start hyper-parameter search
!python sweep.py --home $home --feats_dir $feats_dir --train $train_dir

"""### Run Single Model with Modified Hyper-parameter Configs

#### Early Stopping: True
#### FP16 Precision: True
#### Max Updates: 1000
#### LR Ratio: 0.1
#### Training Warmup: False
"""

os.chdir(home)
os.getcwd()

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/"

tensorboard_log_dir ="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/"
save_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/submissions/"
feats_dir = "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/features/"
train_dir = "hateful_memes/defaults/annotations/train.jsonl"

!mmf_run \
        config="projects/visual_bert/configs/hateful_memes/from_coco.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=3000 \
        training.log_interval=100 \
        training.lr_ratio=0.1 \
        training.use_warmup=False \
        training.batch_size=40 \
        training.evaluate_metrics=True \
        training.early_stop.enabled=True \
        training.early_stop.criteria='hateful_memes/roc_auc' \
        training.early_stop.minimize=False \
        training.fp16=True \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$tensorboard_log_dir \

"""## Fine-Tuning via ViLBERT"""

import os
os.chdir(home)
os.getcwd()

log_dir ="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/baseline/vilbert/"
# save_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/submissions/baseline/vilbert/"
save_dir="/content/submissions/baseline/vilbert/"
feats_dir = "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/features/"
train_dir = "hateful_memes/defaults/annotations/train.jsonl"

!kill 235

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/baseline/"

!mmf_run \
        config="projects/vilbert/configs/hateful_memes/from_cc.yaml" \
        model="vilbert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=vilbert.finetuned.hateful_memes.from_cc_original \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=5000 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        training.lr_ratio=0.3 \
        training.use_warmup=True \
        training.batch_size=40 \
        training.evaluate_metrics=True \
        optimizer.params.lr=5.0e-05 \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$log_dir \

"""## Hyper-Parameter Sweep with Best Config of VisualBERT + Dataset Expansion"""

# !unzip '/content/drive/MyDrive/Colab_Notebooks/hateful_memes/misogynistic_meme_features/features_miso.zip' -d '/content/drive/MyDrive/Colab_Notebooks/hateful_memes/misogynistic_meme_features/features'

# import sys
# sys.path.append('/content/drive/MyDrive/Colab_Notebooks/hateful_memes')
# from hparams_sweep import run_sweep
# from subprocess import call

train_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/misogynistic_meme_features/train_miso.jsonl"
feats_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/misogynistic_meme_features/features/content/features"
log_dir="/content/drive/MyDrive/Colab_Notebooks/hateful_memes/logs/best/visual_bert"
save_dir="/content/submissions/best/visual_bert"

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir $log_dir

hparams = {
        "batch_size": [30, 40, 50],
        "learning_rate": [5e-3, 5e-5, 5e-6],
        "lr_ratio": [0.1, 0.3, 0.5]
}

!OC_DISABLE_DOT_ACCESS_WARNING=1

for batch_size in hparams['batch_size']:
        for lr in hparams['learning_rate']:
            for ratio in hparams['lr_ratio']:

              log_name = f'{batch_size}_{lr}_{ratio}'
              log_path = f'{log_dir}/{log_name}'
              save_path = f'{save_dir}/{log_name}'

              print('==================================================')
              print(f'Current: {log_name}')
              print('==================================================\n')

              !mmf_run \
                config="projects/visual_bert/configs/hateful_memes/from_coco.yaml" \
                model="visual_bert" \
                dataset=hateful_memes \
                run_type=train_val \
                checkpoint.max_to_keep=1 \
                checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
                training.tensorboard=True \
                training.checkpoint_interval=100 \
                training.evaluation_interval=100 \
                training.max_updates=3000 \
                training.log_interval=100 \
                training.early_stop.enabled=True \
                training.early_stop.criteria='hateful_memes/roc_auc' \
                training.early_stop.minimize=False \
                dataset_config.hateful_memes.max_features=100 \
                dataset_config.hateful_memes.annotations.train[0]=$train_dir \
                dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
                dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
                dataset_config.hateful_memes.features.train[0]=$feats_dir \
                dataset_config.hateful_memes.features.val[0]=$feats_dir \
                dataset_config.hateful_memes.features.test[0]=$feats_dir \
                training.lr_ratio=$ratio \
                training.use_warmup=True \
                training.batch_size=$batch_size \
                optimizer.params.lr=$lr \
                env.save_dir=$save_path\
                env.tensorboard_logdir=$log_path 
              
              print('==================================================')
              print('Deleting model files for space saving')
              !rm -rf $save_path
              print('==================================================\n')


''' Loss Experiments '''
""" Use custom config  from_coco_mlsml.yaml, from_coco_focal.yaml, from_coco_mrl.yaml for different losses """
""" Add code from custom_losses.py to /content/mmf/mmf/modules/losses.py """
!mmf_run \
        config="/content/from_coco_mlsml.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=1000 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        training.lr_ratio=0.3 \
        training.use_warmup=True \
        training.batch_size=32 \
        optimizer.params.lr=5.0e-05 \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$log_dir \

!mmf_run \
        config="/content/from_coco_focal.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=1000 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        training.lr_ratio=0.3 \
        training.use_warmup=True \
        training.batch_size=32 \
        optimizer.params.lr=5.0e-05 \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$log_dir \


!mmf_run \
        config="/content/from_coco_mrl.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=1000 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        dataset_config.hateful_memes.annotations.train[0]=$train_dir \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$feats_dir \
        dataset_config.hateful_memes.features.val[0]=$feats_dir \
        dataset_config.hateful_memes.features.test[0]=$feats_dir \
        training.lr_ratio=0.3 \
        training.use_warmup=True \
        training.batch_size=32 \
        optimizer.params.lr=5.0e-05 \
        env.save_dir=$save_dir \
        env.tensorboard_logdir=$log_dir \