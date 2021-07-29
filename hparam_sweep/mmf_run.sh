#!/bin/bash
export OC_DISABLE_DOT_ACCESS_WARNING=1

mmf_run config="projects/visual_bert/configs/hateful_memes/from_coco.yaml" \
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
        dataset_config.hateful_memes.annotations.train[0]=$1 \
        dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
        dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
        dataset_config.hateful_memes.features.train[0]=$2 \
        dataset_config.hateful_memes.features.val[0]=$2 \
        dataset_config.hateful_memes.features.test[0]=$2 \
        training.lr_ratio=$7 \
        training.use_warmup=True \
        optimizer.params.lr=$6 \
      	training.batch_size=$5 \
        env.tensorboard_logdir=$3 \
        env.save_dir=$4 \