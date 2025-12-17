#!/bin/bash
. ./path.sh || exit 1;
unset LD_LIBRARY_PATH
export PYTHONPATH=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr:$PYTHONPATH
export PATH="/home/maqy23/anaconda3/envs/mlc/bin:$PATH"  # 确保虚拟环境的路径优先级最高
# export PATH="$PATH:/home/maqy23/anaconda3/envs/mlc/bin
# export LD_LIBRARY_PATH=~/my_libs:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR="/home/maqy23/MLC-SLM-Baseline-docker/.triton_cache"
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
# export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRITON_CACHE_DIR="/home/maqy23/slp_docker/MLC-SLM-Baseline-docker/.triton_cache" # Triton cache directory
# export TRITON_CACHE_DIR="/root/.triton/autotune"

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=0

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023
nj=30

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
data_ratio=35 # 训练数据集的比例，默认是10%，最大为35，总计115万
num_utts_per_shard=1000

train_data_dir=/home/maqy23/xjgd/data/train

train_segments_path=$train_data_dir/segments
train_split_dir=$train_data_dir/split
train_split_log_dir=$train_data_dir/split_log

dev_data_dir=/home/maqy23/xjgd/data/dev
dev_segments_path=$dev_data_dir/segments
dev_split_dir=$dev_data_dir/split
dev_split_log_dir=$dev_data_dir/split_log

# 如果有测试集也类似设置
test_data_dir=/home/maqy23/xjgd/data/test
test_segments_path=$test_data_dir/segments
test_split_dir=$test_data_dir/split
test_split_log_dir=$test_data_dir/split_log

cmd=local/run.pl



step1_train_config=conf/train_mlcslm_baseline_step1.yaml
step2_train_config=conf19/train_mlcslm_baseline_step2.yaml

# step1_dir=expgdep8ds35/step1
# step2_dir=expgdep8ds35/step2
step1_dir=expgdep8ds35/step1
step2_dir=expgdep19ds35/step2
# test使用
# max_epoch=24

tensorboard_dir=tensorboard
step1_checkpoint=
# step2_checkpoint=expgdep8ds35/step2/epoch_36.pt
step2_checkpoint=
# =expgdep10ds35/step2/epoch_13.pt
# step2_checkpoint=exp/step2/epoch_0
num_workers=8
prefetch=10

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$step2_dir/avg_3.pt
average_num=3
decode_mode=whisper_llm_decode

train_engine=deepspeed

deepspeed_config=conf/ds_stage2.json
deepspeed_save_states=""model+optimizer""

. tools/parse_options.sh || exit 1;

#生成切割标准，切割的数据每条需要有4部分组成
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Generate segments"

    ln -s /home/maqy23/MLC-SLM-Baseline-docker/xjgd_train_data_raw ./xjgd_train_data
    ln -s /home/maqy23/MLC-SLM-Baseline-docker/xjgd_dev_data_raw ./xjgd_dev_data
    python local/prepare_segments.py --data_dir $train_data_dir --segments_path $train_segments_path
    log "Segments file of training dataset is saved into $train_segments_path"

    python local/prepare_segments.py --data_dir $dev_data_dir --segments_path $dev_segments_path
    log "Segments file of development dataset is saved into $dev_segments_path"
    # About 1min
fi
# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     log "Stage 0: Generate segments"

#     # ln -s /home/maqy23/MLC-SLM-Baseline-docker/test_data1_raw ./test_data
#     python local/prepare_segments.py --data_dir $test_data_dir --segments_path $test_segments_path
#     log "Segments file of training dataset is saved into $test_segments_path"

#     # About 1min
# fi

# test_data_dir=test_data
# test_segments_path=test_data/segments 
# test_split_dir=test_data/split
# test_split_log_dir=test_data/split_log

# 切割数据
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: Split wavs using segments"
    
  log "Split train dataset wavs"
  nutt=$(<${train_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $train_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${train_split_log_dir}/segments.${n}"
  done
  local/split_scp.pl "${train_segments_path}" ${split_segments}
  
   
  
  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${train_split_log_dir}/split_wavs.JOB.log" \
      python local/split_wav.py \
          "--segments_path=${train_split_log_dir}/segments.JOB" \
          "--output_dir=${train_split_dir}/split.JOB"

  log  "Split segments done, split wavs"
  
  cat ${train_split_dir}/split.*/wav.scp | shuf > $train_data_dir/wav.scp
  cat ${train_split_dir}/split.*/text | shuf > $train_data_dir/text

  log "Split dev dataset wavs"
  nutt=$(<${dev_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $dev_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${dev_split_log_dir}/segments.${n}"
  done
  local/split_scp.pl "${dev_segments_path}" ${split_segments}

  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${dev_split_log_dir}/split_wavs.JOB.log" \
      python local/split_wav.py \
          "--segments_path=${dev_split_log_dir}/segments.JOB" \
          "--output_dir=${dev_split_dir}/split.JOB"

  cat ${dev_split_dir}/split.*/wav.scp | shuf > $dev_data_dir/wav.scp
  cat ${dev_split_dir}/split.*/text | shuf > $dev_data_dir/text
  
  log "Split wavs done"
  # about 21h 10min
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Simple text normalization for the text file"
    
    python local/text_normalization.py \
        --input $train_data_dir/text \
        --output $train_data_dir/text_tn

    python local/text_normalization.py \
        --input $dev_data_dir/text \
        --output $dev_data_dir/text_tn

    python local/text_normalization.py \
        --input $test_data_dir/text \
        --output $test_data_dir/text_tn

    log "Text normalization done"
    # about 1min
fi
# 数据准备，制作数据列表，记得修改数据路径
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 3: Prepare data, prepare required format"
  
  if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $train_data_dir/wav.scp $train_data_dir/text_tn \
      $(realpath $train_data_dir/shards) $train_data_dir/${data_type}_data.list
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $dev_data_dir/wav.scp $dev_data_dir/text_tn \
      $(realpath $dev_data_dir/shards) $dev_data_dir/${data_type}_data.list

    # 实际用不到
    # tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
    #   --num_threads $nj $test_data_dir/wav.scp $test_data_dir/text_tn \
    #   $(realpath $test_data_dir/shards) $test_data_dir/${data_type}_data.list      
    # about 37min
  else
    # tools/make_raw_list.py $test_data_dir/wav.scp $test_data_dir/text_tn \
    #   $test_data_dir/${data_type}_data.list
    tools/make_raw_list.py $train_data_dir/wav.scp $train_data_dir/text_tn \
      $train_data_dir/${data_type}_data.list
    tools/make_raw_list.py $dev_data_dir/wav.scp $dev_data_dir/text_tn \
      $dev_data_dir/${data_type}_data.list
    # about 1min
  fi

  log "Data preparation done"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  EXCLUDE_PATTERN="DATA"

  train_list_file="$train_data_dir/${data_type}_data.list"
  dev_list_file="$dev_data_dir/${data_type}_data.list"

  if [ "$EXCLUDE_PATTERN" != "PLEASE_DEFINE_YOUR_EXCLUDE_PATTERN_HERE" ] && [ -n "$EXCLUDE_PATTERN" ]; then
    log "Filtering data lists to exclude lines matching: '$EXCLUDE_PATTERN'"

    # 过滤训练数据列表
    if [ -f "$train_list_file" ]; then
      original_train_lines=$(wc -l < "$train_list_file")
      log "Original train list ($train_list_file) has $original_train_lines lines."
      mv "$train_list_file" "${train_list_file}.original"
      grep -v "$EXCLUDE_PATTERN" "${train_list_file}.original" > "$train_list_file"
      filtered_train_lines=$(wc -l < "$train_list_file")
      log "Filtered train list ($train_list_file) now has $filtered_train_lines lines."
      if [ "$original_train_lines" -eq "$filtered_train_lines" ]; then
        log "WARNING: No lines were excluded from the train list. Check your EXCLUDE_PATTERN."
      fi
    else
      log "WARNING: Train data list $train_list_file not found. Skipping filtering."
    fi

    # 过滤开发数据列表
    if [ -f "$dev_list_file" ]; then
      original_dev_lines=$(wc -l < "$dev_list_file")
      log "Original dev list ($dev_list_file) has $original_dev_lines lines."
      mv "$dev_list_file" "${dev_list_file}.original"
      grep -v "$EXCLUDE_PATTERN" "${dev_list_file}.original" > "$dev_list_file"
      filtered_dev_lines=$(wc -l < "$dev_list_file")
      log "Filtered dev list ($dev_list_file) now has $filtered_dev_lines lines."
      if [ "$original_dev_lines" -eq "$filtered_dev_lines" ]; then
        log "WARNING: No lines were excluded from the dev list. Check your EXCLUDE_PATTERN."
      fi
    else
      log "WARNING: Dev data list $dev_list_file not found. Skipping filtering."
    fi
  else
    log "INFO: EXCLUDE_PATTERN is not defined or is empty. No filtering will be applied to data lists."
  fi
  # --- 数据列表过滤结束 ---

  log "Data preparation done"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Stage 4: Training step 1 start"

  # !!! Run below commands to prepare wenet-style whisper-large-v3 !!!
  # Download whisper ckpt from this [link](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17-L30)
  # Convert openai-style ckpt to wenet-style ckpt:
  # python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
  #   --whisper_ckpt pretrained-models/vanilla_whisper_large_v3/large-v3.pt \
  #   --output_dir pretrained-models/vanilla_whisper_large_v3
    # python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
    # --whisper_ckpt /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_large_v3/large-v3.pt \
    # --output_dir /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_large_v3
    # python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
    # --whisper_ckpt /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_v3_turbo/large-v3-turbo.pt \
    # --output_dir /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_v3_turbo
    # python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
    # --whisper_ckpt /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_small/small.pt \
    # --output_dir /home/maqy23/MLC-SLM-Baseline/examples/mlcslm/asr/pretrained-models/vanilla_whisper_small
    
  
  mkdir -p $step1_dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # dist_backend="gloo"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  step1_train_config=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step1/train_mlcslm_baseline_step1.yaml
  step1_checkpoint=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step1/epoch_8.pt
  echo "step1_train_config is $step1_train_config   step1_checkpoint is $step1_checkpoint"

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_mlcslm_baseline.py \
      --train_engine ${train_engine} \
      --config $step1_train_config \
      --data_type  $data_type \
      --train_data $train_data_dir/${data_type}_data_${data_ratio}.list \
      --cv_data $dev_data_dir/${data_type}_data.list \
      ${step1_checkpoint:+--checkpoint $step1_checkpoint} \
      --model_dir $step1_dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     if [ "$deepspeed_save_states" = "model+optimizer" ]; then
#     for subdir in $(find "$step1_dir" -maxdepth 1 -type d | grep -v "^$step1_dir$")
#     do
#       tag=$(basename "$subdir")
#       echo "$tag"
#       python3 ${step1_dir}/zero_to_fp32.py \
#         ${step1_dir} ${step1_dir}/${tag}.pt -t ${tag}
#       rm -rf ${step1_dir}/${tag}
#     done
#   fi
# fi
# 训练后转换
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
step1_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step1
  if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    for subdir in $(find "$step1_dir" -maxdepth 1 -type d | grep -v "^$step1_dir$"); do
      tag=$(basename "$subdir")
      output_file="${step1_dir}/${tag}.pt"

      # 如果已经转换过该目录，则跳过
      if [ -f "$output_file" ]; then
        echo "[SKIP] $tag already converted -> $output_file"
        continue
      fi

      # 检查目录中是否包含 *_optim_states.pt 文件
      if ls "$subdir"/*_optim_states.pt 1>/dev/null 2>&1; then
        echo "[CONVERT] $tag -> $output_file"
        python3 ${step1_dir}/zero_to_fp32.py \
          ${step1_dir} ${step1_dir}/${tag}.pt -t "$tag"
      else
        echo "[SKIP] $tag does not look like a DeepSpeed checkpoint"
      fi
    done
  fi
fi


# step2阶段
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  log "Stage 5: Training step 2 start"
  mkdir -p $step2_dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_mlcslm_baseline.py \
      --train_engine ${train_engine} \
      --config $step2_train_config \
      --data_type  $data_type \
      --train_data $train_data_dir/${data_type}_data_${data_ratio}.list \
      --cv_data $dev_data_dir/${data_type}_data.list \
      ${step2_checkpoint:+--checkpoint $step2_checkpoint} \
      --model_dir $step2_dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}

fi

# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#     if [ "$deepspeed_save_states" = "model+optimizer" ]; then
#     for subdir in $(find "$step2_dir" -maxdepth 1 -type d | grep -v "^$step2_dir$")
#     do
#       tag=$(basename "$subdir")
#       echo "$tag"
#       python3 ${step2_dir}/zero_to_fp32.py \
#         ${step2_dir} ${step2_dir}/${tag}.pt -t ${tag}
#       rm -rf ${step2_dir}/${tag}
#     done
#   fi
# fi

# 训练完成后，转换模型
# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#   if [ "$deepspeed_save_states" = "model+optimizer" ]; then
#     # step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2
#     step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2
#     # for subdir in "$step2_dir"/epoch_13; do
#     for subdir in $(find "$step2_dir" -maxdepth 1 -type d | grep -v "^$step2_dir$"); do
#     # for subdir in $(find "$step2_dir" -maxdepth 1 -type d -name "epoch_*"); do
#       [ -d "$subdir" ] || continue
#       tag=$(basename "$subdir")
#       echo "DEBUG: subdir = $subdir"


#       # # 只转换编号 >= 23 的 checkpoint
#       # epoch_num=$(echo "$tag" | grep -o -E '[0-9]+$')
#       # if [ -z "$epoch_num" ]; then
#       #   echo "[SKIP] $tag is not in valid format"
#       #   continue
#       # fi
#       # if [ "$epoch_num" -lt 23 ]; then
#       #   echo "[SKIP] $tag is before epoch_23"
#       #   continue
#       # fi
#       output_file="${step2_dir}/${tag}.pt"

#       # 如果已经转换过该目录，则跳过
#       if [ -f "$output_file" ]; then
#         echo "[SKIP] $tag already converted -> $output_file"
#         continue
#       fi

#       # 仅转换包含 1 个 *_optim_states.pt 的目录（单卡训练）
#       optim_count=$(ls "$subdir"/*_optim_states.pt 2>/dev/null | wc -l)
#       if [ "$optim_count" -eq 1 ]; then
#         echo "[CONVERT] $tag -> $output_file"
#         python3 ${step2_dir}/zero_to_fp32.py \
#           ${step2_dir} ${step2_dir}/${tag}.pt -t "$tag"
#       else
#         echo "[SKIP] $tag is not a valid single-card checkpoint (found $optim_count optim files)"
#       fi

#       # # 检查目录中是否包含 *_optim_states.pt 文件
#       # if ls "$subdir"/*_optim_states.pt 1>/dev/null 2>&1; then
#       #   echo "[CONVERT] $tag -> $output_file"
#       #   python3 ${step2_dir}/zero_to_fp32.py \
#       #     ${step2_dir} ${step2_dir}/${tag}.pt -t "$tag"
#       # else
#       #   echo "[SKIP] $tag does not look like a DeepSpeed checkpoint"
#       # fi
#     done
#   fi
# fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep19ds35/step2
    
    # 找到所有 epoch_XX 格式的目录（不包括 .pt 后缀的）
    for subdir in $(find "$step2_dir" -maxdepth 1 -type d -name "epoch_*" | grep -v "\.pt$" | sort -V); do
      [ -d "$subdir" ] || continue
      tag=$(basename "$subdir")
      echo "DEBUG: Processing subdir = $subdir"

      # 检查是否需要转换（epoch >= 33 或者没有对应的 .pt 目录）
      epoch_num=$(echo "$tag" | grep -o -E '[0-9]+$')
      if [ -z "$epoch_num" ]; then
        echo "[SKIP] $tag is not in valid format"
        continue
      fi
      
      # 检查对应的 .pt 目录是否存在
      pt_output_dir="${step2_dir}/${tag}.pt"
      if [ -d "$pt_output_dir" ]; then
        echo "[SKIP] $tag already converted -> $pt_output_dir (directory exists)"
        continue
      fi

      # 检查是否为有效的 DeepSpeed checkpoint 目录
      optim_count=$(ls "$subdir"/*_optim_states.pt 2>/dev/null | wc -l)
      if [ "$optim_count" -eq 1 ]; then
        echo "[CONVERT] $tag -> $pt_output_dir"
        python3 ${step2_dir}/zero_to_fp32.py \
          ${step2_dir} ${pt_output_dir} -t "$tag"
      else
        echo "[SKIP] $tag is not a valid single-card checkpoint (found $optim_count optim files)"
      fi
    done
  fi
fi



if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  # Test model, please specify the model you want to test by --checkpoint
  # if [ ${average_checkpoint} == true ]; then
  #   tmp_ckpt_dir=$step2_dir/early_ckpts
  #   mkdir -p $tmp_ckpt_dir

  #   # 创建0.pt 到 10.pt的软链接
  #   for i in $(seq 0 10); do
  #     ln -s $step2_dir/epoch_${i}.pt $tmp_ckpt_dir/epoch_${i}.pt
  #   done

  #   decode_checkpoint=$step2_dir/avg_${average_num}.pt
    
  #   echo "do model average and final checkpoint is $decode_checkpoint"
  #   python wenet/bin/average_model.py \
  #     --dst_model $decode_checkpoint \
  #     --src_path $tmp_ckpt_dir  \
  #     --num ${average_num} \
  #     --val_best
  # fi
  min_epoch=30
  max_epoch=44
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_8ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  # 总的数据集
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other
  # for lang in English-American English-Australian English-British English-Filipino English-Indian French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese; do
  #   grep $lang $step2_dir/$decode_mode/text > $step2_dir/$decode_mode/text_$lang
  #   python tools/compute-wer.py --char=1 --v=1 \
  #       $dev_data_dir/text_tn $step2_dir/$decode_mode/text_$lang > $step2_dir/$decode_mode/wer_$lang
  # done
  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  log "Stage 9: Simple text normalization for the text file and test WER calculation"
  # 如果运行了step2，则无需规范化
    
  # python local/text_normalization.py \
  #     --input $train_data_dir/text \
  #     --output $train_data_dir/text_tn

  # python local/text_normalization.py \
  #     --input /home/maqy23/xjgd/data/test_other/text \
  #     --output /home/maqy23/xjgd/data/test_other/text_tn

  # log "Text normalization done"
  #   # about 1min
  
  python tools/compute-wer.py --char=1 --v=1 \
        $test_data_dir/text_tn /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_14/all/whisper_llm_decode/text > /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_14/all/whisper_llm_decode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: thu and common  Simple text normalization for the text file"
    
    python local/text_normalization.py \
        --input /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text \
        --output /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn

    python local/text_normalization.py \
        --input /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text \
        --output /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn

    log "Text normalization done"
    # about 1min
    tools/make_raw_list.py /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/wav.scp /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn \
      /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list
    tools/make_raw_list.py /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/wav.scp /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn \
      /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  log "Stage 9: Simple text normalization for the text file and test WER calculation"
  # 如果运行了step2，则无需规范化
    
  # python local/text_normalization.py \
  #     --input $train_data_dir/text \
  #     --output $train_data_dir/text_tn

  # python local/text_normalization.py \
  #     --input $dev_data_dir/text \
  #     --output $dev_data_dir/text_tn

  # log "Text normalization done"
  #   # about 1min
  python tools/compute-wer.py --char=1 --v=1 \
        $test_data_dir/text_tn $step2_dir/$decode_mode/text > $step2_dir/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_10_thu/whisper_llm_decode/text > /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_10_thu/whisper_llm_decode//wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_10_common/whisper_llm_decode/text > /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_10_common/whisper_llm_decode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_12/thu/whisper_llm_decode/text > /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_12/thu/whisper_llm_decode/wer
fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  # Test model, please specify the model you want to test by --checkpoint
  # if [ ${average_checkpoint} == true ]; then
  #   tmp_ckpt_dir=$step2_dir/early_ckpts
  #   mkdir -p $tmp_ckpt_dir

  #语音识别部分
  # 总的数据集
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data $dev_data_dir/${data_type}_data.list \
  #   --checkpoint /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2/avg_3_by_12.pt \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/dev 
  # step2_dir=expgdep8ds35/step2
  step2_dir=expgdep8ds35/backup/step2
  max_epoch=26
  echo "$step2_dir, $max_epoch "
  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/dev/text_tn   /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_14/all/whisper_llm_decode/text >  /home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep8ds35/step2_14/all/whisper_llm_decode/wer

fi
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  if [ ${average_checkpoint} == true ]; then
    max_epoch=20
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${max_epoch}_5ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch 0 \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best
  fi

  # # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}_5ds/thu

  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}_5ds/thu/$decode_mode/text > ${step2_dir}_${max_epoch}_5ds/thu/$decode_mode/wer
  #语音识别部分
  # # 总的数据集
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data $dev_data_dir/${data_type}_data.list \
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 10 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}_5ds/all 
  # python tools/compute-wer.py --char=1 --v=1 \
  #       $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer

fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  if [ ${average_checkpoint} == true ]; then
    min_epoch=30
    max_epoch=30
    # decode_checkpoint=$step2_dir/avg_${average_num}_by_${max_epoch}_8ds.pt
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_8ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    # python wenet/bin/average_model.py \
    #   --min_epoch $min_epoch \
    #   --max_epoch $max_epoch \
    #   --dst_model $decode_checkpoint \
    #   --src_path $step2_dir  \
    #   --num ${average_num} \
    #   --val_best
  fi

  # thu部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${min_epoch}_${max_epoch}_8ds/thu

  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${min_epoch}_${max_epoch}_8ds/thu/$decode_mode/text > ${step2_dir}_${min_epoch}_${max_epoch}_8ds/thu/$decode_mode/wer
  #语音识别部分
  # # 总的数据集
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data $dev_data_dir/${data_type}_data.list \
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 10 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}_5ds/all 
  # python tools/compute-wer.py --char=1 --v=1 \
  #       $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer

fi


if [ ${stage} -le 152 ] && [ ${stop_stage} -ge 152 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=2
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi



if [ ${stage} -le 152 ] && [ ${stop_stage} -ge 152 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=2
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data $dev_data_dir/${data_type}_data.list \
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  # python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi


if [ ${stage} -le 152 ] && [ ${stop_stage} -ge 152 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=2
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 154 ] && [ ${stop_stage} -ge 154 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=4
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi


if [ ${stage} -le 156 ] && [ ${stop_stage} -ge 156 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=6
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi


if [ ${stage} -le 158 ] && [ ${stop_stage} -ge 158 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=8
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi


if [ ${stage} -le 1510 ] && [ ${stop_stage} -ge 1510 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=10
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep15ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_15ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # # other部分
  # python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
  #   --modes $decode_mode \
  #   --config $step2_dir/train.yaml \
  #   --data_type $data_type \
  #   --test_data  /home/maqy23/xjgd/data/test_other/data.list\
  #   --checkpoint $decode_checkpoint \
  #   --batch_size 16 \
  #   --dtype bf16 \
  #   --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  # python tools/compute-wer.py --char=1 --v=1 \
  #       /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi



if [ ${stage} -le 68 ] && [ ${stop_stage} -ge 68 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=18
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=40
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 92 ] && [ ${stop_stage} -ge 92 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=42
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 94 ] && [ ${stop_stage} -ge 94 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=44
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 96 ] && [ ${stop_stage} -ge 96 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=46
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 80 ] && [ ${stop_stage} -ge 80 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=30
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 82  ] && [ ${stop_stage} -ge 82 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=32
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 84 ] && [ ${stop_stage} -ge 84 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=34
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 86 ] && [ ${stop_stage} -ge 86 ]; then
  # 注释记得取消，测试模型，选择最好的几个进行评估
  min_epoch=0
  max_epoch=36
  step2_dir=/home/maqy23/MLC-SLM-Baseline-docker/examples/mlcslm/asr/expgdep10ds35/step2

  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$step2_dir/avg_${average_num}_by_${min_epoch}_${max_epoch}_10ds.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --min_epoch $min_epoch \
      --max_epoch $max_epoch \
      --dst_model $decode_checkpoint \
      --src_path $step2_dir  \
      --num ${average_num} \
      --val_best

  fi
  #语音识别部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/all 

  # thu部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/thu_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/thu

  # common部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/common_data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/common

  # other部分
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 1 \
    --modes $decode_mode \
    --config $step2_dir/train.yaml \
    --data_type $data_type \
    --test_data  /home/maqy23/xjgd/data/test_other/data.list\
    --checkpoint $decode_checkpoint \
    --batch_size 16 \
    --dtype bf16 \
    --result_dir ${step2_dir}_${max_epoch}/other

  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/thuyg20/test/text_tn ${step2_dir}_${max_epoch}/thu/$decode_mode/text > ${step2_dir}_${max_epoch}/thu/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/ug_all_1010_yxx/data_split_new/commonvoice/test/text_tn ${step2_dir}_${max_epoch}/common/$decode_mode/text > ${step2_dir}_${max_epoch}/common/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        /home/maqy23/xjgd/data/test_other/text_tn ${step2_dir}_${max_epoch}/other/$decode_mode/text > ${step2_dir}_${max_epoch}/other/$decode_mode/wer

fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
  # step2_dir=expgdep8ds35/backup/step2
  max_epoch=30
  echo "$step2_dir, $max_epoch "
  # python tools/compute-wer.py --char=1 --v=1 \
  #       $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}/all/$decode_mode/text >  ${step2_dir}_${max_epoch}/all/$decode_mode/wer
  python tools/compute-wer.py --char=1 --v=1 \
        $dev_data_dir/text_tn  ${step2_dir}_${max_epoch}_5ds/all/$decode_mode/text >  ${step2_dir}_${max_epoch}_5ds/all/$decode_mode/wer

fi