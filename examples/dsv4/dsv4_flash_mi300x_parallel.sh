#!/usr/bin/env bash
# Parallel layout for DeepSeek-V4-Flash full model on 2×8 MI300X/MI308X (16 GPUs).
#
# Matches miles/scripts/amd/run_deepseek_v4.py _get_parallel_config() for total_gpus==16.

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

TP=4
PP=4
CP=1
EP=4
ETP=1

DECODER_FIRST_PP_LAYERS=11
DECODER_LAST_PP_LAYERS=10
