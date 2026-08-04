###############################################################################
# Qwen3-8B arguments shared by the Lumen and the stock-Megatron+TE launchers.
#
# Sourced, not executed. Populates the COMMON_ARGS array from variables the
# caller has already set: NUM_LAYERS MBS GBS SEQ_LEN TRAIN_STEPS SEED
# TOKENIZER_PATH TRAIN_JSONL VALID_JSONL SPLIT EVAL_ITERS EVAL_INTERVAL.
#
# The two launchers exist to compare quantization stacks, so every
# hyperparameter that is not the thing under test lives here — duplicating this
# list per launcher is how the two arms silently drift apart and produce a
# comparison that means nothing.
###############################################################################

COMMON_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --qk-layernorm
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --no-position-embedding
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --train-iters "${TRAIN_STEPS}"
    --lr 1.0e-5 --min-lr 0.0
    --lr-decay-style cosine
    --lr-warmup-iters 2
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
    --bf16
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --overlap-grad-reduce
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_PATH}"
    --train-data-path "${TRAIN_JSONL}"
    --valid-data-path "${VALID_JSONL}"
    --test-data-path "${VALID_JSONL}"
    --split "${SPLIT}"
    --seed "${SEED}"
    --eval-iters "${EVAL_ITERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --save-interval 1000000
    --log-interval 1
)
