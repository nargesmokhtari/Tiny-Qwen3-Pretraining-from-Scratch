# Tiny-Qwen3-Pretraining-from-Scratch
Pretraining a small language model with the Qwen3 architecture from scratch on a single GPU.
The model has 113M parameters and trains on 500M tokens of FineWeb-Edu. 
From scratch: random initialization. Only the Qwen3 tokenizer is reused, not its weights.
Streaming data pipeline: tokenizes FineWeb-Edu on the fly into flat uint32 binary files, without downloading the full dataset.
Memory-mapped loading: np.memmap reads training data without loading it into RAM.
Mixed precision (fp16): training uses torch.autocast with GradScaler.
