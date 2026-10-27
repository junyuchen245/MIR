# Mixed precision and multi-GPU training

A self-contained example (synthetic data, no download) of training a MIR registration network in float32, float16
or bfloat16 on one GPU, several GPUs (`DistributedDataParallel` via `torchrun`, or `DataParallel`) or the CPU.
See the [user guide](https://junyuchen.me/MIR/user_guide/precision_multigpu.html) for the details.

```bash
python train_multi_gpu_amp.py --precision fp16                                   # one GPU
torchrun --nproc_per_node=4 train_multi_gpu_amp.py --precision bf16 --model transmorph   # 4 GPUs, DDP
python train_multi_gpu_amp.py --parallel dp --precision fp16                     # all GPUs, one process
torchrun --nproc_per_node=2 train_multi_gpu_amp.py --device cpu --iters 4        # CPU smoke test (gloo)
```

The checkpoint is saved without the `module.` prefix that wrapped models add, so it loads into a plain model with
`load_state_dict(..., strict=True)`, like the published MIR weights.
