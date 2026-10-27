"""Mixed-precision and multi-GPU training with MIR (self-contained: synthetic data, no download).

Trains a registration network (VoxelMorph, TransMorph or VFA) with an image-similarity loss plus a smoothness
penalty, in float32, float16 or bfloat16, on one GPU, several GPUs with DistributedDataParallel (``torchrun``) or
DataParallel, or the CPU. The saved checkpoint has no ``module.`` prefix and loads into a plain model with
``strict=True``.

Examples::

    # one GPU (or CPU), float16 mixed precision
    python train_multi_gpu_amp.py --precision fp16

    # 4 GPUs, one process per GPU (recommended)
    torchrun --nproc_per_node=4 train_multi_gpu_amp.py --precision bf16 --model transmorph

    # all visible GPUs from one process
    python train_multi_gpu_amp.py --parallel dp --precision fp16

    # CPU-only distributed smoke test (gloo backend)
    torchrun --nproc_per_node=2 train_multi_gpu_amp.py --device cpu --precision bf16 --iters 4

``torchrun`` is the launcher of the Python environment it belongs to; if MIR is installed in another interpreter, use
``python3.8 -m torch.distributed.run --nproc_per_node=N ...`` instead.
"""

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

import MIR.models as M
import MIR.models.configs_TransMorph as configs_TransMorph
import MIR.models.configs_VFA as configs_VFA
import MIR.models.configs_VoxelMorph as configs_VoxelMorph
from MIR.deformation_regularizer import Grad3d
from MIR.image_similarity import NCC_vxm
from MIR.utils import distributed as dist_utils
from MIR.utils import precision as prec


class SyntheticPairs(Dataset):
    """Random smooth volume pairs where the fixed image is a smoothly warped copy of the moving image."""

    def __init__(self, n, size, seed=0):
        self.n, self.size, self.seed = n, size, seed
        self.warp = M.SpatialTransformer(size)

    def __len__(self):
        return self.n

    def _smooth(self, x, k=5, reps=3):
        for _ in range(reps):
            x = torch.nn.functional.avg_pool3d(torch.nn.functional.pad(x, (k // 2,) * 6, mode="replicate"), k, 1)
        return x

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed + i)
        mov = self._smooth(torch.rand(1, 1, *self.size, generator=g))
        mov = (mov - mov.min()) / (mov.max() - mov.min() + 1e-6)
        disp = self._smooth(torch.randn(1, 3, *self.size, generator=g), k=7, reps=4) * 40
        fix = self.warp(mov, disp)
        return mov[0], fix[0]


def build_model(name, size):
    if name == "vxm":
        cfg = configs_VoxelMorph.get_VXM_default_config()
        cfg.img_size = size
        return M.VxmDense(cfg)
    if name == "transmorph":
        cfg = configs_TransMorph.get_3DTransMorph3Lvl_config()
        cfg.img_size, cfg.window_size, cfg.out_chan = size, tuple(s // 32 for s in size), 3
        return M.TransMorph(cfg, SVF=False)
    if name == "vfa":
        cfg = configs_VFA.get_VFA_default_config()
        cfg.img_size = size
        return M.VFA(cfg, device="cpu")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="vxm", choices=["vxm", "transmorph", "vfa"])
    ap.add_argument("--precision", default="fp32", choices=list(prec.PRECISIONS))
    ap.add_argument("--parallel", default="auto", choices=["auto", "ddp", "dp", "none"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--size", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--batch-size", type=int, default=1, help="per process (DDP) or total (DataParallel)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reg-weight", type=float, default=1.0)
    ap.add_argument("--out", default="multi_gpu_amp_checkpoint.pth.tar")
    args = ap.parse_args()
    size = tuple(args.size)

    device = dist_utils.init_distributed(device_type=args.device)  # nccl on GPUs, gloo on the CPU
    device_type = device.type
    prec.check_precision(args.precision, device_type)
    torch.manual_seed(0)

    model = build_model(args.model, size).to(device)
    warp = M.SpatialTransformer(size).to(device)
    model = dist_utils.wrap_model(model, mode=args.parallel, device=device)
    sim, reg = NCC_vxm(), Grad3d(penalty="l2")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = prec.grad_scaler(args.precision, device_type)

    data = SyntheticPairs(n=args.iters * args.batch_size * dist_utils.get_world_size(), size=size)
    sampler = dist_utils.distributed_sampler(data)
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None)
    if sampler is not None:
        sampler.set_epoch(0)

    model.train()
    for step, (mov, fix) in enumerate(loader):
        mov, fix = mov.to(device), fix.to(device)
        optimizer.zero_grad(set_to_none=True)
        with prec.autocast(args.precision, device_type):
            flow = model((mov, fix))
            loss = sim(fix, warp(mov, flow)) + args.reg_weight * reg(flow, None)  # losses compute in float32
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mean_loss = dist_utils.all_reduce_mean(loss.detach())
        if dist_utils.is_main_process() and (step % 5 == 0 or step == len(loader) - 1):
            print(f"step {step:3d}  loss {float(mean_loss):.4f}  ({args.precision}, world size {dist_utils.get_world_size()})")

    if dist_utils.is_main_process():
        torch.save({"state_dict": dist_utils.state_dict_for_saving(model), "args": vars(args)}, args.out)
        plain = build_model(args.model, size)
        plain.load_state_dict(torch.load(args.out, map_location="cpu")["state_dict"], strict=True)
        print(f"saved {args.out} (loads into a plain {type(plain).__name__} with strict=True)")
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
