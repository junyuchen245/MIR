#!/usr/bin/env python3.8
"""Export the trained 2D HyperVFA / HyperVFASPR checkpoints to ONNX for onnxruntime-web.

Writes <out>/<name>.onnx (fp32) and <out>/<name>_fp16w.onnx (fp16 initializers + Cast, fp32 compute),
checks numerical parity with PyTorch and with onnxruntime, and prints sizes and CPU latency.
"""
import argparse, os, time, json, numpy as np, torch, onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from train_web_hypervfa import build, HYPER_MAP

class Wrap(torch.nn.Module):
    def __init__(self, m, spr): super().__init__(); self.m = m; self.spr = spr
    def forward(self, mov, fix, h):
        if self.spr:
            flow, w = self.m((mov, fix), h); return flow, w
        return self.m((mov, fix), h)

def fp16_weights(model_path, out_path, min_elems=1024):
    """Store large float initializers as float16 and insert Cast->float32 so compute stays fp32."""
    m = onnx.load(model_path); g = m.graph
    new_inits, casts, renamed = [], [], {}
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= min_elems:
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            new_inits.append(h); renamed[init.name] = init.name + "_fp16"
            casts.append(helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
        else:
            new_inits.append(init)
    del g.initializer[:]; g.initializer.extend(new_inits)
    # casts must come before first use: prepend them
    nodes = list(g.node); del g.node[:]; g.node.extend(casts + nodes)
    onnx.checker.check_model(m); onnx.save(m, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--model', choices=['spr', 'plain'], required=True)
    ap.add_argument('--out', default='web/models'); ap.add_argument('--name', default=None)
    ap.add_argument('--data_dir', default='data', help='folder with test.npz (used for the parity check)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    name = a.name or f'hypervfa_{a.model}_2d'
    ck = torch.load(a.ckpt, map_location='cpu'); cfg = ck['cfg']
    model, _ = build(a.model, tuple(cfg['img_size']), cfg['start_channels'], cfg['max_channels'], cfg['nb_hyp_units'], beta0=cfg.get('initialize', 0.1), device='cpu')
    model.load_state_dict(ck['state_dict']); model.eval()
    H, W = cfg['img_size']; nh = 2 if a.model == 'spr' else 1
    d = np.load(os.path.join(a.data_dir, 'test.npz')); img = d['img']; img = img / img.reshape(len(img), -1).max(1)[:, None, None]
    mov = torch.from_numpy(img[2:3]).float().unsqueeze(1); fix = torch.from_numpy(img[7:8]).float().unsqueeze(1)
    h = torch.full((1, nh), 0.5)
    with torch.no_grad(): ref = Wrap(model, a.model == 'spr')(mov, fix, h)
    ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    p32 = os.path.join(a.out, name + '.onnx')
    outputs = ['flow', 'weights'] if a.model == 'spr' else ['flow']
    torch.onnx.export(Wrap(model, a.model == 'spr'), (mov, fix, h), p32, input_names=['moving', 'fixed', 'hyper'],
                      output_names=outputs, opset_version=16, do_constant_folding=True)
    onnx.checker.check_model(p32)
    p16 = os.path.join(a.out, name + '_fp16w.onnx'); fp16_weights(p32, p16)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    feed = {'moving': mov.numpy(), 'fixed': fix.numpy(), 'hyper': h.numpy()}
    for p in (p32, p16):
        s = ort.InferenceSession(p, so, providers=['CPUExecutionProvider']); o = s.run(None, feed)
        t = time.time(); [s.run(None, feed) for _ in range(5)]; dt = (time.time() - t) / 5
        diffs = [float(np.abs(x - y).max()) for x, y in zip(o, ref)]
        print(f'{os.path.basename(p)}: {os.path.getsize(p)/1e6:.2f} MB, 1-thread {dt*1000:.0f} ms, max|diff| vs torch {diffs} (flow absmax {np.abs(ref[0]).max():.2f})')
    meta = {'name': name, 'model': a.model, 'img_size': [H, W], 'n_hyper': nh, 'hyper_map': ck.get('hyper_map', dict(HYPER_MAP)),
            'params': int(sum(p.numel() for p in model.parameters())), 'iters': ck.get('it'), 'val': {str(k): v for k, v in ck.get('val', {}).items()},
            'cfg': cfg}
    json.dump(meta, open(os.path.join(a.out, name + '.json'), 'w'), indent=1); print('meta', meta['val'])

if __name__ == '__main__':
    main()
