#!/usr/bin/env python3
import argparse
import torch
import joblib
from torch import nn
from braindecode.models import Deep4Net
from sklearn.pipeline import Pipeline

def parse_args():
    p = argparse.ArgumentParser(
        description="Trace a TL‐trained Deep4Net to TorchScript + export joblib wrapper"
    )
    p.add_argument("in_pth",     help="Path to your tl_pooled_model.pth (state dict)")
    p.add_argument("out_joblib", help="Path to write the final .joblib (no extension)")
    p.add_argument("--n_chans",   type=int,   required=True)
    p.add_argument("--n_outputs", type=int,   required=True)
    p.add_argument("--n_times",   type=int,   required=True,
                   help="Number of time samples your model expects")
    p.add_argument("--sfreq",     type=float, required=True,
                   help="Sampling frequency used when training")
    p.add_argument("--device",    default="cpu")
    return p.parse_args()

# 1) A small Deep4Net subclass that makes chs_info readable without any extra code
class TraceNet(Deep4Net):
    @property
    def chs_info(self):
        # Braindecode needs this to script properly
        return [(self.n_chans, self.n_times)]

# 2) A tiny nn.Module that holds your backbone+head
class Traceable(nn.Module):
    def __init__(self, ckpt, n_chans, n_outputs, n_times, sfreq, device):
        super().__init__()
        # instantiate Deep4Net exactly as in your TLModel's shared backbone + one head
        self.net = TraceNet(
            n_chans   = n_chans,
            n_outputs = n_outputs,
            n_times   = n_times,
            sfreq     = sfreq
        ).to(device)
        # load the state dict you saved offline
        state = torch.load(ckpt, map_location=device)
        self.net.load_state_dict(state, strict=False)

    def forward(self, x):
        # forward signature: (batch, chans, times)
        return self.net(x)

def main():
    args = parse_args()
    dev = torch.device(args.device)

    # 1) Trace to TorchScript
    model = Traceable(
        ckpt     = args.in_pth,
        n_chans  = args.n_chans,
        n_outputs= args.n_outputs,
        n_times  = args.n_times,
        sfreq    = args.sfreq,
        device   = dev,
    ).eval()

    # dummy input to match (1, chans, times)
    dummy = torch.zeros(1, args.n_chans, args.n_times, device=dev)
    ts = torch.jit.trace(model, dummy)

    # save the TorchScript artifact
    ts_path = args.out_joblib + ".pt"
    ts.save(ts_path)

    # 2) Build the sklearn pipeline with your recorder's TorchScriptWrapper
    #    (this class lives in your recorder codebase under lib/*.py)
    from torch_wrapper import TorchScriptWrapper

    pipe = Pipeline([
        ("tsmodel", TorchScriptWrapper(ts_path, device=args.device)),
    ])

    # 3) Dump the pipeline to joblib
    joblib.dump(pipe, args.out_joblib + ".joblib")
    print(f"✅ Export complete:\n"
          f"  - TorchScript → {ts_path}\n"
          f"  - Joblib pipeline → {args.out_joblib}.joblib")

if __name__ == "__main__":
    main()
