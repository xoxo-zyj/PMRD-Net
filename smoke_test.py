# -*- coding: utf-8 -*-
"""Shape and loss smoke test. Fallback is allowed only in this test."""

import torch

from future_process import MSDSPDDCompositeLoss, MSDSPDDDehazer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSDSPDDDehazer(
        enable_end_to_end=True,
        weight_path="./future_process/efficientvit_b1_r256.pt",
        use_mamba=torch.cuda.is_available(),
        allow_backbone_fallback=True,
    ).to(device)
    model.train()

    hazy = torch.rand(1, 3, 64, 64, device=device)
    clean = torch.rand(1, 3, 64, 64, device=device)
    outputs = model(hazy)

    criterion = MSDSPDDCompositeLoss().to(device)
    loss, terms = criterion(
        outputs,
        hazy,
        clean,
        model,
        route_regularization_factor=1.0,
    )
    loss.backward()

    final_out, tau_ode, base, indicator, routes, t0, A0 = outputs
    print("device:", device)
    print("final:", tuple(final_out.shape))
    print("base:", tuple(base.shape))
    print("t1:", tuple(tau_ode.shape))
    print("indicator:", tuple(indicator.shape))
    print("five routes:", len(routes))
    print("backbone fallback:", model.backbone_is_fallback)
    print("Mamba enabled:", model.mamba_enabled)
    print("loss:", float(loss.detach()))
    print("loss terms:", {k: float(v) for k, v in terms.items()})
    for scale, weight in model.last_route_weights.items():
        print(scale, tuple(weight.shape), float(weight.sum(dim=1).mean()))

    assert final_out.shape == hazy.shape
    assert base.shape == hazy.shape
    assert tau_ode.shape == hazy[:, :1].shape
    assert len(routes) == 5
    assert torch.isfinite(loss)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
