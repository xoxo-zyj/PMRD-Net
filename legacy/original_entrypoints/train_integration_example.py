# -*- coding: utf-8 -*-
"""Minimal integration fragment for the existing training loop."""

from future_process import MSDSPDDCompositeLoss, MSDSPDDDehazer

model = MSDSPDDDehazer(
    enable_end_to_end=True,
    weight_path="./future_process/efficientvit_b1_r256.pt",
)
criterion = MSDSPDDCompositeLoss()

# In each training iteration:
# outputs = model(hazy)
# loss, loss_dict = criterion(
#     outputs,
#     hazy,
#     clean,
#     model,
#     route_regularization_factor=route_factor,
# )
# optimizer.zero_grad(set_to_none=True)
# loss.backward()
# optimizer.step()
#
# Temperature example:
# model.set_route_temperature(1.5 - 0.5 * progress)  # 1.5 -> 1.0
# Route survival factor example:
# route_factor = max(0.0, 1.0 - progress / 0.20)     # only first 20%
