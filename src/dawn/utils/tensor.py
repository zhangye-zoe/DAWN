from __future__ import annotations
import numpy as np
import torch


def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5); w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        return x[h0:h0+crop_shape[0], w0:w0+crop_shape[1]]
    h0 = int((orig_shape[2] - crop_shape[0]) * 0.5); w0 = int((orig_shape[3] - crop_shape[1]) * 0.5)
    return x[:, :, h0:h0+crop_shape[0], w0:w0+crop_shape[1]]


def crop_op(x, cropping, data_format="NCHW"):
    if isinstance(cropping, int): cropping = [cropping, cropping]
    if data_format == "NCHW":
        return x[:, :, cropping[0]//2:x.shape[2]-cropping[0]+cropping[0]//2, cropping[1]//2:x.shape[3]-cropping[1]+cropping[1]//2]
    return x[:, cropping[0]//2:x.shape[1]-cropping[0]+cropping[0]//2, cropping[1]//2:x.shape[2]-cropping[1]+cropping[1]//2, :]


def crop_to_shape(x, y, data_format="NCHW"):
    if data_format == "NCHW":
        x_shape = x.shape[2:]; y_shape = y.shape[2:]
    else:
        x_shape = x.shape[1:3]; y_shape = y.shape[1:3]
    cropping = [x_shape[0]-y_shape[0], x_shape[1]-y_shape[1]]
    if cropping[0] == 0 and cropping[1] == 0: return x
    return crop_op(x, cropping, data_format)
