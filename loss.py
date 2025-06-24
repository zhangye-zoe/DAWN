import torch
import torch.nn.functional as F

def calculate_loss(true, pred, weight):
    loss_item = ["encode loss", "fuse prob loss"]#, "semi_sup loss"]
    loss = 0
    for item in loss_item:
        if item == "encode loss":
            enc_loss = mse_loss(true["det_encoding"], pred["seg_encoding"])
            loss += enc_loss*weight["encode loss"]

        elif item == "fuse prob loss":
            seg_prob = pred["seg_np"].permute(0,2,3,1).contiguous()
            xent_loss = xentropy_loss(true["fused_prob"], seg_prob)
            loss += xent_loss*weight["fuse prob loss"]


        elif item == "semi_sup loss":
            # pass
            
            if true["true_np"].sum() == 0:
                return loss
            else:
                true_np = true["true_np"].type(torch.int64).permute(0,2,3,1).contiguous()
                # print(true_np.type, true_np.shape)
                true_np_onehot = (F.one_hot(true_np.squeeze(-1), num_classes=2)).type(torch.float32)
                true_hv = true["true_hv"].type(torch.float32).permute(0,2,3,1).contiguous()

                pred_np = pred["seg_np"].permute(0,2,3,1).contiguous()
                pred_hv = pred["seg_hv"].permute(0,2,3,1).contiguous()

                # print(true_np_onehot.shape, pred_np.shape)

                np_xent = xentropy_loss_(true_np_onehot, pred_np)
                np_dice = dice_loss(true_np_onehot, pred_np)
                hv_mse = mse_loss(true_hv, pred_hv)
                hv_msge = msge_loss(true_hv, pred_hv, true_np_onehot[...,1])

                semi_loss = np_xent + np_dice + hv_mse + hv_msge

                loss += semi_loss*weight["semi_sup loss"]
                

    return loss


def mse_loss(true, pred,):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    # epsilon =  10e-8
    # pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    b_true = torch.logical_not(true)
    true =  torch.concat((b_true,true), dim=-1)
    # scale preds so that the class probs of each sample sum to 1
    # pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    # print("pred", pred.shape)
    # print("true", true.shape)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


def xentropy_loss_(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    # print("pred", pred.shape)
    # print("true", true.shape)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss

def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss

def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss


