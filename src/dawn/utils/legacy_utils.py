
import os
import cv2
import numpy as np
import random
import torch
import skimage.morphology as ski_morph
from scipy.ndimage import distance_transform_edt
from skimage import measure
from termcolor import colored
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_accuracy(pred, gt, radius, return_distance=False):
    """ compute detection accuracy: recall, precision, F1 """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = ski_morph.label(pred)
    pred_regions = measure.regionprops(pred_labeled)
    pred_points = []
    for region in pred_regions:
        pred_points.append(region.centroid)
    pred_points = np.array(pred_points)
    Np = pred_points.shape[0]

    gt_points = np.argwhere(gt == 255)
    Ng = gt_points.shape[0]
    TP = 0.0
    FN = 0.0
    d_list = []   # the distances between true locations and TP detections
    for i in range(Ng):   # for each gt point, find the nearest pred point
        if np.size(pred_points) == 0:
            FN += 1
            continue
        gt_point = gt_points[i, :]
        dist = np.linalg.norm(pred_points - gt_point, axis=1)
        if np.min(dist) < radius:  # the nearest pred point is in the radius of the gt point
            pred_idx = np.argmin(dist)
            pred_points = np.delete(pred_points, pred_idx, axis=0)   # delete the TP
            TP += 1
            d_list.append(np.min(dist))
        else:  # the nearest pred point is not in the radius
            FN += 1

    FP = Np - TP

    if return_distance:
        return TP, FP, FN, d_list
    else:
        return TP, FP, FN


def split_forward(model, input, size, overlap, outchannel=2):
    '''
    split the input image for forward process
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch_ = model(input_var)
                output_patch = output_patch_["det_prob"]
                seg_prob = output_patch_["np"]
                seg_prob = F.softmax(seg_prob,dim=1)[:,1:,:,:]

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output, seg_prob.cpu()


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_txt(results, filename, mode='w'):
    """ Save the result of losses and F1 scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num-1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num-1]))


def save_results(header, all_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for value in vals:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    elif data_format == "NHW":
        x = x[:, crop_t:-crop_b, crop_l:-crop_r]
    elif data_format == "HW":
        x = x[crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


    
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    
    # assert (
    #     y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    # ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.shape
    y_shape = y.shape
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    elif data_format == "NHW":
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    elif data_format == "HW":
        crop_shape = (x_shape[0] - y_shape[0], x_shape[1] - y_shape[1])
    return crop_op(x, crop_shape, data_format)


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[2] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[3] - crop_shape[1]) * 0.5)
        x = x[:,:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x

def generate_training_np(det_prob):

    det_map = torch.logical_and(det_prob > 0.35, det_prob<0.9).cpu().numpy()
    loc_map = torch.logical_and(det_prob > 0.7, det_prob<0.9).cpu().numpy().astype(np.uint8)
    # print(loc_map.shape)


    dist_map_ = []
    for _ in range(loc_map.shape[0]):
        loc_map_ = loc_map[_,...]
        # print("sum", loc_map_.sum())

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(loc_map_, connectivity=8)
        
        poi_map = np.zeros_like(loc_map_)
        for i in range(centroids.shape[0]):
            poi_map[int(centroids[i,1]),int(centroids[i,0])] = 1
        poi_map = poi_map.astype(np.float32)

        poi_map_trans_fb = 255 - poi_map*255
        dist_map = distance_transform_edt(poi_map_trans_fb)+1
        # plt.imshow(dist_map)
        # plt.show()
        # plt.savefig("distance.png")

        # print("dist", dist_map)
        dist_map = dist_map * [dist_map < 22]
        # print("dist map", dist_map.shape)
        dist_map_.extend(dist_map)

    return np.array(dist_map_)

