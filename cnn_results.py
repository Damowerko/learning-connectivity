import pytorch_lightning as pl
from pathlib import Path
from hdf5_dataset_utils import kernelized_config_img, cnn_image_parameters, subs_to_pos, pos_to_subs
from math import ceil, sqrt
from cnn import BetaVAEModel, load_model_for_eval
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import numpy as np
from skimage.filters.thresholding import threshold_local
from skimage.filters import gaussian
from connectivity_maximization import circle_points
from network_planner.connectivity_optimization import ConnectivityOpt as ConnOpt
from feasibility import connect_graph
import torch
from hdf5_dataset_utils import ConnectivityDataset
import h5py


def compute_peaks(image, threshold_val=80, blur_sigma=2, region_size=11):

    # remove noise in image
    blurred_img = gaussian(image, sigma=blur_sigma)

    # only keep the max value in a local region
    thresh_fcn = lambda a : max(a.max(), 0.01)
    thresh_mask = threshold_local(blurred_img, region_size, method='generic', param=thresh_fcn)
    peaks = np.argwhere(blurred_img >= thresh_mask)

    # only pixels above a threshold value should be considered peaks
    peaks = peaks[image[peaks[:,0], peaks[:,1]] > threshold_val]

    # form list of unique peaks, averaging any that are near each other
    out_peaks = np.zeros((0,2))
    used = np.zeros((peaks.shape[0],), dtype=bool)
    for i in range(peaks.shape[0]-1):
        if used[i]:
            continue
        near_peaks = np.where(np.linalg.norm(peaks[i+1:] - peaks[i], axis=1) < 4)[0].tolist()
        if len(near_peaks) == 0:
            out_peaks = np.vstack((out_peaks, peaks[i]))
            used[i] = True
        else:
            import pdb;pdb.set_trace()
            near_peaks.append(i)
            out_peaks = np.vstack((out_peaks, np.mean(peaks[near_peaks], axis=0)))
            used[near_peaks] = True

    return peaks


def connectivity_from_image(task_config, out_img, p):
    peaks = compute_peaks(out_img)
    comm_config = np.zeros_like(peaks)
    for i, peak in enumerate(peaks):
        comm_config[i,:] = subs_to_pos(p['meters_per_pixel'], p['img_size'][0], peak)
    connectivity = ConnOpt.connectivity(p['channel_model'], task_config, comm_config)
    return connectivity, comm_config


def connectivity_from_config(task_config, p, viz=False):
    comm_config = connect_graph(task_config, p['comm_range'])
    opt = ConnOpt(p['channel_model'], task_config, comm_config)
    conn = opt.maximize_connectivity(viz=viz)
    return conn, opt.get_comm_config()


def segment_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file['test']['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    input_image = hdf5_file['test']['task_img'][idx,...]
    output_image = hdf5_file['test']['comm_img'][idx,...]
    model_image = model.inference(input_image)

    peaks = compute_peaks(model_image)

    fig, ax = plt.subplots()
    if args.isolate:
        ax.imshow(model_image)
    else:
        ax.imshow(np.maximum(model_image, input_image))
    ax.axis('off')
    ax.plot(peaks[:,1], peaks[:,0], 'ro')
    ax.invert_yaxis()
    plt.show()


def line_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    params = cnn_image_parameters()
    img_extents = params['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1])

    start_config = np.asarray([[0, 20], [0, -20]])
    step = 2*np.asarray([[0, 1],[0, -1]])
    for i in range(20):
        task_config = start_config + i*step
        img = kernelized_config_img(task_config, params)
        out = model.inference(img)

        cnn_conn, cnn_config = connectivity_from_image(task_config, out, params)
        opt_conn, opt_config = connectivity_from_config(task_config, params)

        fig, ax = plt.subplots()

        ax.imshow(np.maximum(out, img), extent=img_extents)
        ax.plot(task_config[:,1], task_config[:,0], 'ro', label='task')
        ax.plot(opt_config[:,1], opt_config[:,0], 'rx', label='comm. opt.', ms=9, mew=3)
        ax.plot(cnn_config[:,1], cnn_config[:,0], 'bx', label='comm. CNN', ms=9, mew=3)

        ax.invert_yaxis()
        ax.set_yticks(np.arange(-80, 80, 20))
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(loc='best', fontsize=14)
        ax.set_title(f'opt. = {opt_conn:.3f}, cnn = {cnn_conn:.3f}', fontsize=18)

        plt.tight_layout()

        if args.save:
            filename = f'line_{i:02d}_{model_file.stem}.png'
            plt.savefig(filename, dpi=150)
            print(f'saved image {filename}')
        else:
            plt.show()


def circle_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    params = cnn_image_parameters()
    img_extents = params['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1])

    task_agents = 4
    min_rad = (params['comm_range']+2.0) / (2.0 * np.sin(np.pi/task_agents))
    rads = np.linspace(min_rad, 60, 15)
    for i, rad in enumerate(rads):
        task_config = circle_points(rad, task_agents)
        img = kernelized_config_img(task_config, params)
        out = model.inference(img)

        cnn_conn, cnn_config = connectivity_from_image(task_config, out, params)
        opt_conn, opt_config = connectivity_from_config(task_config, params, viz=True)
        print(f'it {i+1:2d}: rad = {rad:.1f}m, cnn # = {cnn_config.shape[0]}, '
              f'cnn conn = {cnn_conn:.4f}, opt # = {opt_config.shape[0]}, '
              f'opt conn = {opt_conn:.4f}')

        fig, ax = plt.subplots()

        ax.imshow(np.maximum(out, img), extent=img_extents)
        ax.plot(task_config[:,1], task_config[:,0], 'ro', label='task')
        ax.plot(opt_config[:,1], opt_config[:,0], 'rx', label=f'opt ({opt_config.shape[0]})', ms=9, mew=3)
        ax.plot(cnn_config[:,1], cnn_config[:,0], 'bx', label=f'CNN ({cnn_config.shape[0]})', ms=9, mew=3)

        ax.invert_yaxis()
        ax.set_yticks(np.arange(-80, 80, 20))
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(loc='best', fontsize=14)
        ax.set_title(f'opt. = {opt_conn:.3f}, cnn = {cnn_conn:.3f}', fontsize=18)

        plt.tight_layout()

        if args.save:
            filename = f'line_{i:02d}_{model_file.stem}.png'
            plt.savefig(filename, dpi=150)
            print(f'saved image {filename}')
        else:
            plt.show()


def extrema_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    dataset = ConnectivityDataset(dataset_file, train=False)

    if args.best:
        extrema_test = lambda l : l < extreme_loss
        extreme_loss = np.Inf
        print(f'seeking the best performing sample')
    else:
        extrema_test = lambda l : l > extreme_loss
        extreme_loss = 0.0
        print(f'seeking the worst performing sample')
    with torch.no_grad():
        print(f'looping through {len(dataset)} test samples in {dataset_file}')
        for i in range(len(dataset)):
            print(f'\rprocessing sample {i+1} of {len(dataset)}\r', end="")
            batch = [torch.unsqueeze(ten, 0) for ten in dataset[i]]
            loss = model.validation_step(batch, None)
            if extrema_test(loss):
                extreme_idx = i
                extreme_loss = loss

    hdf5_file = h5py.File(dataset_file, mode='r')
    input_image = hdf5_file['test']['task_img'][extreme_idx,...]
    output_image = hdf5_file['test']['comm_img'][extreme_idx,...]
    model_image = model.inference(input_image)

    print(f'worst sample is {extreme_idx}{20*" "}')
    ax = plt.subplot(1,3,1)
    ax.imshow(input_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('input')
    ax = plt.subplot(1,3,2)
    ax.imshow(output_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('target')
    ax = plt.subplot(1,3,3)
    ax.imshow(model_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('output')
    plt.tight_layout()
    plt.show()

    hdf5_file.close()


def connectivity_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    mode = 'train' if args.train else 'test'

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file[mode]['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    input_image = hdf5_file[mode]['task_img'][idx,...]
    opt_conn = hdf5_file[mode]['connectivity'][idx]
    task_config = hdf5_file[mode]['task_config'][idx,...]
    comm_config = hdf5_file[mode]['comm_config'][idx,...]
    model_image = model.inference(input_image)

    p = cnn_image_parameters()
    img_extents = p['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1])

    cnn_conn, cnn_config = connectivity_from_image(task_config, model_image, p)

    ax = plt.subplot()
    ax.imshow(np.maximum(input_image, model_image), extent=img_extents)
    ax.plot(task_config[:,1], task_config[:,0], 'ro', label='task')
    ax.plot(comm_config[:,1], comm_config[:,0], 'rx', label='comm. opt.', ms=9, mew=3)
    ax.plot(cnn_config[:,1], cnn_config[:,0], 'bx', label='comm. CNN', ms=9, mew=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(-80, 80, 20))
    ax.legend(loc='best', fontsize=14)
    ax.set_title(f'{idx}: opt. = {opt_conn:.3f}, cnn = {cnn_conn:.3f}', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    if not args.save:
        print(f'showing sample {idx} from {dataset_file.name}')
        plt.show()
    else:
        filename = str(idx) + '_' + dataset_file.stem + '.png'
        plt.savefig(filename, dpi=150)
        print(f'saved image {filename}')

    hdf5_file.close()


def stats_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')

    mode = 'train' if args.train else 'test'
    dataset_len = hdf5_file[mode]['task_img'].shape[0]

    opt_connectivity = hdf5_file[mode]['connectivity'][:]
    cnn_connectivity = np.zeros_like(opt_connectivity)

    p = cnn_image_parameters()

    for i in range(dataset_len):
        print(f'\rprocessing sample {i+1} of {dataset_len}\r', end="")
        model_image = model.inference(hdf5_file[mode]['task_img'][i,...])
        task_config = hdf5_file[mode]['task_config'][i,...]
        cnn_connectivity[i], _ = connectivity_from_image(task_config, model_image, p)
    print(f'processed {dataset_len} test samples in {dataset_file.name}')

    eps = 1e-10
    opt_feasible = opt_connectivity > eps
    cnn_feasible = cnn_connectivity > eps
    both_feasible = opt_feasible & cnn_feasible

    # sometimes the CNN performs better
    better = np.where(cnn_connectivity > opt_connectivity)[0]
    better = better[cnn_feasible[better]]

    absolute_error = opt_connectivity[both_feasible] - cnn_connectivity[both_feasible]
    percent_error = absolute_error / opt_connectivity[both_feasible]

    print(f'{np.sum(opt_feasible)}/{dataset_len} feasible with optimization')
    print(f'{np.sum(cnn_feasible)}/{dataset_len} feasible with CNN')
    print(f'{np.sum(cnn_feasible & ~opt_feasible)} cases where only the CNN was feasible')
    if better.shape[0] > 50:
        print(f'{better.shape[0]} samples where the CNN outperformed the optimization')
    else:
        print(f'samples where the CNN performed better: {", ".join(map(str, better))}')
    print(f'absolute error: mean = {100*np.mean(absolute_error):.2f}, std = {100*np.std(absolute_error):.2f}')
    print(f'percent error:  mean = {100*np.mean(percent_error):.2f}, std = {100*np.std(percent_error):.2f}')


def variation_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file['test']['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    x = hdf5_file['test']['task_img'][idx,...]

    with torch.no_grad():
        x1 = torch.from_numpy(np.expand_dims(x / 255.0, axis=(0,1))).float()
        y_hat1, _, _ = model(x1)
        y_out1 = torch.clamp(255*y_hat1, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()

        x2 = torch.from_numpy(np.expand_dims(x / 255.0, axis=(0,1))).float()
        y_hat2, _, _ = model(x2)
        y_out2 = torch.clamp(255*y_hat1, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()

    diff_x = x1 - x2
    diff_y_hat = y_hat1 - y_hat2
    diff_y_out = y_out1 - y_out2

    print(f'diff_x: min = {diff_x.min()}, max = {diff_x.max()}')
    print(f'diff_y_hat: min = {diff_y_hat.min()}, max = {diff_y_hat.max()}')
    print(f'diff_y_out: min = {diff_y_out.min()}, max = {diff_y_out.max()}')

    # ax = plt.subplot(1,3,1)
    # ax.imshow(model_image1, vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('model_image1')
    # ax = plt.subplot(1,3,2)
    # ax.imshow(model_image2, vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('model_image2')
    # ax = plt.subplot(1,3,3)
    # ax.imshow(np.abs(diff), vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('abs(diff)')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN network tests')
    subparsers = parser.add_subparsers(dest='command', required=True)

    line_parser = subparsers.add_parser('line', help='run line test on provided model')
    line_parser.add_argument('model', type=str, help='model to test')
    line_parser.add_argument('--save', action='store_true')

    circ_parser = subparsers.add_parser('circle', help='run circle test on provided model')
    circ_parser.add_argument('model', type=str, help='model to test')
    circ_parser.add_argument('--save', action='store_true')

    extrema_parser = subparsers.add_parser('extrema', help='show examples where the provided model performs well/poorly on the given dataset')
    extrema_parser.add_argument('model', type=str, help='model to test')
    extrema_parser.add_argument('dataset', type=str, help='test dataset')
    extrema_parser.add_argument('--best', action='store_true', help='look for best results instead of worst')

    seg_parser = subparsers.add_parser('segment', help='test out segmentation method for extracting distribution from image')
    seg_parser.add_argument('model', type=str, help='model')
    seg_parser.add_argument('dataset', type=str, help='test dataset')
    seg_parser.add_argument('--sample', type=int, help='sample to test')
    seg_parser.add_argument('--isolate', action='store_true')

    conn_parser = subparsers.add_parser('connectivity', help='compute connectivity for a CNN output')
    conn_parser.add_argument('model', type=str, help='model')
    conn_parser.add_argument('dataset', type=str, help='test dataset')
    conn_parser.add_argument('--sample', type=int, help='sample to test')
    conn_parser.add_argument('--save', action='store_true')
    conn_parser.add_argument('--train', action='store_true', help='draw sample from training data')

    stats_parser = subparsers.add_parser('stats', help='compute performance statistics for a dataset')
    stats_parser.add_argument('model', type=str, help='model')
    stats_parser.add_argument('dataset', type=str, help='test dataset')
    stats_parser.add_argument('--train', action='store_true', help='run stats on training data partition')

    var_parser = subparsers.add_parser('variation', help='show variation in model outputs')
    var_parser.add_argument('model', type=str, help='model')
    var_parser.add_argument('dataset', type=str, help='test dataset')
    var_parser.add_argument('--sample', type=int, help='sample to test')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_test(args)
    elif args.command == 'circle':
        circle_test(args)
    elif args.command == 'extrema':
        extrema_test(args)
    elif args.command == 'segment':
        segment_test(args)
    elif args.command == 'connectivity':
        connectivity_test(args)
    elif args.command == 'stats':
        stats_test(args)
    elif args.command == 'variation':
        variation_test(args)
