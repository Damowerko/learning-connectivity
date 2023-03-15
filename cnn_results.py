import argparse
import json
import time
from math import ceil
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy import spatial

from cnn import get_file_name, load_model_for_eval
from connectivity_maximization import circle_points
from hdf5_dataset_utils import ConnectivityDataset, cnn_image_parameters, plot_image
from mid import lloyd
from mid.channel_model import PiecewisePathLossModel
from mid.connectivity_optimization import ConnectivityOpt as ConnOpt
from mid.connectivity_optimization import round_sf
from mid.feasibility import adaptive_bbx, connect_graph, min_feasible_sample


def scale_from_filename(filename):
    parts = filename.split('_')
    if '256' in parts:
        return 2
    return 1


def compute_peaks(image, threshold_val=80, blur_sigma=1, region_size=7, view=False):
    out_peaks, blurred_img = lloyd.compute_peaks(image, threshold_val, blur_sigma, region_size)
    if view:
        fig, ax = plt.subplots()
        ax.plot(out_peaks[:,0], out_peaks[:,1], 'ro')
        ax.imshow(blurred_img.T)
        ax.invert_yaxis()
        plt.show()
    return out_peaks


def connected_components(adjacency_matrix):
    nodes = list(range(adjacency_matrix.shape[0]))
    visited = adjacency_matrix.shape[0]*[False]
    components = []

    while len(nodes) > 0:

        frontier = [nodes[0]]
        comp = []
        while len(frontier) > 0:
            node = frontier.pop()
            if visited[node]:
                continue
            nodes.remove(node)
            comp.append(node)
            visited[node] = True
            frontier += np.nonzero(adjacency_matrix[node,:])[0].tolist()

        components.append(comp)

    return components


def connectivity_from_CNN(input_image, model, x_task, params, samples=1, viz=False,
                          variable_power=True, min_config=True):

    tol = 5e-4  # connectivity threshold

    conn = np.zeros((samples,))
    power = np.zeros((samples,))
    cnn_imgs = np.zeros((samples,) + input_image.shape, dtype=input_image.dtype)
    agents = np.zeros((samples,), dtype=int)
    x_comm = []
    for i in range(samples):

        # run inference and extract the network team configuration

        cnn_imgs[i] = model.evaluate(torch.from_numpy(input_image)).cpu().detach().numpy()
        x_comm += [compute_coverage(cnn_imgs[i], params, viz=viz)]
        agents[i] = x_comm[i].shape[0]

        # remove extranious communication agents
        if min_config:

            config = np.vstack((x_task, x_comm[i]))
            rate = params['channel_model'].predict(config)[0]
            edm = spatial.distance_matrix(config, config)
            edm[rate < 1e-4] = 0
            cc = connected_components(rate)
            while len(cc) > 1:

                # find the two connected components as well as the points in each
                # that are closest together
                cc_inds = []
                cc_min_dist = None
                config_inds = []
                for k in range(len(cc)):
                    for j in range(k+1,len(cc)):
                        dists = spatial.distance_matrix(config[cc[k],:], config[cc[j],:])
                        if cc_min_dist is None or dists.min() < cc_min_dist:
                            cc_min_dist = dists.min()
                            ind1, ind2 = np.where(dists == dists.min())
                            cc_inds = [k, j]
                            config_inds = [cc[k][ind1[0]], cc[j][ind2[0]]]

                # connect the two closets connected components
                cc[cc_inds[0]] = cc[cc_inds[0]] + cc[cc_inds[1]]
                del cc[cc_inds[1]]
                edm[config_inds, config_inds[::-1]] = cc_min_dist

            graph = nx.from_numpy_array(edm)
            terminals = list(range(x_task.shape[0]))
            st = nx.algorithms.approximation.steiner_tree(graph, terminals)

            st_nodes = list(st.nodes)
            if len(st_nodes) < config.shape[0]:
                comm_idcs = np.asarray(st_nodes[x_task.shape[0]:]) - x_task.shape[0]
                x_comm[i] = x_comm[i][comm_idcs,:]

        # find connectivity
        conn[i] = ConnOpt.connectivity(params['channel_model'], x_task, x_comm[i])

        # increase transmit power until the network is connected
        power[i] = params['channel_model'].t
        while variable_power and conn[i] < tol:
            power[i] += 0.2
            cm = PiecewisePathLossModel(print_values=False, t=power[i])
            conn[i] = ConnOpt.connectivity(cm, x_task, x_comm[i])

    # return the best result
    # prioritize: lowest power > fewer agents > highest connectivity
    if samples == 1:
        best_idx = 0
    else:
        mask = power == np.min(power)
        mask &= agents == np.min(agents[mask])
        best_idx = np.where(conn == np.max(conn[mask]))[0][0]

    return conn[best_idx], x_comm[best_idx], power[best_idx], cnn_imgs[best_idx]


def connectivity_from_opt(task_config, p, viz=False):
    comm_config = connect_graph(task_config, p['comm_range'])
    opt = ConnOpt(p['channel_model'], task_config, comm_config)
    conn = opt.maximize_connectivity(viz=viz)
    return conn, opt.get_comm_config()


def compute_coverage(image, params, viz=False):
    """compute coverage using Lloyd's Algorithm"""

    # extract peaks of intensity image
    config_subs = compute_peaks(image, threshold_val=60, view=viz)
    config = lloyd.sub_to_pos(params['meters_per_pixel'], params['img_size'][0], config_subs)
    peaks = np.copy(config)

    # Lloyd's algorithm
    coverage_range = params['coverage_range']
    it = 1
    while True:
        voronoi_cells = lloyd.compute_voronoi(config, params['bbx'])
        new_config = lloyd.lloyd_step(image, params['xy'], config, voronoi_cells, coverage_range)
        update_dist = np.sum(np.linalg.norm(config - new_config, axis=1))

        cell_patches = []
        circle_patches = []
        if viz:
            for i, cell in enumerate(voronoi_cells):
                cell_patches.append(mpl.patches.Polygon(cell.points, True))
                circle_patches.append(mpl.patches.Circle(config[i], radius=coverage_range))

            p = mpl.collections.PatchCollection(cell_patches, alpha=0.2)
            p.set_array(np.arange(len(cell_patches))*255/(len(cell_patches)-1))
            p.set_cmap('jet')

            fig, ax = plt.subplots()
            plot_image(ax, image, params=params)
            ax.add_collection(p)
            ax.add_collection(mpl.collections.PatchCollection(circle_patches, ec='r', fc='none'))
            ax.plot(peaks[:,0], peaks[:,1], 'rx', label='peaks')
            ax.plot(new_config[:,0], new_config[:,1], 'bo', label='centroids')
            # ax.plot(config[:,1], config[:,0], 'bx', color=(0,1,0), label='prev config')
            ax.set_title(f'it {it}, cond = {round_sf(update_dist,3)}')
            ax.legend()
            plt.show()

        # exit if the configuration hasn't appreciably changed
        if update_dist < 1e-5:
            break

        config = new_config
        it += 1

    return config


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
    model_image = model.inference(input_image)

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    params = cnn_image_parameters(img_scale_factor)

    coverage_points = compute_coverage(model_image, params, viz=args.view)

    fig, ax = plt.subplots()
    if args.isolate:
        plot_image(ax, model_image, params=params)
    else:
        plot_image(ax, np.maximum(model_image, input_image), params=params)
    ax.plot(coverage_points[:,0], coverage_points[:,1], 'ro')
    # ax.axis('off')
    plt.show()
    print(f'optimal coverage computed for {dataset_file.name} test partition sample {idx}')


def extract_128px_center_image(image, scale):
    center_idx = 128 * scale // 2
    return image[center_idx-64:center_idx+64, center_idx-64:center_idx+64]


def save_case(data, existing_fig=True):

    if existing_fig:
        # figure
        plt.savefig(data['filename'] + '.png', dpi=150)
        plt.close()

    # raw image
    fig = plt.figure()
    fig.set_size_inches((4,4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.flipud(data['img'].T), aspect='equal', cmap='gist_gray_r')
    plt.savefig(data['filename'] + '_raw.png', dpi=150)
    plt.close()

    # data
    with open(data['filename'] + '.json', 'w') as f:
        json.dump(data['json'], f, indent=4)

    print(f"saved figure, image, and data {data['filename']+'{.png,_raw.png,.json}'}")


def line_main(args):
    line_test(args.model, args.draws, args.save, args.steps)


def line_test(model_file, draws=1, save=True, steps=20):

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    scale = scale_from_filename(model_name)
    params = cnn_image_parameters(scale)

    dists = np.linspace(40, 120, steps)
    for i, dist in enumerate(dists):
        x_task = np.asarray([[-dist,0], [dist,0]]) / 2
        img = lloyd.kernelized_config_img(x_task, params)

        cnn_conn, x_cnn, cnn_pwr, cnn_img = connectivity_from_CNN(img, model, x_task, params, draws)
        opt_conn, x_opt = connectivity_from_opt(x_task, params)

        disp_img = extract_128px_center_image(np.maximum(cnn_img, img), scale)

        fig, ax = plt.subplots()
        plot_image(ax, disp_img, x_task=x_task, x_opt=x_opt, x_cnn=x_cnn)
        ax.set_yticks(np.arange(-60, 80, 30))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='best', fontsize=14)

        pwr_diff = cnn_pwr - params['channel_model'].t
        pwr_str = '0' if pwr_diff < 0.01 else f'{pwr_diff:.1f}'
        ax.set_title(f'opt. ({opt_conn:.3f}, 0) | CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=18)

        plt.tight_layout()

        if save:
            data = {'filename': f'line_{i:02d}_{model_name}', 'img': disp_img,
                    'json': {'task': {'config': x_task.tolist()},
                             'cnn': {'config': x_cnn.tolist(), 'power': cnn_pwr, 'connectivity': cnn_conn},
                             'opt': {'config': x_opt.tolist(), 'power': params['channel_model'].t,
                                     'connectivity': opt_conn}}}
            save_case(data)
        else:
            plt.show()


def circle_main(args):
    circle_test(args.model, args.agents, args.draws, args.save, args.steps)


def circle_test(model_file, task_agents=3, draws=1, save=True, steps=15):

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    scale = scale_from_filename(model_name)
    params = cnn_image_parameters(scale)

    min_rad = (params['comm_range']+2.0) / (2.0 * np.sin(np.pi / task_agents))
    rads = np.linspace(min_rad, 60, steps)
    for i, rad in enumerate(rads):
        x_task = circle_points(rad, task_agents)
        img = lloyd.kernelized_config_img(x_task, params)

        cnn_conn, x_cnn, cnn_pwr, cnn_img = connectivity_from_CNN(img, model, x_task, params, draws)
        opt_conn, x_opt = connectivity_from_opt(x_task, params)

        # print(f'it {i+1:2d}: rad = {rad:.1f}m, cnn # = {x_cnn.shape[0]}, '
        #       f'cnn conn = {cnn_conn:.4f}, opt # = {x_opt.shape[0]}, '
        #       f'opt conn = {opt_conn:.4f}')

        disp_img = extract_128px_center_image(np.maximum(cnn_img, img), scale)

        fig, ax = plt.subplots()
        plot_image(ax, disp_img, x_task=x_task, x_opt=x_opt, x_cnn=x_cnn)
        ax.set_yticks(np.arange(-60, 80, 30))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='best', fontsize=14)

        pwr_diff = cnn_pwr - params['channel_model'].t
        pwr_str = '0' if pwr_diff < 0.01 else f'{pwr_diff:.1f}'
        ax.set_title(f'opt. ({opt_conn:.3f}, 0) | CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=18)

        plt.tight_layout()

        if save:
            data = {'filename': f'circle_{i:02d}_agents_{task_agents}_{model_name}', 'img': disp_img,
                    'json': {'task': {'config': x_task.tolist()},
                             'cnn': {'config': x_cnn.tolist(), 'power': cnn_pwr, 'connectivity': cnn_conn},
                             'opt': {'config': x_opt.tolist(), 'power': params['channel_model'].t,
                                     'connectivity': opt_conn}}}
            save_case(data)
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
        print('seeking the best performing sample')
    else:
        extrema_test = lambda l : l > extreme_loss
        extreme_loss = 0.0
        print('seeking the worst performing sample')
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

    extrema_type = 'best' if args.best else 'worst'
    print(f'{extrema_type} sample is {extreme_idx}{20*" "}')
    ax = plt.subplot(1,3,1)
    plot_image(ax, input_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('input')
    ax = plt.subplot(1,3,2)
    plot_image(ax, output_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('target')
    ax = plt.subplot(1,3,3)
    plot_image(ax, model_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('output')
    plt.tight_layout()
    plt.show()

    hdf5_file.close()


def connectivity_main(args):
    connectivity_test(args.model, args.dataset, args.sample, args.save, args.train, args.nost)


def connectivity_test(model_file, dataset, sample=None, save=True, train=False, nost=False):

    model = load_model_for_eval(model_file)
    if model is None:
        return

    mode = 'train' if train else 'test'

    dataset_file = Path(dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file[mode]['task_img'].shape[0]

    if sample is None:
        idx = np.random.randint(dataset_len)
    elif sample >= dataset_len:
        print(f'sample index {sample} out of dataset index range: [0-{dataset_len-1}]')
        return
    else:
        idx = sample

    task_img = hdf5_file[mode]['task_img'][idx,...]
    opt_conn = hdf5_file[mode]['connectivity'][idx]
    x_task = hdf5_file[mode]['task_config'][idx,...]
    x_opt = hdf5_file[mode]['comm_config'][idx,...]
    x_opt = x_opt[~np.isnan(x_opt[:,0])]

    img_scale_factor = hdf5_file['train']['task_img'].shape[1] // 128
    params = cnn_image_parameters(img_scale_factor)

    cnn_conn, x_cnn, cnn_L0, cnn_img = connectivity_from_CNN(task_img, model, x_task, params, min_config=not nost)

    disp_img = np.maximum(task_img, cnn_img)

    ax = plt.subplot()
    plot_image(ax, disp_img, x_task=x_task, x_opt=x_opt, x_cnn=x_cnn, params=params)
    ax.legend(loc='best', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)

    pwr_diff = cnn_L0 - params['channel_model'].t
    pwr_str = '0' if pwr_diff < 0.01 else f'{pwr_diff:.1f}'
    ax.set_title(f'{idx}: opt. ({opt_conn:.3f}, 0), CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=14)

    plt.tight_layout()

    if not save:
        print(f'showing sample {idx} from {dataset_file.name}')
        plt.show()
    else:
        data = {'filename': str(idx) + '_' + dataset_file.stem, 'img': disp_img,
                'json': {'task': {'config': x_task.tolist()},
                         'cnn': {'config': x_cnn.tolist(), 'power': cnn_pwr, 'connectivity': cnn_conn},
                         'opt': {'config': x_opt.tolist(), 'power': params['channel_model'].t,
                                 'connectivity': opt_conn}}}
        save_case(data)

    plt.close()
    hdf5_file.close()


def stats_computer(dataset_params, sample_queue, results_queue):

    for sample in iter(sample_queue.get, None):

        idx = sample['idx']
        cnn_img = sample['cnn_img']
        x_task = sample['x_task']

        # TODO sometimes models don't place any agents... need to handle this better
        try:
            x_cnn = compute_coverage(cnn_img, dataset_params)
        except:
            print(f'compute_coverate(...) failed for sample {idx}. skipping')
            continue

        cnn_conn = ConnOpt.connectivity(dataset_params['channel_model'], x_task, x_cnn)

        # increase transmit power until the network is connected
        cnn_power = dataset_params['channel_model'].t
        while cnn_conn < 5e-4:
            cnn_power += 0.2
            cm = PiecewisePathLossModel(print_values=False, t=cnn_power)
            cnn_conn = ConnOpt.connectivity(cm, x_task, x_cnn)

        out_dict = {}
        out_dict['idx'] = idx
        out_dict['cnn_conn'] = cnn_conn
        out_dict['cnn_count'] = x_cnn.shape[0]
        out_dict['cnn_power'] = cnn_power
        out_dict['opt_conn'] = sample['opt_conn']
        out_dict['opt_count'] = sample['opt_count']

        results_queue.put(out_dict)


def stats_compiler(params, results_queue):

    dataset_len = params['dataset_len']
    filename = params['filename']
    nosave = params['nosave']
    default_transmit_power = params['default_transmit_power']

    opt_conn = np.zeros((dataset_len, 1))
    cnn_conn = np.zeros_like(opt_conn)
    cnn_power = np.zeros_like(opt_conn)
    cnn_count = np.zeros_like(opt_conn, dtype=int)
    opt_count = np.zeros_like(opt_conn, dtype=int)

    processed_count = 1
    for result in iter(results_queue.get, None):
        idx = result['idx']
        opt_conn[idx] = result['opt_conn']
        cnn_conn[idx] = result['cnn_conn']
        cnn_power[idx] = result['cnn_power']
        cnn_count[idx] = result['cnn_count']
        opt_count[idx] = result['opt_count']
        print(f'\rprocessed sample {processed_count} of {dataset_len}\r', end="")
        processed_count += 1
    print(f'finished processing {dataset_len} samples')

    if not nosave:
        stats = np.hstack((opt_conn, cnn_conn, cnn_power, opt_count, cnn_count))
        np.save(filename, stats)
        print(f'saved data to {filename}.npy')
    else:
        print('NOT saving data')

    eps = 1e-10
    opt_feasible = opt_conn > eps
    cnn_feasible = cnn_conn > eps
    cnn_morepower = cnn_power > default_transmit_power
    both_feasible = opt_feasible & cnn_feasible

    agent_count_diff = cnn_count - opt_count

    absolute_error = opt_conn[both_feasible] - cnn_conn[both_feasible]
    percent_error = absolute_error / opt_conn[both_feasible]

    print(f'{np.sum(opt_feasible)}/{dataset_len} feasible with optimization')
    print(f'{np.sum(cnn_feasible)}/{dataset_len} feasible with CNN')
    print(f'{np.sum(cnn_feasible & ~opt_feasible)} cases where only the CNN was feasible')
    print(f'{np.sum(cnn_morepower)} cases where the CNN required more transmit power')
    print(f'cnn power use:  mean = {np.mean(cnn_power):.2f}, std = {np.std(cnn_power):.4f}')
    print(f'cnn agent diff: mean = {np.mean(agent_count_diff):.3f}, std = {np.std(agent_count_diff):.4f}')
    print(f'absolute error: mean = {np.mean(absolute_error):.4f}, std = {np.std(absolute_error):.4f}')
    print(f'percent error:  mean = {100*np.mean(percent_error):.2f}%, std = {100*np.std(percent_error):.2f}%')


def compute_stats_main(args):
    compute_stats_test(args.model, args.dataset, args.train, args.samples, args.nosave, args.jobs)


def compute_stats_test(model_file, dataset_file, train=False, samples=None, nosave=False, jobs=None):

    # load model and dataset

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    dataset_file = Path(dataset_file)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')

    mode = 'train' if train else 'test'
    dataset_len = hdf5_file[mode]['task_img'].shape[0]
    if samples is not None:
        if samples > dataset_len:
            print(f'requested sample count ({samples}) > dataset length ({dataset_len})')
            return
        else:
            dataset_len = samples

    results_filename = f'{dataset_len}_samples_{model_name}_{dataset_file.stem}_stats'

    # configure mutli-processing

    num_processes = max(cpu_count()-2, 1) if jobs is None else jobs
    sample_queue = Queue(maxsize=num_processes*2)
    results_queue = Queue()

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    dataset_params = cnn_image_parameters(img_scale_factor)

    results_proc_params = {}
    results_proc_params['dataset_len'] = dataset_len
    results_proc_params['filename'] = results_filename
    results_proc_params['nosave'] = nosave
    results_proc_params['default_transmit_power'] = dataset_params['channel_model'].t
    results_proc = Process(target=stats_compiler, args=(results_proc_params, results_queue))
    results_proc.start()

    worker_procs = []
    for i in range(num_processes):
        p = Process(target=stats_computer, args=(dataset_params, sample_queue, results_queue))
        worker_procs.append(p)
        p.start()

    # load the sample queue until all test cases have been processed

    print(f'processing {dataset_len} samples with {num_processes} processes')
    for i in range(dataset_len):

        # do inference
        input_image = hdf5_file[mode]['task_img'][i,...]
        cnn_image = model.evaluate(torch.from_numpy(input_image)).cpu().detach().numpy()

        sample_dict = {}
        sample_dict['idx'] = i
        sample_dict['cnn_img'] = cnn_image
        sample_dict['x_task'] = hdf5_file[mode]['task_config'][i,...]
        sample_dict['opt_conn'] = hdf5_file[mode]['connectivity'][i,...]
        sample_dict['opt_count'] = np.sum(~np.isnan(hdf5_file[mode]['comm_config'][i,:,0]))
        sample_queue.put(sample_dict)

    # each worker process exits once it receives a None
    for i in range(num_processes):
        sample_queue.put(None)
    for proc in worker_procs:
        proc.join()

    # finally, the writer process also exits once it receives a None
    results_queue.put(None)
    results_proc.join()

    return results_filename + '.npy' if not nosave else None


def parse_stats_main(args):
    parse_stats_test(args.stats, args.labels, args.save, args.nolog)


def parse_stats_test(stats_files, labels, save=True, nolog=False):

    if len(stats_files) != len(labels):
        print(f'number of stats files ({len(stats_files)}) must match number of labels ({len(labels)})')
        return

    stats = {}
    for filename, label in zip(stats_files, labels):
        stats_file = Path(filename)
        if not stats_file.exists():
            print(f'{stats_file} does not exist')
            return
        data = np.load(stats_file)
        stats[label] = {'power': data[:,2], 'opt_count': data[:,3], 'cnn_count': data[:,4]}

    # histogram of transmit powers

    powers = [np.round(stats[label]['power'], 1) for label in labels]
    bins = np.hstack(([-0.9, 0.1], np.arange(1.1, 10.1, 1.0)))

    fig, ax = plt.subplots()
    ax.hist(powers, bins=bins, stacked=False, log=not nolog, density=True, label=labels)
    ax.legend(loc='best', fontsize=16)
    ax.set_xticks(np.arange(0, 10, 3))
    if not nolog: ax.set_yticks([10 ** (-i) for i in range(5)])
    ax.set_xlabel('$P_T$ dBm', fontsize=18)
    ax.set_ylabel('fraction of test cases', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    if save:
        tmp_powers = np.vstack([stats[k]['power'].reshape(-1,1) for k in stats.keys()])
        np.savetxt('transmit_power.csv', tmp_powers, fmt='%.1f', delimiter=',', header='power')

        plt.savefig('power_histogram.pdf', dpi=150)
        print('saved power_histogram.pdf and transmit_power.csv')

    # histogram of difference between CNN agent count and opt agent count

    p = cnn_image_parameters()
    agent_diffs = []
    for label in labels:
        mask = np.round(stats[label]['power'], 1) == p['channel_model'].t  # all samples where CNN used default power
        agent_diffs += [(stats[label]['cnn_count'][mask] - stats[label]['opt_count'][mask]).astype(int)]

    bins = np.arange(np.min(np.hstack(agent_diffs)), np.max(np.hstack(agent_diffs))-2, 1)

    fig, ax = plt.subplots()
    ax.hist(agent_diffs, bins=bins, stacked=False, log=not nolog, density=True, align='left', label=labels)
    ax.legend(loc='best', fontsize=18)
    ax.set_xticks(bins[:-1])
    if not nolog: ax.set_yticks([10 ** (-i) for i in range(5)])
    ax.set_xlabel('# CNN agents $-$ # opt agents', fontsize=18)
    ax.set_ylabel('fraction of test cases', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    if save:
        diffs = np.vstack([arr.reshape(-1,1) for arr in agent_diffs])
        np.savetxt('agent_diffs.csv', diffs, fmt='%d', delimiter=',', header='diff')

        plt.savefig('agent_count_histogram.pdf', dpi=150)
        print('saved agent_count_histogram.pdf and agent_diffs.csv')
    else:
        plt.show()

    avg_team_size = np.mean(np.hstack([stats[l]['cnn_count'] + stats[l]['opt_count'] for l in labels]))
    print(f'power: mean = {np.mean(np.hstack(powers)):.2f}, std = {np.std(np.hstack(powers)):.3f}')
    print(f'diffs: mean = {np.mean(np.hstack(agent_diffs)):.4f}, std = {np.std(np.hstack(agent_diffs)):.3f}'
          f' ({sum([len(d) for d in agent_diffs])} / {sum([len(p) for p in powers])})')
    print(f'average team size: {avg_team_size:.2f}')


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


def compute_time_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    min_agents = 3
    max_agents = 20
    team_sizes = np.arange(min_agents, max_agents+1)
    samples = 20

    img_scale_factor = 2
    params = cnn_image_parameters(img_scale_factor)

    cnn_time = np.zeros((team_sizes.shape[0], samples))
    opt_time = np.zeros_like(cnn_time)

    for team_idx, total_agents in enumerate(team_sizes):

        for sample_idx in range(samples):

            # generate sample with fixed number of agents
            task_agents = ceil(total_agents / 2.0)
            bbx = adaptive_bbx(task_agents, params['comm_range'])
            while True:
                x_task, x_comm = min_feasible_sample(task_agents, params['comm_range'], bbx)
                if x_task.shape[0] + x_comm.shape[0] == total_agents:
                    break
            input_image = lloyd.kernelized_config_img(x_task, params)

            print(f'\rtiming sample {sample_idx}/{samples} for team {total_agents} agent team\r', end="")

            # run CNN
            t0 = time.time()
            model.inference(input_image)
            connectivity_from_CNN(input_image, model, x_task, params, args.draws)
            dt = time.time() - t0
            cnn_time[team_idx, sample_idx] = dt

            # run optimization
            opt = ConnOpt(params['channel_model'], x_task, x_comm)
            t0 = time.time()
            opt.maximize_connectivity(max_its=20)
            dt = time.time() - t0
            opt_time[team_idx, sample_idx] = dt

    res = {}
    res['agents'] = team_sizes.tolist()
    res['cnn'] = {'mean': np.mean(cnn_time, axis=1).tolist(), 'std': np.std(cnn_time, axis=1).tolist()}
    res['opt'] = {'mean': np.mean(opt_time, axis=1).tolist(), 'std': np.std(opt_time, axis=1).tolist()}
    with open('computation_time.json', 'w') as f:
        json.dump(res, f, indent=4)


def plot_time_test(args):

    data_file = Path(args.datafile)
    if not data_file.exists():
        print(f'provided datafile {data_file} not found')
        return

    with open(data_file, 'r') as f:
        data = json.load(f);

    # computation time with error bars
    fig, ax = plt.subplots()
    ax.errorbar(data['agents'], data['cnn']['mean'], yerr=data['cnn']['std'], color='r', lw=2, label='CNN')
    ax.errorbar(data['agents'], data['opt']['mean'], yerr=data['opt']['std'], color='b', lw=2, label='opt')
    ax.set_xlabel('total agents', fontsize=16)
    ax.set_ylabel('computation time (s)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN network tests')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # TODO remove draws (since reparameterization is disabled in eval mode)

    line_parser = subparsers.add_parser('line', help='run line test on provided model')
    line_parser.add_argument('model', type=str, help='model to test')
    line_parser.add_argument('--save', action='store_true')
    line_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')
    line_parser.add_argument('--steps', metavar='S', type=int, default=20, help='number of steps to take between min and max radius')

    circ_parser = subparsers.add_parser('circle', help='run circle test on provided model')
    circ_parser.add_argument('model', type=str, help='model to test')
    circ_parser.add_argument('--save', action='store_true')
    circ_parser.add_argument('--agents', metavar='A', type=int, default=3, help='number of agents in the circle')
    circ_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')
    circ_parser.add_argument('--steps', metavar='S', type=int, default=15, help='number of steps to take between min and max radius')

    extrema_parser = subparsers.add_parser('extrema', help='show examples where the provided model performs well/poorly on the given dataset')
    extrema_parser.add_argument('model', type=str, help='model to test')
    extrema_parser.add_argument('dataset', type=str, help='test dataset')
    extrema_parser.add_argument('--best', action='store_true', help='look for best results instead of worst')

    seg_parser = subparsers.add_parser('segment', help='test out segmentation method for extracting distribution from image')
    seg_parser.add_argument('model', type=str, help='model')
    seg_parser.add_argument('dataset', type=str, help='test dataset')
    seg_parser.add_argument('--sample', type=int, help='sample to test')
    seg_parser.add_argument('--isolate', action='store_true', help='show the CNN output without the input image overlayed')
    seg_parser.add_argument('--view', action='store_true', help="show each iteration of Lloyd's algorithm")

    conn_parser = subparsers.add_parser('connectivity', help='compute connectivity for a CNN output')
    conn_parser.add_argument('model', type=str, help='model')
    conn_parser.add_argument('dataset', type=str, help='test dataset')
    conn_parser.add_argument('--sample', type=int, help='sample to test')
    conn_parser.add_argument('--save', action='store_true')
    conn_parser.add_argument('--train', action='store_true', help='draw sample from training data')
    conn_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')
    conn_parser.add_argument('--nost', action='store_true', help='disable removing redundant agents')

    comp_parser = subparsers.add_parser('compute_stats', help='compute performance statistics for a dataset')
    comp_parser.add_argument('model', type=str, help='model')
    comp_parser.add_argument('dataset', type=str, help='test dataset')
    comp_parser.add_argument('--train', action='store_true', help='run stats on training data partition')
    comp_parser.add_argument('--samples', type=int, help='number of samples to process; if omitted all samples in the dataset will be used')
    comp_parser.add_argument('--nosave', action='store_true', help='don\'t save connectivity data')
    comp_parser.add_argument('--jobs', '-j', type=int, metavar='N', help='number of worker processes to use; default is # of CPU cores')

    parse_parser = subparsers.add_parser('parse_stats', help='parse performance statistics saved by compute_stats')
    parse_parser.add_argument('--stats', type=str, help='stats.npy files generated from compute_stats', nargs='+')
    parse_parser.add_argument('--labels', type=str, help='labels to use with each stats file', nargs='+')
    parse_parser.add_argument('--save', action='store_true')
    parse_parser.add_argument('--nolog', action='store_true', help='disable log y-axis')

    var_parser = subparsers.add_parser('variation', help='show variation in model outputs')
    var_parser.add_argument('model', type=str, help='model')
    var_parser.add_argument('dataset', type=str, help='test dataset')
    var_parser.add_argument('--sample', type=int, help='sample to test')

    comp_time_parser = subparsers.add_parser('compute_time', help='compare CNN inference time with optimization time')
    comp_time_parser.add_argument('model', type=str, help='model')
    comp_time_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    plot_time_parser = subparsers.add_parser('plot_time', help='compare CNN inference time with optimization time')
    plot_time_parser.add_argument('datafile', type=str, help='JSON datafile with computation time stats')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_main(args)
    elif args.command == 'circle':
        circle_main(args)
    elif args.command == 'extrema':
        extrema_test(args)
    elif args.command == 'segment':
        segment_test(args)
    elif args.command == 'connectivity':
        connectivity_main(args)
    elif args.command == 'compute_stats':
        compute_stats_main(args)
    elif args.command == 'parse_stats':
        parse_stats_main(args)
    elif args.command == 'variation':
        variation_test(args)
    elif args.command == 'compute_time':
        compute_time_test(args)
    elif args.command == 'plot_time':
        plot_time_test(args)
