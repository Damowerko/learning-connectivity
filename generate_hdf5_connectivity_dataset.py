#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
from scipy.stats import norm
from pathlib import Path
import shutil
import os
import inspect
import time
import datetime
import h5py
import json
import argparse

from network_planner.connectivity_optimization import ConnectivityOpt
from socp.channel_model import PiecewiseChannel
from view_hdf5_dataset import display_samples


# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def pos_to_subs(res, pts):
    """
    assume origin is at (0,0) and x,y res is equal
    """
    return np.floor(pts / res).astype(int)


def human_readable_duration(dur):
    t_str = []
    for unit, name in zip((86400., 3600., 60., 1.), ('d','h','m','s')):
        if dur / unit > 1.:
            t_str.append(f'{int(dur / unit)}{name}')
            dur -= int(dur / unit) * unit
    return ' '.join(t_str)


def kernelized_config_img(config, params):
    img = np.zeros(params['img_size'])
    for agent in config:
        dist = np.linalg.norm(params['xy'] - agent, axis=2)
        mask = dist < 3.0*params['kernel_std']
        img[mask] = np.maximum(img[mask], norm.pdf(dist[mask], scale=params['kernel_std']))
    img *= 255.0 / norm.pdf(0, scale=params['kernel_std']) # normalize image to [0.0, 255.0]
    return np.clip(img, 0, 255)


def generate_hdf5_image_data(hdf5_file, mode, sample_count, params):

    # initialize hdf5 datastructure
    #
    # file structure is:
    # hdf5 file
    #  - ...
    #  - mode
    #    - task_config
    #    - init_img
    #    - comm_config
    #    - final_img
    #    - connectivity

    hdf5_grp = hdf5_file.create_group(mode)

    hdf5_grp.create_dataset('task_config', (sample_count, params['task_agents'], 2), np.float64)
    hdf5_grp.create_dataset('task_img', (sample_count,) + params['img_size'], np.uint8)
    hdf5_grp.create_dataset('comm_config', (sample_count, params['comm_agents'], 2), np.float64)
    hdf5_grp.create_dataset('comm_img', (sample_count,) + params['img_size'], np.uint8)
    hdf5_grp.create_dataset('connectivity', (sample_count,), np.float64)

    # image generation loop

    bbx = params['bbx']
    t0 = time.time()
    comm_idcs = np.arange(params['comm_agents']) + params['task_agents']

    for i in range(sample_count):

        # initial configuration of task agents as numpy array
        task_config = np.random.random((params['task_agents'],2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        hdf5_grp['task_config'][i,...] = task_config

        # configuration of task agents as image
        hdf5_grp['task_img'][i,...] = kernelized_config_img(task_config, params)

        # configuration of network agents as numpy array
        comm_config = np.random.random((params['comm_agents'],2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        conn_opt = ConnectivityOpt(params['channel_model'], task_config, comm_config)
        conn_opt.maximize_connectivity()
        comm_config = conn_opt.config[comm_idcs,:]
        hdf5_grp['comm_config'][i,...] = comm_config

        # configuration of comm team as image
        hdf5_grp['comm_img'][i,...] = kernelized_config_img(comm_config, params)

        # connectivity
        l2 = ConnectivityOpt.connectivity(params['channel_model'], task_config, comm_config)
        hdf5_grp['connectivity'][i] = l2

        duration_str = human_readable_duration(time.time()-t0)
        print(f'saved sample {i+1} of {sample_count}, elapsed time: {duration_str}\r', end="")


if __name__ == '__main__':

    # params

    task_agents = 4
    comm_agents = 3
    samples = 10
    train_percent = 0.85
    space_side_length = 30  # length of a side of the image in meters
    img_res = 128           # pixels per side of a square image
    kernel_std = 1.0        # standard deviation of gaussian kernel marking node positions
    datadir = Path(__file__).resolve().parent / 'data'

    # parse input

    parser = argparse.ArgumentParser(description='generate dataset for learning connectivity')
    parser.add_argument('--filename', default='connectivity', type=str, help='name of database to be generated')
    parser.add_argument('--overwrite', action='store_true', help='overwrite database if one already exists with the same name')
    parser.add_argument('--samples', default=10, type=int, help=f'number of samples to generate, default is {samples}')
    parser.add_argument('--display', action='store_true', help='display a sample of the data after generation')
    p = parser.parse_args()

    # init params

    params = {}
    params['task_agents'] = task_agents
    params['comm_agents'] = comm_agents
    params['img_size'] = (img_res, img_res)
    params['bbx'] = [0, space_side_length, 0, space_side_length]
    params['meters_per_pixel'] = space_side_length / img_res
    params['kernel_std'] = kernel_std

    param_file_name = datadir / (p.filename + '.json')
    with open(param_file_name, 'w') as f:
        json.dump(params, f, indent=4, separators=(',', ': '))

    # these params don't need to be saved but save time to precompute
    params['bbx'] = np.asarray(params['bbx'])
    params['channel_model'] = PiecewiseChannel(print_values=False)
    ij = np.stack(np.meshgrid(np.arange(img_res), np.arange(img_res), indexing='ij'), axis=2)
    params['xy'] = params['meters_per_pixel'] * (ij + 0.5)

    print(f"using {img_res}x{img_res} images with {params['meters_per_pixel']} meters/pixel")

    # generate random training data

    data_file_name = datadir / (p.filename + '.hdf5')
    if data_file_name.exists() and not p.overwrite:
        print(f'no action taken: database {data_file_name} already exists and overwriting is disabled')
        exit(0)
    hdf5_file = h5py.File(data_file_name, mode='w')

    sample_counts = np.round(p.samples * (np.asarray([0,1]) + train_percent*np.asarray([1,-1]))).astype(int)
    for count, mode in zip(sample_counts, ('train','test')):
        print(f'generating {count} samples for {mode}ing')

        t0 = time.time()
        generate_hdf5_image_data(hdf5_file, mode, count, params)
        duration_str = human_readable_duration(time.time()-t0)
        print(f'generated {count} samples for {mode}ing in {duration_str}')

    # view a selection of the generated data

    if p.display:
        display_samples(hdf5_file, 3)

    print(f'saved data to: {hdf5_file.filename}')
    hdf5_file.close()
