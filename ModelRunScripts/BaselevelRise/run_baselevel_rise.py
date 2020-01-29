#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:10:56 2018

@author: gtucker
"""

import numpy as np
from grainhill import GrainFacetSimulator
from landlab.io.native_landlab import save_grid
import os


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory ' + directory)


params = {
    'grid_size' : (111, 81),
    'report_interval' : 5.0, 
    'run_duration' : 130000.0, 
    'output_interval' : 1.0e99, 
    'disturbance_rate' : 0.0,
    'weathering_rate' : 0.0,
    'dissolution_rate': 0.0,
    'uplift_interval' : 866.0,
    'baselevel_rise_interval' : 4000.0,
    'plot_interval' : 13000.0,
    'friction_coef' : 1.0,
    'fault_x' : 23.0, 
    'cell_width' : 0.5, 
    'grav_accel' : 9.8,
    }



# Sweep through a range of parameters
for dist_exp in np.arange(-4, 0):
    for weath_exp in np.arange(-4, 0):

        weath_rate = 10.0**weath_exp
        dist_rate = 10.0**dist_exp
        params['disturbance_rate'] = dist_rate
        params['weathering_rate'] = weath_rate
        print('Disturbance rate: ' + str(params['disturbance_rate']) + ' 1/y')
        print('Weathering rate: ' + str(params['weathering_rate']) + ' 1/y')

        opname = ('d' + str(int(10 * dist_exp)) + 'w' + str(int(10 * weath_exp)))
        create_folder(opname)
        params['plot_file_name'] = opname + '/' + opname

        gfs = GrainFacetSimulator(**params)
        gfs.run()
        
        save_grid(gfs.grid, opname + '.grid', clobber=True)
