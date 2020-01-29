#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:10:56 2018

@author: gtucker
"""

import numpy as np
import datetime
from grainhill import GrainFacetSimulator
from grainhill import SlopeMeasurer
import landlab
from landlab.io.native_landlab import save_grid
import os


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory ' + directory)


params = {
    'grid_size' : (166, 121),
    'report_interval' : 5.0, 
    'run_duration' : 200000.0, 
    'output_interval' : 1.0e99, 
    'disturbance_rate' : 0.0,
    'weathering_rate' : 0.0,
    'dissolution_rate': 0.0,
    'uplift_interval' : 866.0,
    'plot_interval' : 20000.0,
    'friction_coef' : 1.0,
    'fault_x' : -0.01, 
    'cell_width' : 0.5, 
    'grav_accel' : 9.8,
    }


# Open a file to record output:
d = datetime.datetime.today()
today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
results_file = open('results_sadw_large' + today_str + '.csv', 'w')
results_file.write('Landlab version,' + landlab.__version__ + ',\n')


# Print header in file
results_file.write('Disturbance rate parameter (1/yr),Weathering rate '
                   + 'parameter (1/yr),Gradient (m/m),'
                   + 'Slope angle (deg)\n')


# Sweep through a range of dissolution rate parameters
for dist_exp in np.arange(-4, 0):
    for weath_exp in np.arange(-4, -0.9, 0.1):

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

        sm = SlopeMeasurer(gfs)
        sm.pick_rock_surface()
        (m, b) = sm.fit_straight_line_to_surface()
        angle = np.degrees(np.arctan(m))

        results_file.write(str(weath_rate) + ',' + str(dist_rate) + ',' 
                           + str(m) + ',' + str(angle) + '\n')
        results_file.flush()

        save_grid(gfs.grid, opname + '/' + opname + '.grid', clobber=True)

results_file.close()
