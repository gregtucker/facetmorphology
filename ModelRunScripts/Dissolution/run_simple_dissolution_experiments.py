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
    'grid_size' : (111, 81),
    'report_interval' : 5.0, 
    'run_duration' : 130000.0, 
    'output_interval' : 1.0e99, 
    'disturbance_rate' : 0.0,
    'weathering_rate' : 0.0,
    'uplift_interval' : 866.0,
    'plot_interval' : 2600.0,
    'friction_coef' : 1.0,
    'fault_x' : -0.01, 
    'cell_width' : 0.5, 
    'grav_accel' : 9.8,
    }


# Open a file to record output:
d = datetime.datetime.today()
today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
results_file = open('results' + today_str + '.csv', 'w')
results_file.write('Landlab version,' + landlab.__version__ + ',\n')


# Print header in file
results_file.write('Dissolution rate parameter (1/yr),Gradient (m/m),'
                   + 'Slope angle (deg)\n')


# Sweep through a range of dissolution rate parameters
for diss_rate in np.arange(4.0e-5, 5.0e-4, 4.0e-5):

    params['dissolution_rate'] = diss_rate
    print('Dissolution rate: ' + str(params['dissolution_rate']) + ' 1/y')

    opname = 'dissolve_dr' + str(int(diss_rate * 1.0e5))
    create_folder(opname)
    params['plot_file_name'] = opname + '/' + opname

    gfs = GrainFacetSimulator(**params)
    gfs.run()

    sm = SlopeMeasurer(gfs)
    sm.pick_rock_surface()
    (m, b) = sm.fit_straight_line_to_surface()
    angle = np.degrees(np.arctan(m))

    results_file.write(str(diss_rate) + ',' + str(m) + ',' + str(angle) + '\n')
    
    save_grid(gfs.grid, opname + '/' + opname + '.grid')

results_file.close()
