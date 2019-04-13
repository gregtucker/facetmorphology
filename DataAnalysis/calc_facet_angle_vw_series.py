#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:12:18 2018

@author: gtucker
"""

from landlab.io.native_landlab import load_grid
import os
import datetime
import numpy as np



# FUNCTIONS

def get_profile_and_soil_thickness(grid, node_state):
    """Calculate and return profiles of elevation and soil thickness.
    
    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> import numpy as np
    >>> hg = HexModelGrid(4, 6, orientation='vert', shape='rect')
    >>> ns = hg.add_zeros('node', 'node_state', dtype=int)
    >>> ns[:18] = np.array([8, 8, 8, 8, 8, 8, 0, 7, 8, 0, 7, 8, 0, 0, 7, 0, 0, 7])
    >>> x, z, h = get_profile_and_soil_thickness(hg, ns)
    >>> x
    array([ 0.8660254 ,  1.73205081,  2.59807621,  3.46410162])
    >>> z
    array([ 0.5,  1. ,  1.5,  2. ])
    >>> h
    array([ 0.,  1.,  1.,  1.])
    """
    nc = grid.number_of_node_columns
    elev = np.zeros(nc - 2)
    soil = np.zeros(nc - 2)
    x = np.zeros(nc - 2)
    for col in range(1, nc - 1):
        nodes_in_col = grid.nodes[:, col]
        x[col - 1] = grid.x_of_node[nodes_in_col[0]]
        states = node_state[nodes_in_col]
        (rows_with_rock_or_sed, ) = np.where(states > 0)
        if len(rows_with_rock_or_sed) == 0:
            elev[col - 1] = 0.0
        else:
            elev[col - 1] = np.amax(rows_with_rock_or_sed) + 0.5 * (col % 2)
        soil[col - 1] = np.count_nonzero(np.logical_and(states > 0, states < 8))

    return x, elev, soil


def fit_zero_intercept_line_to_surface(surface_x, surface_z):
    """
    Fit a straight line with zero intercept to a surface that has already
    been found with pick_rock_surface().

    Examples
    --------
    >>> from grainhill import GrainHill
    >>> import numpy as np
    >>> gh = GrainHill((4, 8), show_plots=False)
    >>> gh.ca.node_state[:] = 0
    >>> gh.ca.node_state[:8] = 8
    >>> gh.ca.node_state[9:12] = 8
    >>> gh.ca.node_state[13:16] = 8
    >>> other_rock_nodes = np.array([18, 19, 22, 23, 27, 31])
    >>> gh.ca.node_state[other_rock_nodes] = 8
    >>> (x, z, _) = get_profile_and_soil_thickness(gh.grid, gh.ca.node_state)
    >>> x
    array([ 0.8660254 ,  1.73205081,  2.59807621,  3.46410162,  4.33012702,
            5.19615242])
    >>> z
    array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ])
    >>> p = fit_zero_intercept_line_to_surface(x, z)
    >>> round(1000 * p[1])
    577.0
    >>> round(p[0])
    30.0
    """
    surface_x = surface_x[:,np.newaxis]
    fit_params = np.linalg.lstsq(surface_x, surface_z, rcond=None)
    m = fit_params[0][0]
    S = np.abs(m)
    dip_angle = np.arctan(S)*180./np.pi

    return dip_angle, S, fit_params


def upper_row_nodes(grid):
    """Return IDs of nodes in upper interior row (next to top).
    
    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> grid = HexModelGrid(3, 5, shape='rect', orientation='vert')
    >>> list(upper_row_nodes(grid))
    [5, 6, 7, 8, 9]
    """
    return (np.arange(grid.number_of_node_columns)
            + (grid.number_of_node_rows - 2) * grid.number_of_node_columns)


def calc_length_of_footwall_at_upper_boundary(grid):
    """Calculate and return length over which footwall touches upper edge.
    
    Length is in cells.
    
    Examples
    --------
    >>> from grainhill import GrainHill
    >>> import numpy as np
    >>> gh = GrainHill((3, 5), show_plots=False)
    >>> gh.ca.node_state[:] = 0
    >>> calc_length_of_footwall_at_upper_boundary(gh.grid)
    0.0
    >>> gh.ca.node_state[9] = 8
    >>> int(1000 * calc_length_of_footwall_at_upper_boundary(gh.grid))
    866
    >>> gh.ca.node_state[6] = 8
    >>> int(1000 * calc_length_of_footwall_at_upper_boundary(gh.grid))
    1732
    >>> gh.ca.node_state[8] = 8
    >>> int(1000 * calc_length_of_footwall_at_upper_boundary(gh.grid))
    2598
    """
    upper_row = upper_row_nodes(grid)
    ns = grid.at_node['node_state']
    fw_top_bnd = upper_row[ns[upper_row] > 0]
    if len(fw_top_bnd > 0):
        fw_len = np.amax(grid.x_of_node - np.amin(grid.x_of_node[fw_top_bnd]))
    else:
        fw_len = 0.0
    return fw_len


def main(run_dir, results_basename):

    # INITIALIZE
    
    # Open a file for output of results
    d = datetime.datetime.today()
    today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
    results_file = open(results_basename + today_str + '.csv', 'w')
    results_file.write('Run name,Slip interval,'
                       + 'Weathering rate parameter,Slope angle,'
                       + 'Slope gradient,Intercept,'
                       + 'Height of right column,'
                       + 'Footwall upper bound length ratio\n')

    # RUN
    for d in os.listdir(run_dir):
    
        # Assume if it starts with 'tau' it's a folder containing a run
        if d[0:3] == 'tau':
            
            # Read the grid
            g = load_grid(run_dir + '/' + d + '/' + d + '.grid')

            # Get the profile
            (x, z, _) = get_profile_and_soil_thickness(g, g.at_node['node_state'])
    
            # Fit the profile
            polyparams = np.polyfit(x, z, 1)
            S = np.abs(polyparams[0])
            dip_angle = np.arctan(S)*180./np.pi
            
            # Get the relative length of the footwall's contact with the
            # upper boundary
            fw_len = calc_length_of_footwall_at_upper_boundary(g)
            len_ratio = fw_len / np.amax(g.x_of_node)

            # Figure out the weathering rate and uplift interval from run name
            f = d.find('w')
            tau = 10.0 ** (0.1 * float(d[f-2:f]))
            ww = 10.0 ** -(0.1 * float(d[f+2:]))
    
            # Write the results to our file
            line_to_write = (d + ',' + str(tau) + ',' + str(ww)
                              + ',' + str(dip_angle) + ',' + str(S) + ','
                             + str(polyparams[1]) + ',' + str(z[-2]) + ','
                             + str(len_ratio))
            print(line_to_write)
            results_file.write(line_to_write + '\n')

    # FINALIZE
    results_file.close()


if __name__ == '__main__':

    import doctest
    import sys

    doctest.testmod()
    
    try:
        run_dir = sys.argv[1]
        results_basename = sys.argv[2]
    except:
        print('Need run folder and name for results file on the command line')
        raise

    main(run_dir, results_basename)

