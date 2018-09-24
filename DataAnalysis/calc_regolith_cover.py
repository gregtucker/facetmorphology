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

def calc_rock_and_regolith(grid):
    """Calculate and return fraction of solid-air pairs that have regolith.
    
    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> import numpy as np
    >>> hg = HexModelGrid(6, 6, orientation='vert', shape='rect')
    >>> ns = hg.add_zeros('node', 'node_state', dtype=np.int)
    >>> ns[:18] = 7
    >>> p = calc_rock_and_regolith(hg)
    >>> p[0]
    1.0
    >>> p[1]
    0.0
    >>> p[2]
    9
    >>> p[3]
    0
    >>> p[4]
    9
    >>> ns[13] = 8
    >>> ns[15:17]= 8
    >>> p = calc_rock_and_regolith(hg)
    >>> round(100 * p[0])
    22.0
    >>> round(100 * p[1])
    78.0
    >>> p[2]
    2
    >>> p[3]
    7
    >>> p[4]
    9
    """
    num_rock_air_pairs = 0
    num_reg_air_pairs = 0

    ns = grid.at_node["node_state"]
    for al in grid.active_links:
        t = grid.node_at_link_tail[al]
        h = grid.node_at_link_head[al]
        if (ns[t] + ns[h]) == 7:  # regolith-air pair
            num_reg_air_pairs += 1
        elif (ns[t] + ns[h]) == 8:  # rock-air pair
            num_rock_air_pairs += 1
    num_surface_pairs = num_rock_air_pairs + num_reg_air_pairs
    prop_reg = float(num_reg_air_pairs) / num_surface_pairs
    prop_rock = float(num_rock_air_pairs) / num_surface_pairs

    return (prop_reg, prop_rock, num_reg_air_pairs, num_rock_air_pairs,
            num_surface_pairs)


def main():

    # INITIALIZE
    # Set parameters/variables
    run_dir = '../ModelRuns/SADW2/'
    results_basename = 'regolith_analysis'
    
    # Open a file for output of results
    d = datetime.datetime.today()
    today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
    results_file = open(results_basename + today_str + '.csv', 'w')
    results_file.write('Run name,Disturbance rate parameter,'
                       + 'Weathering rate parameter,Proportion regolith,'
                       + 'Proportion rock,Number regolith-air pairs,'
                       + 'Number rock-air pairs,Number surface pairs,'
                       + 'Total regolith cells,Regolith thickness\n')

    # RUN
    for d in os.listdir(run_dir):

        # Assume if it starts with 'd' it's a folder containing a run
        if d[0] == 'd':

            # Read the grid
            g = load_grid(run_dir + d + '/' + d + '.grid')

            # Get the regolith and rock proportions
            p = calc_rock_and_regolith(g)

            # Get regolith thickness
            num_reg = np.count_nonzero(g.at_node['node_state'] == 7)
            reg_thick = num_reg / (g.number_of_node_columns - 2.0)
    
            # Figure out the weathering and disturbance rates from run name
            f = d.find('w')
            dd = 10.0 ** -(0.1 * float(d[f-2:f]))
            ww = 10.0 ** -(0.1 * float(d[f+2:]))
    
            # Write the results to our file
            line_to_write = (d + ',' + str(dd) + ',' + str(ww) + ','
                             + str(p[0]) + ',' + str(p[1]) + ','
                             + str(p[2]) + ',' + str(p[3]) + ','
                             + str(p[4]) + ',' + str(num_reg) + ','
                             + str(reg_thick))
            print(line_to_write)
            results_file.write(line_to_write + '\n')


    # FINALIZE
    results_file.close()


if __name__ == '__main__':

    import doctest

    doctest.testmod()
    main()
