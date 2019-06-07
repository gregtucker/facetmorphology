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


ROOT3 = np.sqrt(3.0)
DELTA = 0.5


# FUNCTIONS

def calc_fractional_soil_cover(grid):
    """Calculate and return fractional soil versus rock cover."""
    num_soil_air_faces = 0.0
    num_rock_air_faces = 0.0

    node_state = grid.at_node['node_state']

    for link in range(grid.number_of_links):
        tail = grid.node_at_link_tail[link]
        head = grid.node_at_link_head[link]
        if node_state[tail] == 0:  # if tail is air, see if head is rock/sed
            if node_state[head] == 7:
                num_soil_air_faces += 1
            elif node_state[head] == 8:
                num_rock_air_faces += 1
        elif node_state[head] == 0:  # if head is air, see if tail is rock/sed
            if node_state[tail] == 7:
                num_soil_air_faces += 1
            elif node_state[tail] == 8:
                num_rock_air_faces += 1

    total_surf_faces = num_soil_air_faces + num_rock_air_faces
    frac_rock = num_rock_air_faces / total_surf_faces
    frac_soil = num_soil_air_faces / total_surf_faces
    print('Total number of surface faces: ' + str(total_surf_faces))
    print('Number of soil-air faces: ' + str(num_soil_air_faces))
    print('Number of rock-air faces: ' + str(num_rock_air_faces))
    print('Percent rock-air faces: ' + str(100.0 * frac_rock))
    print('Percent soil-air faces: ' + str(100.0 * frac_soil))
    return frac_soil


def get_profile_and_soil_thickness(grid, node_state):
    """Calculate and return profiles of elevation and soil thickness.

    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> import numpy as np
    >>> hg = HexModelGrid(4, 6, orientation='vert', node_layout='rect')
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
    >>> grid = HexModelGrid(3, 5, node_layout='rect', orientation='vert')
    >>> list(upper_row_nodes(grid))
    [5, 6, 7, 8, 9]
    """
    return (np.arange(grid.number_of_node_columns)
            + (grid.number_of_node_rows - 2) * grid.number_of_node_columns)


def calc_length_of_footwall_at_upper_boundary(grid):
    """Calculate and return length over which footwall touches upper edge.

    Length is in cells.

    Notes
    -----
    We apply a small geometric correction, derived as follows. Assume the top
    grid edge is 110 grid units above the fault trace. Then, 110 tan 30o equals
    the horizontal distance from left side to fault plane intercept with
    the top edge: ~63.509. The grid width (from middle of left cells to middle
    of right cells) is 80 * (1/2) * sqrt(3) = ~69.282. The difference is
    ~5.7735. Meanwhile, in the discretized grid, when there is no erosion at
    the top, the length is calculated as ~6.0622. This reflects a 
    grid-resolution-induced error, which we correct by simply applying the
    difference: 0.2887.

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
    577
    >>> gh.ca.node_state[6] = 8
    >>> int(1000 * calc_length_of_footwall_at_upper_boundary(gh.grid))
    1443
    >>> gh.ca.node_state[8] = 8
    >>> int(1000 * calc_length_of_footwall_at_upper_boundary(gh.grid))
    2309
    """
    upper_row = upper_row_nodes(grid)
    ns = grid.at_node['node_state']
    fw_top_bnd = upper_row[ns[upper_row] > 0]
    if len(fw_top_bnd > 0):
        fw_len = np.amax(grid.x_of_node - np.amin(grid.x_of_node[fw_top_bnd]))
        fw_len -= 0.2887  # geometric correction
        fw_len = max(fw_len, 0.0)
    else:
        fw_len = 0.0
    return fw_len


def count_solid_cells_per_inner_column(grid):
    """Returns number of rock or regolith cells per column.
    
    Also returns the number of the right-most inner column that contains at
    least one air cell.
    
    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> grid = HexModelGrid(shape=(8, 7), node_layout='rect',
    ...                     orientation='vert')
    >>> ns = grid.add_zeros('node', 'node_state', dtype=np.int)
    >>> ns[:7] = 8
    >>> count_solid_cells_per_inner_column(grid)  # 8x7, 0 degrees
    array([1, 1, 1, 1])
    >>> ns[:7] = 1
    >>> ns[8:11] = 1
    >>> ns[12:14] = 1
    >>> ns[16:18] = 1
    >>> ns[20] = 1
    >>> ns[24] = 1
    >>> count_solid_cells_per_inner_column(grid)  # 8x7, 30 degrees
    array([2, 2, 3, 3])
    >>> ns[11] = 1
    >>> ns[15] = 1
    >>> ns[19] = 1
    >>> ns[22:24] = 1
    >>> ns[26:28] = 1
    >>> ns[30:32] = 1
    >>> ns[33:35] = 1
    >>> ns[37:39] = 1
    >>> ns[41] = 1
    >>> ns[44:46] = 1
    >>> ns[48] = 1
    >>> ns[52] = 1
    >>> ns[55] = 1
    >>> count_solid_cells_per_inner_column(grid)  # 8x7, 60 degrees
    array([4, 5])
    """
    ns = grid.at_node['node_state']
    num_solid = np.ones(grid.number_of_node_columns - 2, dtype=np.int)
    last_col_with_air = 0
    for n in range(grid.number_of_nodes):
        if not grid.node_is_boundary(n):
            (r, c) = grid.node_row_and_column(n)
            if ns[n] != 0:
                num_solid[c-1] += 1
            elif c > last_col_with_air:
                last_col_with_air = c
    return num_solid[1:last_col_with_air]  # skip col 1 and any that are "full"



def calc_ero_rate_from_topo(grid, delta, tau):
    """Calculate the average erosion rate from the topography.

    Notes
    -----
    The erosion rate is averaged over each column in the grid. For each column,
    the cumulative erosion depth in cells equals the difference between the
    number of rock/soil cells that would be present if there were no erosion,
    and the number actually present. (The number could be positive, if there
    were deposition). To do this, we need two numbers: the height of the column
    without erosion, and the duration that the column has been exposed to
    erosion.
        The first number comes from geometry. Let the fault be a 60o-dipping
    feature that crosses point (Xf0, 0), where Xf0 is the x-coordinate of the
    fault's zero intercept. Let Xc denote the x-coordinate of the column
    center. The height of the fault plane is then Yf = Xc tan 60.
    
    Examples
    --------
    >>> from landlab import HexModelGrid
    >>> grid = HexModelGrid(shape=(8, 7), node_layout='rect',
    ...                     orientation='vert')
    >>> ns = grid.add_zeros('node', 'node_state', dtype=np.int)
    >>> (ero_mean, ero_col) = calc_ero_rate_from_topo(grid, 1.0, 100.0)
    >>> int(1000 * ero_mean)
    15
    >>> np.ceil(1000 * ero_col)
    array([ 15.,  15.,  15.,  15.])
    >>> ns[:7] = 1
    >>> ns[8:11] = 1
    >>> ns[12:14] = 1
    >>> ns[16:18] = 1
    >>> ns[20] = 1
    >>> ns[24] = 1
    >>> (ero_mean, ero_col) = calc_ero_rate_from_topo(grid, 1.0, 100.0)
    >>> int(1000 * ero_mean)
    10
    >>> np.ceil(1000 * ero_col)
    array([ 10.,  10.,  10.,  10.])
    >>> ns[11] = 1
    >>> ns[15] = 1
    >>> ns[19] = 1
    >>> ns[22:24] = 1
    >>> ns[26:28] = 1
    >>> ns[30:32] = 1
    >>> ns[33:35] = 1
    >>> ns[37:39] = 1
    >>> ns[41] = 1
    >>> ns[44:46] = 1
    >>> ns[48] = 1
    >>> ns[52] = 1
    >>> ns[55] = 1
    >>> (ero_mean, ero_col) = calc_ero_rate_from_topo(grid, 1.0, 100.0)
    >>> np.abs(ero_mean) < 1.0e-18
    True
    >>> np.abs(ero_col) < 1.0e-17
    array([ True,  True], dtype=bool)
    """
    nsol = count_solid_cells_per_inner_column(grid)
    c = np.arange(2, len(nsol) + 2)  # column numbers
    x = 0.5 * ROOT3 * c  # horizontal position
    height = nsol - 1.0 / (1.0 + (c % 2))  # column height
    height[nsol < 2] = 0.0   # height is zero if one or fewer cells
    sliprate = ROOT3 * delta / tau
    ero_rate_per_col = 0.5 * sliprate * (ROOT3 - height / x)
    return (np.mean(ero_rate_per_col), ero_rate_per_col)


def main(run_dir, results_basename):

    # INITIALIZE
    
    # Open a file for output of results
    d = datetime.datetime.today()
    today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
    results_file = open(results_basename + today_str + '.csv', 'w')
    results_file.write('Run name,Slip interval,'
                       + 'Weathering rate parameter,Slope angle,'
                       + 'Slope gradient,Intercept,'
                       + 'Average erosion rate'
                       + 'Fractional soil cover\n')

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
            
            # Figure out the weathering rate and uplift interval from run name
            f = d.find('w')
            tau = 10.0 ** (0.1 * float(d[f-2:f]))
            ww = 10.0 ** -(0.1 * float(d[f+2:]))
    
            # Calculate erosion rate
            (ero_mean, ero_col) = calc_ero_rate_from_topo(g, DELTA, tau)

            # Calculate fractional soil cover thickness
            frac_soil = calc_fractional_soil_cover(g)

            # Write the results to our file
            line_to_write = (d + ',' + str(tau) + ',' + str(ww)
                              + ',' + str(dip_angle) + ',' + str(S) + ','
                             + str(polyparams[1]) + ',' + str(ero_mean) + ','
                             + str(frac_soil))
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

