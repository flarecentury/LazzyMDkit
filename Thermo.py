import numpy as np
import pandas as pd
from LazzyMDkit.PlotCustomizer import *


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def read_thermo_log_to_df(logfile, timestep=0.5):
    """
    Reads a LAMMPS thermo log file and converts it to a DataFrame.

    Parameters:
    logfile (str): Path to the log file.
    timestep (float): Timestep in femtoseconds used in the simulation.

    Returns:
    tuple: Contains the column titles, times, and DataFrame of the log file.
    """
    with open(logfile, encoding="utf8", errors='ignore') as log:
        log_lines = log.readlines()

    is_data_section = False
    data_lines = []
    titles = []
    num_columns = None

    for line in log_lines:
        if line.startswith("Loop"):
            is_data_section = False
            continue
            
        if "Step" in line and not is_data_section:
            titles = line.split()
            num_columns = len(titles)
            is_data_section = True
            continue

        if is_data_section and "ERROR" not in line and 'Last c' not in line and 'WARNING:' not in line:
            split_line = line.split()
            if split_line and split_line[0].isdigit() and len(split_line) == num_columns: # Check if split_line is not empty before accessing its elements
                # Check if all elements can be converted to float
                if all(is_float(elem) for elem in split_line):
                    data_lines.append(split_line)
                else:
                    print('Skipping invalid data line:', line)
            else:
                print('Found incomplete line:', line)
        if "Total wall time" in line:
            print(f'Job done, {line}')
    df = pd.DataFrame(data_lines, columns=titles).drop_duplicates(subset=['Step']).astype(float)
    df['Step'] = df['Step'].astype(int)
    simulated_times = [i * timestep / 1000 for i in df['Step'].to_list()]

    return titles, simulated_times, df

def plot_thermo_data(df, a=0, b=-1, interval=1, dpi=200):
    """
    Plots data from the DataFrame using subplots.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - a, b, interval (int): Parameters for slicing the data.
    - figsize (tuple): Size of the figure.
    - dpi (int): Resolution of the plot.
    """
    c_reaxdict = {
        #... your dictionary here ...
    }
    
    subplots_per_row = 5
    x_cols = []
    for col in list(df.columns):
        if col in ['Step','Time']:
            x_cols.append(col)
    other_columns = [col for col in df.columns if col not in x_cols]
    print(f'Detected cols: {x_cols}')
    print(f'Data cols: {other_columns}')
    # Define the layout of the subplots
    nrows = len(other_columns) // subplots_per_row + bool(len(other_columns) % subplots_per_row)
    ncols = min(subplots_per_row, len(other_columns))
    figsize = (4 * ncols, 4 * nrows)
    
    # Create separate figures
    print('Data points:',len(df[x_cols[0]][::interval]))
    for x_col in x_cols:
        fig, axes = plt.subplots(nrows, subplots_per_row, figsize=figsize, dpi=dpi, squeeze=False)
        # fig.suptitle(f'{x_col} vs Other Columns', fontsize=16)
        for i, col in enumerate(other_columns):
            row, col_idx = divmod(i, subplots_per_row)
            ax = axes[row, col_idx]

            ax.plot(df[x_col][::interval][a:b:], df[col][::interval][a:b:], label=col)
            ax.set_xlabel(x_col)
            ax.set_ylabel(col)
            title = c_reaxdict.get(col, col)
            ax.set_title(title)
            ax.legend()
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top padding to accommodate the suptitle
        plt.show()

def readthermolog(logfile, timestep=0.5, plot=True, a=0, b=-1, skip=1):
    """
    Reads single or multiple LAMMPS thermo log files.

    Parameters:
    logfile (str or list): Path(s) to the log file(s).
    timestep (float): Timestep in femtoseconds used in the simulation.
    """
    if isinstance(logfile, list):
        print('Processing multiple log files...')
        df_list = []
        for logf in logfile:
            _, _, df = read_thermo_log_to_df(logf, timestep)  # Extract only the DataFrame
            df_list.append(df)

        combined_df = pd.concat(df_list, axis=0).drop_duplicates(subset=['Step'])
    else:
        print('Processing a single log file...')
        _, _, combined_df = read_thermo_log_to_df(logfile, timestep)  # Extract only the DataFrame
    if plot:
        plot_thermo_data(combined_df, a=a, b=b, interval=skip)
        
    return combined_df

def checklog(Lastlog=0, logfile=''):
    with open(logfile, encoding="utf8", errors='ignore') as log:
        f = log.readlines()
        print(logfile)
    print('lmplog 行数: ', len(f))
    print('================  WARNINGs ================')
    for i in f:
        if 'ERROR' in i:
            print(i)
        if 'Last c' in i:
            print(i)
        if 'WARNING:' in i:
            if 'Kokkos' not in i:
                print(i)
            print(i)
        if '@@@@@@@' in i:
            print(i)
    print('================  Last logs ================')
    try:
        for i in range(len(f) - Lastlog, len(f)):
            if 'Kokkos::' not in f[i]:
                print(f[i])
        print('===' * 25)
    except:
        for i in range(len(f)):
            if 'Kokkos::' not in f[i]:
                print(f[i])
        print('===' * 25)
