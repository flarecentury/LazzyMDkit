import numpy as np
import pandas as pd
from .PlotCustomizer import *


def readthermolog(logfile, timestep=0.5, plot=True, a=0, b=-1, skip=1):
    global title, check, check0, NU_param
    with open(logfile, encoding="utf8", errors='ignore') as log:
        lmplog = log.readlines()
    print(logfile)

    wflag = False
    isdata = False
    newline = []
    result = []
    counter = 0
    index = 'X'
    # replica = [] # 对应下方replica代码

    for line in lmplog:  # 按行读入文件，此时line的type是str
        if line.startswith("Loop"):  # 重置写标记
            wflag = False
            isdata = False
            index = 'X'
        if "Step" in line:  # 检验是否到了要写入的内容
            counter += 1
            check0 = counter
            wflag = True
            result = [[j] for j in line.split()]
            NU_param = len(line.split())
            title = line.split()
            index = 0
            # isdata == False
            continue
        if wflag:
            # print(index)
            check = counter
            index += 1
            if index == 2:  # 设置为1则从第一行开始，设置为2则从第二行数据开始，但有点小问题，会缺少一些数据
                isdata = True
        if isdata and index != 'X' and check == check0:
            if "ERROR" not in line and 'Last c' not in line and 'WARNING:' not in line:
                # print(line)
                if line.split()[0].isnumeric and len(line.split()) == NU_param:  # 防止数据还未生成完全的row被加入结果
                    newline.append(line)
                    # remove replica ####
                    # if line not in newline:
                    #     newline.append(line)
                    # else:
                    #     replica.append(line)
                else:
                    print('Found incomplete line:', line)
    alllines = []
    for line in newline:
        linetolist = line.split()
        alllines.append(linetolist)
        for ncol in range(len(linetolist)):
            result[ncol].append(linetolist[ncol])

    float_result = []
    print('ncols: ', len(result))
    for i in result:
        i.pop(0)  # 去除i list的第一个数据 title
        i = list(np.float_(i))  # 转换剩余数据为np array
        float_result.append(i)
    print('loop: ', counter)  # loop数目

    dt = float(timestep)  # fs
    times = [i * dt / 1000 for i in float_result[0]]
    print('simulated frames (duplica results are includes): ', len(result[0]))
    print('simulated time: ', round(float(times[-1]), 2), ' ps')

    title = '\t'.join(title) + '\n'
    titles = title.split()

    datas = []
    for line in alllines:
        d = '\t'.join(line) + '\n'
        datas.append([float(i) for i in d.split()])
    columns = titles
    df = pd.DataFrame.from_records(datas, columns=columns)
    df['Step'] = df['Step'].astype(int)
    df = df.drop_duplicates()

    title_dic = {}
    for idx, t in enumerate(titles):
        title_dic[idx] = t
    print(title_dic)

    if plot:
        c_reaxdict = {'v_eb': 'bond energy', 'v_ea': 'atom energy', 'v_elp': 'lone-pair energy',
                      'v_emol': 'molecule energy (always 0.0)', 'v_ev': 'valence angle energy',
                      'v_epen': 'double-bond valence angle penalty', 'v_ecoa': 'valence angle conjugation energy',
                      'v_ehb': 'hydrogen bond energy', 'v_et': 'torsion energy', 'v_eco': 'conjugation energy',
                      'v_ew': 'van der Waals energy', 'v_ep': 'Coulomb energy',
                      'v_efi': 'electric field energy (always 0.0)', 'v_eqeq': 'charge equilibration energy'}
        c_reax = list(c_reaxdict.keys())

        for x_picked_col in [0, 1]:  # xaxis数据源
            fig = plt.figure(figsize=(18, 18), dpi=200)
            for y_picked_col in range(0, len(columns)):
                if x_picked_col != y_picked_col:
                    subplotindex = y_picked_col
                    ax3 = fig.add_subplot(5, 6, 1 + subplotindex)
                    if x_picked_col == 0:  # turn step to time
                        ax3.plot(times[a:b:skip], df[columns[y_picked_col]][a:b:skip],
                                 label=columns[y_picked_col])  # 每隔10点取数据
                        if columns[y_picked_col] in c_reax:
                            ax3.set_title(c_reaxdict[columns[y_picked_col]], fontsize=7)
                        # ax3.set_title('Fig. ' + str(subplotindex) + '  ' + y_picked_col] + ' VS. ' + 'time (ps)')
                        ax3.set_xlabel('time (ps)')
                    else:
                        ax3.plot(df[columns[x_picked_col]][a:b:skip], df[columns[y_picked_col]][a:b:skip],
                                 label=columns[y_picked_col])  # 每隔10点取数据
                        ax3.set_xlabel(columns[x_picked_col])
                        if columns[y_picked_col] in c_reax:
                            ax3.set_title(c_reaxdict[columns[y_picked_col]], fontsize=7)
                        # ax3.set_ylabel(y_picked_col])
                        # ax3.set_title('Fig. ' + str(subplotindex) + '  ' + y_picked_col] + ' VS. ' + x_picked_col])
                    ax3.legend()
                    # plt.subplots_adjust(top=2)
                    plt.subplots_adjust(hspace=0.5)
                else:
                    print('skipping meaningless plot: ' + str(columns[y_picked_col]) + ' vs. ' + str(
                        columns[y_picked_col]))
            plt.show()
    return df[a:b:skip]


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
