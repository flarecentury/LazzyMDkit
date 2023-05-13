def readlammpsdata(inputdata='lmp.S5.0--1_EM+Annealling.lmp', Mname=None, withxyz=False):
    """
    withxyz #使用xyz文件作为缓存文件
    """
    if Mname is None:
        Mname = ['Al', 'O', 'Al']
    print(f'visualizing {inputdata}')
    F = open(inputdata)
    f = F.readlines()

    Positions = []
    Mmass = []
    dims = []
    Matomtypes = None
    for i in f:
        # print(i)
        x = i.split()
        if len(x) == 9 and x[0].isnumeric():
            # lammps data file (charge)
            # atom-ID atom-type q
            posi = [x[0], x[1], str(float(x[3])), str(float(x[4])), str(float(x[5]))]
            # posi = [atom-ID atom-type x y z]
            # i = '\t'.join(posi) + '\n'
            # FF.append(i)
            # atom-ID atom-type q -> zatom-ID atom-type q x y z
            Positions.append(posi)

        elif len(x) == 2 and x[0].isnumeric():
            try:
                type(float(x[-1]))
                Mmass.append(i.split()[1])
                Matomtypes = int(x[0])
            except:
                yyy = 1

        elif len(x) == 4:
            if str(x[-1]).endswith('hi'):  # dimansion
                i = i.strip('\n')
                dims.append(i)
        else:
            yyy = 1

    if len(Positions) == 0:  # 防止是vmd转换的data文件
        for i in f:
            x = i.split()
            if len(x) == 6 and x[0].isnumeric():
                posi = [x[0], x[1], str(float(x[3])), str(float(x[4])), str(float(x[5]))]
                Positions.append(posi)
    F.close()

    # 计算box边长
    dim_edges = []
    for i, dim in enumerate(dims):
        dim_i = [float(i) for i in dims[0].split()[:2]]
        dim_i_edges = dim_i[-1] - dim_i[0]
        dim_edges.append(dim_i_edges)
    dim_edges = ['{:0>7.3f}'.format(x) for x in dim_edges]

    # print(Positions)
    print(f"Mmass {Mmass} \nMname:{Mname} \ndims:{dims} \nMatomtypes:{Matomtypes}")
    # 将获取的atom 坐标信息按照原子id 从小到大排序
    # posi = [atom-ID atom-type x y z]
    Positions = natsorted(Positions)

    posi_types = []
    for i in range(Matomtypes):
        type_i_posi = []
        posi_types.append(type_i_posi)
        element = Mname[i]
        for k in Positions:
            if k[1] == str(i + 1):
                k[1] = element  # 替换原子的type 为原子名称
                k.pop(0)  # 删除atom id
                type_i_posi.append(k)  # k=[elements,x,y,z]
        posi_types[i] = type_i_posi

    ABCDE = string.ascii_uppercase
    if withxyz:
        New = '/tmp/xxxx.xyz'
        N = open(New, 'w')
        N.write(str(len(Positions)) + '\n\n')
        print('总原子数', len(Positions))
        # 按原子顺序，依次写入 elements x y z到文件
        for index, type_i_posi in enumerate(posi_types):
            print('type', index, len(type_i_posi))
            for i in type_i_posi:
                N.write('  '.join(i) + '\n')
        N.close()

        u = mda.Universe(New)
        print('加入chainID')
        indices = 0
        u.add_TopologyAttr('chainID')
        for type_i, type_i_posi in enumerate(posi_types):
            stri = ' '.join([str(i) for i in list(range(indices, indices + len(type_i_posi)))])
            indices += len(type_i_posi)
            # u.select_atoms('index '+stri).chainID=[ABCDE[i]]*len(u.select_atoms('index '+stri))
            u.select_atoms('index ' + stri).chainIDs = ABCDE[type_i]

    else:
        print('write with pdb')
        New = '/tmp/xxxx.pdb'
        N = open(New, 'w')
        N.write('TITLE     MDANALYSIS FRAMES FROM 0, STEP 1: Created by PDBWriter\n')
        # 可从dim变量中获取box尺寸
        #N.write('CRYST1  180.000  180.000  180.000  90.00  90.00  90.00 P 1           1\n')
        N.write(f'CRYST1  {str(dim_edges[0])}  {str(dim_edges[0])}  {str(dim_edges[0])}  90.00  90.00  90.00 P 1           1\n')
        # N.write(f"CRYST1  {str(dim_edges[0])}  {str(dim_edges[1])}  {str(dim_edges[2])}  90.00  90.00  90.00 P 1           1\n")
        N.write('MODEL        1\n')

        print('总原子数', len(Positions))
        # 按原子顺序，依次写入 elements x y z到文件
        indices = 0
        for type_i, type_i_posi in enumerate(posi_types):
            print('type', type_i, len(type_i_posi))
            for idx, pos in enumerate(type_i_posi):
                fmt = {
                    'ATOM': (
                        "ATOM  {serial:5d} {name:<4s}{altLoc:<1s}{resName:<4s}"
                        "{chainID:1s}{resSeq:4d}{iCode:1s}"
                        "   {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{occupancy:6.2f}"
                        "{tempFactor:6.2f}      {segID:<4s}{element:>2s}{charge:2s}\n"),
                    'REMARK': "REMARK     {0}\n",
                    'CONECT': "CONECT{0}\n"
                }

                vals = {'serial': indices + idx + 1, 'name': str(Mname[type_i]), 'chainID': str(ABCDE[type_i]),
                        'pos': [float(p) for p in pos[1:]], 'charge': '0', 'altLoc': ' ', 'resName': 'UNK',
                        'resSeq': 1,
                        'iCode': ' ', 'occupancy': 1.0, 'tempFactor': 0.0, 'segID': ' ', 'element': ' '}
                # --------------------

                # print(fmt['ATOM'])
                line_i = fmt['ATOM'].format(**vals)
                N.write(line_i)

            indices += len(type_i_posi)
        N.write('ENDMDL\n')
        N.write('END\n')
        N.close()
        # ensure the position is not out of boundary
        u = mda.Universe(New)
        # determine boundry
        print('修正Al原子质量')
        u.select_atoms('name Al').masses = [26.98153860] * len(u.select_atoms('name Al'))

    atoms = u.select_atoms('all')
    dist_arr = distances.distance_array(atoms.positions, atoms.centroid(), box=u.dimensions)
    dist_arr_self = distances.self_distance_array(atoms.positions, box=u.dimensions)
    print(f"共计算{len(dist_arr)}个原子距box center的距离，最小是{dist_arr.min()}，最大是{dist_arr.max()}")
    print(f"共计算{len(dist_arr_self)}对原子间距离，最小距离为{dist_arr_self.min()},最大距离为{dist_arr_self.max()}")
    v = nv.show_mdanalysis(atoms)
    return u, v

def visualize_mda_universe_via_plotly(u):
    import MDAnalysis as mda
    import plotly.graph_objects as go

    # get the coordinates of all atoms in the simulation
    positions = u.atoms.positions
    box_dims = u.dimensions[:3]

    # create a trace with the positions
    trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.7
        )
    )

    layout = go.Layout(width=800,height=800)
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    return fig