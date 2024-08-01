import MDAnalysis as mda
import nglview as nv
from natsort import natsorted
from MDAnalysis.analysis import distances
import string
from pymatgen.io.vasp import Poscar
from pymatgen.io.lammps.data import LammpsData

def readlammpsdata_to_universe(lammpsdata, atom_style='id type x y z', names=None):
    '''
    return a mda universe object
    '''
    # Load the LAMMPS data file
    u = mda.Universe(lammpsdata, atom_style=atom_style)

    # Add necessary topology attributes
    u.add_TopologyAttr('name')
    u.add_TopologyAttr('resnames')
    u.add_TopologyAttr('chainIDs')

    # Get the unique atom types
    types = list(set(u.atoms.types))
    types.sort()

    # Assign names and chainIDs based on atom types
    if names is None:
        # If names are not provided, use default names
        names = [f'Type{i}' for i in types]
    else:
        # Ensure the number of names matches the number of atom types
        assert len(names) == len(types), "Number of names should match the number of atom types"

    chainIDs = [chr(ord('A') + i) for i in range(len(types))]

    for atom_type, name, chainID in zip(types, names, chainIDs):
        sel = u.select_atoms(f'type {atom_type}')
        sel.names = [name] * len(sel)
        sel.chainIDs = [chainID] * len(sel)

    return u

def readlammpsdata(inputdata='lmp.S5.0--1_EM+Annealling.lmp', Mname=None, withxyz=False, charge=True):
    """
    withxyz #使用xyz文件作为缓存文件
    data是否是chage style, 见lammps文档
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

    def read_charge_style_data(f):
        for i in f:
            # print(i)
            x = i.split()
            if len(x) == 9 and x[0].isnumeric():
                # lammps data file (charge)
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
        return Positions,Mmass,dims,Matomtypes

    def read_atoms_style_data(f):
        for i in f:
            # print(i)
            x = i.split()
            if len(x) == 8 and x[0].isnumeric():
                
                # lammps data file (charge)
                # atom-ID atom-type q
                posi = [x[0], x[1], str(float(x[2])), str(float(x[3])), str(float(x[4]))]
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
                    posi = [x[0], x[1], str(float(x[2])), str(float(x[3])), str(float(x[4]))]
                    Positions.append(posi)
        F.close()
        return Positions,Mmass,dims,Matomtypes

    if charge:
        print('data file style: charge')
        Positions,Mmass,dims,Matomtypes = read_charge_style_data(f)
    else:
        print('data file style: atomic')
        Positions,Mmass,dims,Matomtypes = read_atoms_style_data(f)

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
            if len(type_i_posi) > 0:
                u.select_atoms('index ' + stri).chainIDs = ABCDE[type_i]
            else:
                print('warning! current atom type defined in the head of lmp file is not in the system:',type_i)

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
                    'CONECT': "CONECT{0}\n"}

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

def rewrite_mda_lammpsdata(lmpfiles,zerocharge=False): ### 有charge 针对DCD文件导出
    '''
    
    '''
    F = open(lmpfiles)
    f = F.readlines()
    FF = []
    charged = False
    noncharged = False
    for i in f:
        if len(i.split()) <= 4: ## 非数据内容
            if i.startswith('Atoms'):
                i = 'Atoms # charge\n'
            if i.endswith('xhi\n') or i.endswith('yhi\n') or i.endswith('zhi\n'):
                k = i.split()
                dim = float(k[1])/2
                i = str(-dim) +' '+ str(dim)+' '+str(k[-2])+' '+str(k[-1])+'\n'
            else:
                i = i
            FF.append(i)
        elif len(i.split()) == 7: #deal with chargeed atom
            x = i.split()
            del x[1] ##################################################### charge 做了修改 保持不变！！！！！！！！！！！！！！！！！！！！！
            if zerocharge:
                x[2] = str('0.000000')
            i = '\t'.join(x) + '\n'
            FF.append(i)
            charged = True
        elif len(i.split()) == 6: # deal with non-charge atom
            x = i.split()
            del x[1]
            x.insert(2, '0.000000')
            i = '\t'.join(x) + '\n'
            FF.append(i)
            noncharged = True
            charged = False
        else:
            i = i
            FF.append(i)
    if charged and not zerocharge:
        print('Rewrite charge data')
        MyFile=open(lmpfiles+'_withcharge.lmp','w')
    if charged and zerocharge:
        print('Rewrite charge data but zerolize_charge')
        MyFile=open(lmpfiles+'_withzerocharge.lmp','w')
    if noncharged:
        print('Rewrite non-charge data')
        MyFile=open(lmpfiles+'_withoutcharge.lmp','w')
    MyFile.writelines(FF)
    MyFile.close()

def merge_lammps_data(Master,Slave,New):
    ## read master
    M = open(Master)
    f = M.readlines()
    Mvolicity = []
    Matom = []
    Mmass = []
    dims = []
    for i in f:
        # print(i)
        x = i.split()
        if len(x) >= 6 and x[0].isnumeric():
            x = x
            i = ' '.join(x) + '\n'
            # FF.append(i)
            Matom.append(i.split())

        elif len(x) == 2 and x[0].isnumeric():
            try:
                type(float(x[-1]))
                Mmass.append(i.split()[1])
                Matomtypes = int(x[0])
            except:
                yyy=1

        elif len(x) == 4:
            if x[0].isnumeric():
                Mvolicity.append(i.split())
            if str(x[-1]).endswith('hi'): ## dimansion
                dims.append(i)
        else:
            yyy = 1
    M.close()

    ##### read slave
    S = open(Slave)
    f = S.readlines()
    Svolicity = []
    Satom = []
    Smass = []
    for i in f:  ## isnumeric() 是否由数字组成
        # print(i)
        x = i.split()
        if len(x) >= 6 and x[0].isnumeric():
            x = x
            i = ' '.join(x) + '\n'
            # FF.append(i)
            Satom.append(i.split())
        elif len(x) == 2 and x[0].isnumeric():
            try:
                type(float(x[-1]))
                Smass.append(i.split()[1])
                Satomtypes = int(x[0])
            except:
                yyy=1

        elif len(x) == 4 and x[0].isnumeric():
            Svolicity.append(i.split())
        else:
            yyy =1
    S.close()

    try:
        print(Svolicity[0])
    except:
        print('Slave has no volicity')

    totaltypes = Matomtypes + Satomtypes
    masses = Mmass + Smass

    ### add missed data
    ##
    for i in range(0,len(Satom)):
        Satomindex = i+1+len(Matom)
        Satomtype = int(Satom[i][1]) + Matomtypes
        Satom[i][1] = str(Satomtype)
        Satom[i][0] = str(Satomindex)
        Satom[i].insert(6,'0')
        Satom[i].insert(7,'0')
        Satom[i].insert(8,'0')
        Svolicity.append([str(Satomindex), '0.000000', '0.000000', '0.000000'])

    ## write to file
    N = open(New,'w')
    N.write('LAMMPS data file via MDAnalysis\n')
    N.write('\n')
    N.write('{:>12d}  atoms\n'.format(len(Matom)+len(Satom)))
    N.write('{:>12d}  atom types\n'.format(totaltypes))
    N.write('\n')
    N.write(dims[0])
    N.write(dims[1])
    N.write(dims[2])
    N.write('\n')
    N.write('Masses\n')
    N.write('\n')

    for i in range(totaltypes):
        N.write(str(i+1)+'\t'+masses[i]+'\n')

    N.write('\n')
    N.write('Atoms # charge\n')
    N.write('\n')


    Matom = natsorted(Matom)
    Satom = natsorted(Satom)
    Mvolicity = natsorted(Mvolicity)
    Svolicity = natsorted(Svolicity)


    print('Write Matom: ',len(Matom))
    for i in range(len(Matom)):
        item = '\t'.join(Matom[i])+'\n'
        N.write(item)

    print('Write Satom: ',len(Satom))
    for i in range(len(Satom)):
        item = '\t'.join(Satom[i])+'\n'
        N.write(item)

    N.write('\n')
    N.write('Velocities\n')
    N.write('\n')

    print('Write Mvolicity: ',len(Mvolicity))
    for i in range(len(Mvolicity)):
        item = '\t'.join(Mvolicity[i])+'\n'
        N.write(item)

    print('Write Svolicity: ',len(Svolicity))
    for i in range(len(Svolicity)):
        item = '\t'.join(Svolicity[i])+'\n'
        N.write(item)
    print('done!')
    N.close()

def lmpdata_formater(input, origin_style='ID type charge x y z', target_style='ID type x y z'):
    output = 'converted_' + input
    # Split the origin and target styles into lists
    origin_fields = origin_style.split()
    target_fields = target_style.split()

    # Create a dictionary to map field names to their indices in the origin style
    origin_field_indices = {field: index for index, field in enumerate(origin_fields)}

    # Open the input file for reading and the output file for writing
    with open(input, 'r') as infile, open(output, 'w') as outfile:
        # Flag to track if we are in the Atoms section
        in_atoms_section = False

        # Iterate over each line in the input file
        for line in infile:
            # Check if we are entering the Atoms section
            if line.startswith('Atoms'):
                in_atoms_section = True
                outfile.write(line)
                continue

            # If we are in the Atoms section, reformat the line
            if in_atoms_section:
                # Split the line into fields based on whitespace
                fields = line.split()

                # Extract the required fields based on the target style
                target_values = []
                for field in target_fields:
                    if field in origin_field_indices and origin_field_indices[field] < len(fields):
                        target_values.append(fields[origin_field_indices[field]])
                    else:
                        target_values.append('')  # Append an empty string if the field is missing

                # Write the reformatted line to the output file
                outfile.write(' '.join(target_values) + '\n')
            else:
                # If we are not in the Atoms section, write the line as is
                outfile.write(line)

    print(f"Reformatted data file written to: {output}")


def poscar_to_lmpdata(input = 'Al2O3.poscar'):
    output = input + '.data'
    '''
    lammpsdata output: ID type charge x y z
    '''
    # Load the structure from the CIF file
    poscar = Poscar.from_file(input)
    structure = poscar.structure
    # Create a LammpsData object from the structure
    lammps_data = LammpsData.from_structure(structure)
    # Write the LAMMPS data file
    lammps_data.write_file(output)
    
def cif_to_lmpdata(input = 'Al2O3.poscar'):
    output = input + '.data'
    '''
    lammpsdata output: ID type charge x y z
    '''
    # Load the structure from the CIF file
    parser = CifParser('path/to/your/structure.cif')
    structure = parser.get_structures()[0]
    # Create a LammpsData object from the structure
    lammps_data = LammpsData.from_structure(structure)
    # Write the LAMMPS data file
    lammps_data.write_file(output)
