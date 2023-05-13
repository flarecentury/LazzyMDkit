import string
import uuid

import MDAnalysis as mda
import nglview as nv
import numpy as np
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from natsort import natsorted


# class A_TrajTools:
def addpresentation(v):
    v.clear()
    v.clear_representations()
    atom_selections = ''
    v.add_representation('ball')
    v.add_representation('spacefill', selection=':A' + atom_selections, color='silver', radius='0.8')  # ## atom
    # name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    v.add_representation('spacefill', selection=':B' + atom_selections, color='red',
                         radius='0.8')  # atom name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    v.add_representation('spacefill', selection=':C' + atom_selections, color='black',
                         radius='0.8')  # atom name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    v.add_representation('spacefill', selection=':D' + atom_selections, color='green',
                         radius='0.8')  # atom name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    v.add_representation('spacefill', selection=':E' + atom_selections, color='white',
                         radius='0.4')  # atom name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    v.add_representation('spacefill', selection=':F' + atom_selections, color='pink',
                         radius='0.8')  # atom name: .CA, .C, .N  element name: _H, _C, _O, chain name: :A
    return v


def importtrj(topo, trj, elements, dt=0, chainIDs=None, topo_format='DATA', in_memory=True,
              in_memory_step=100,
              info=False):
    # also can use ############
    # u.transfer_to_memory(step=10)
    atom_types = elements

    # get atom type and mass
    # with open(lammpsdata,'r') as lmpreader:
    #     #dump al1 all custom 100 lammpstrj.${name}.lammpstrj  id type x y z q xu yu zu vx vy vz proc element
    #     lmpreader = lmpreader.readlines()
    # atom_masses = {}
    # for line_lmp in lmpreader:
    #     line_lmp_l = line_lmp.split()
    #     if len(line_lmp_l) == 2 and line_lmp_l[0].isnumeric():
    #         try:
    #             type(float(line_lmp_l[-1]))## 如果不可以被float 说明不是mass数据所在行!
    #             atom_masses[int(line_lmp_l[0])] = float(line_lmp_l[1])
    #         except:
    #             yyy=1
    #     atom_types = len(atom_masses)
    if isinstance(trj, list):
        if 'lammpstrj' in trj[0]:
            trj_format = 'LAMMPSDUMP'
            # print('lammpstrj file detected')
        elif '.dcd' in trj[0]:
            trj_format = 'LAMMPS'
            # print('dcd file detected')
        else:
            trj_format = 'LAMMPSDUMP'
            print('file type not reconized, load as lammpstrj file')
    else:
        if 'lammpstrj' in trj:
            trj_format = 'LAMMPSDUMP'
            # print('lammpstrj file detected')
        elif '.dcd' in trj:
            trj_format = 'LAMMPS'
            # print('dcd file detected')
        else:
            trj_format = 'LAMMPSDUMP'
            print('file type not reconized, load as lammpstrj file')
    a_lengthunit = "A"
    a_timeunit = "fs"
    a_atom_style = 'id type charge x y z'  # Only when use data as topo format. Required fields: id, resid, x, y, z  Optional fields: resid, charge

    if dt != 0:
        if info:
            print('Using self_defined dt (time between two frame in dump file)!!!!! Disable by setting dt = 0')
        arggs = {'dt': dt / 1000}
    else:
        arggs = {}
    if topo_format == 'DATA':
        u = mda.Universe(topo, trj, topology_format=topo_format, format=trj_format,
                         atom_style=a_atom_style,
                         lengthunit=a_lengthunit, timeunit=a_timeunit, in_memory=in_memory,
                         in_memory_step=in_memory_step, **arggs)
    elif topo_format != 'DATA':
        u = mda.Universe(topo, trj, topology_format=topo_format, format=trj_format,
                         lengthunit=a_lengthunit, timeunit=a_timeunit, in_memory=in_memory,
                         in_memory_step=in_memory_step, **arggs)
    else:
        u = None
    ABCDE = string.ascii_uppercase
    if chainIDs:
        # 如果指定了chainid 则使用自定义的chainid
        ABCDE = chainIDs
    u.add_TopologyAttr('resname', [''] * len(u.segments.residues))
    u.add_TopologyAttr('names', range(len(u.atoms)))
    u.add_TopologyAttr('chainID')
    for i in range(len(atom_types)):
        u.select_atoms('type  ' + str(i + 1)).names = [atom_types[i]] * len(u.select_atoms('type  ' + str(i + 1)))
        u.select_atoms('type  ' + str(i + 1)).chainIDs = ABCDE[i]

    if info:
        print(u.kwargs)
        print(len(u.trajectory), ' frames')
        print(u.trajectory.dt, ' dt (ps) for two frames')
    return u


def get_atoms_index(u, L_traj, inputfile, indexfile, viewangle, write, frame, info=False):
    global indexTmpfile
    L_traj = L_traj
    inputfile = inputfile

    if indexfile != 0:
        indexTmpfile = False
    else:
        try:
            with open('half_index_tmp', 'r') as FFF:
                FFF.readline()
                indexTmpfile = True
        except:
            print('No indextempfile,generating...')
            index = []
            for i in range(4):
                locals()['index_type_' + str(i + 1)] = u.select_atoms('type ' + str(i + 1)).indices
                index.append(locals()['index_type_' + str(i + 1)])
            # indexing half by atom index
            halfindex = []
            for i in range(len(index)):
                if len(index[i]) > 0:
                    locals()['half' + str(i)] = [str(index[i][0]),
                                                 str(int(int(index[i][-1] - index[i][0]) / 2) + int(index[i][0]))]
                    halfindex.append(locals()['half' + str(i)])

            atoms = u.select_atoms('all')
            allatom_indexs = atoms.atoms.indices

            # half by geo
            atoms = u.select_atoms('all')
            for i in range(len(atoms)):
                if atoms[i].position[0] > 0:
                    atoms[i].residue = u.add_Residue(segment=u.segments[0], resid=1, resname='cf',
                                                     resnum=1)  # clusterResidue
            atoms = u.select_atoms('resname cf')
            half_atom_indexs = atoms.atoms.indices  # half index by geometry

            # half by atom index
            selections = []
            for i in halfindex:
                selection_i = ' or index ' + i[0] + ':' + i[1]
                selections.append(selection_i)
            selcetions = ''.join(selections)
            select_string = selcetions.replace(selcetions[0:3], '', 1)
            # print(select_string)
            atoms = u.select_atoms(select_string)
            half_by_indexs = atoms.atoms.indices  # half index by atom index

            with open('half_index_tmp', 'w') as file_list:
                file_list.write('### half index by geometry \n')
                file_list.write(str(half_atom_indexs.tolist()) + '\n')
                file_list.write('## half index by atom index \n')
                file_list.write(str(half_by_indexs.tolist()) + '\n')
                file_list.write('## all atom index \n')
                file_list.write(str(allatom_indexs.tolist()) + '\n')
    if not indexTmpfile:
        indexfile = indexfile
    else:
        indexfile = 'half_index_tmp'
    with open(indexfile, 'r') as file_list:
        k = file_list.readlines()
        half_atom_indexs, half_by_indexs, allatom_indexs = k[1].split(','), k[3].split(','), k[5].split(',')
        atom_indexs = [half_atom_indexs, half_by_indexs, allatom_indexs]
        atom_indexs_all = []
        for j in range(3):
            atom_indexs_i = []
            for i in atom_indexs[j]:
                i = i.strip('[')
                i = i.strip(']\n')
                atom_indexs_i.append(i)
            atom_indexs_all.append(atom_indexs_i)
    if viewangle == 'All':
        atoms = u.select_atoms('all')
        if info:
            print('atoms: ', len(atoms.select_atoms('all')))
            print(atoms.dimensions)
        if write != 0:
            for t in frame:
                u.trajectory[t]
                atoms.write(L_traj + inputfile + '-All-f-' + str(t) + '.' + write)
                if info:
                    print('Writting file to: ', L_traj + inputfile + '-All-f-' + str(t) + '.' + write)
    elif viewangle == 'Half_byindex':
        # deal with selection string
        atomindex = ' '.join(atom_indexs_all[1]).strip(']\n').strip('[').split()
        # atoms = 0
        atoms = u.select_atoms('all') - u.select_atoms('all')
        for i in atomindex:
            atoms += u.select_atoms('index ' + str(int(i)))
        if info:
            print(len(atoms))
            print(atoms.dimensions)
        if write != 0:
            for t in frame:
                u.trajectory[t]
                atoms.write(L_traj + inputfile + '-Half-f-' + str(t) + '.' + write)
                print('Writting file to: ', L_traj + inputfile + '-Half-f-' + str(t) + '.' + write)
    elif viewangle == 'Half':
        atomindex = ' '.join(atom_indexs_all[0]).strip(']\n').strip('[').split()
        # atoms = 0
        atoms = u.select_atoms('all') - u.select_atoms('all')
        for i in atomindex:
            atoms += u.select_atoms('index ' + str(int(i)))
        if write != 0:
            for t in frame:
                u.trajectory[t]
                atoms.write(L_traj + inputfile + '-Half-f-' + str(t) + '.' + write)
                print('Writting file to: ', L_traj + inputfile + '-Half-f-' + str(t) + '.' + write)
    else:
        print('error: trajviewer(viewangle Half/All/Half_byindex,write="pdb",frame=0):')
        atoms = u.select_atoms('all')
    vvv = nv.show_mdanalysis(atoms)
    MDAobject = [u, atoms]
    return MDAobject, atom_indexs_all, vvv


def construct_sub_universe(u, selection="type 1 2", singleframe=None):
    select = u.select_atoms(selection)

    print('Atoms in sub_universe:', len(select.atoms))
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                       select).run().results['timeseries']
    u2 = mda.Merge(select)  # create the xxxxx-only Universe
    if singleframe is not None:
        print('All frames:', len(u.universe.trajectory), 'Selected frame:', singleframe)
        print('singleframe')
        u2.load_new(coordinates[singleframe], format=MemoryReader)
    else:
        print('All frames:', len(u.universe.trajectory))
        u2.load_new(coordinates, format=MemoryReader)
    u2.dimensions = u.dimensions
    return u2


def cut_a_singleframe(u, frame, info=False, format=None):
    """

    :param u: mda universe/atomgroup
    :param frame: frame need to be cut
    :param info: bool
    :return: path of a tmp dcd
    """
    if format is None:
        format = 'dcd'
    try:
        if info:
            print('using universe as input')
        totalframe = len(u.trajectory)
        system = u.select_atoms('all')
        system.universe.trajectory[frame]
    except:
        if info:
            print('using atom group as input')
        totalframe = len(u.universe.trajectory)
        system = u.select_atoms('all')
        system.universe.trajectory[frame]
    currentframe = system.universe.trajectory.frame
    if info:
        print('totalframe:', totalframe)
        print('selectedframe:', currentframe)
    uuidNAME = str(uuid.uuid4())
    if format == 'dcd':
        tempfile = '/tmp/Cutframe-' + uuidNAME + '.dcd'
        if info:
            print('Using dcd as output!topo file are needed')
        # tempDCDfile = '/tmp/Cutframe-'+str(frame)+'.dcd'
        system.write(tempfile)
    elif format == 'xyz':
        tempfile = '/tmp/Cutframe-' + uuidNAME + '.xyz'
        system.write(tempfile)
    elif format == 'gro':
        tempfile = '/tmp/Cutframe-' + uuidNAME + '.gro'
        system.write(tempfile)
    elif format == 'pdb':
        tempfile = '/tmp/Cutframe-' + uuidNAME + '.pdb'
        system.write(tempfile)
    else:
        print('error!')
        tempfile = 'error'
    return tempfile


def wrap_the_cluster(u):  # wrap 220604
    # wrap the cluster is better
    # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
    # https://userguide.mdanalysis.org/stable/trajectories/transformations.html
    anp = u.select_atoms('type 1  2 3')
    eO = u.select_atoms('type 4  ')
    # nv.show_mdanalysis(newsys)
    transforms = [
        trans.unwrap(anp),
        trans.center_in_box(anp, center='mass', wrap=True),
        trans.unwrap(eO),
    ]
    #     transforms = [

    #                   trans.wrap(eO),
    #                   trans.wrap(anp),
    #                   trans.center_in_box(anp,center='mass',wrap=True),
    #                   trans.center_in_box(eO,center='mass',wrap=True),

    #                  ]
    u.trajectory.add_transformations(*transforms)
    return u


def recenter_singleframe(singleframe, info=False):
    def workflow(centro_id):
        def wrapped(ts):
            x, y, z = centro_id
            if info:
                print('Origin center_of_mass', centro_id)
            # here's where the magic happens
            # we create a numpy float32 array to avoid reduce floating
            # point errors
            ts.positions += np.asarray([-x, -y, -z])
            return ts

        return wrapped

    # making the unmodified universe
    recentered_frame = singleframe.copy()
    recentered_frame.dimensions = singleframe.dimensions  # this is mandatory!!
    centroid = recentered_frame.select_atoms('all').center_of_mass()
    workflow = (workflow(centroid))
    recentered_frame.trajectory.add_transformations(workflow)
    if info:
        print('Done! center_of_mass', recentered_frame.select_atoms('all').center_of_mass())
    return recentered_frame


def fabricate_u_from_atom_index(u, atomindex, info=False):
    # atoms=mda.Universe.empty(0, 0, atom_resindex=[], trajectory=True).select_atoms('all')
    atoms = u.select_atoms('all') - u.select_atoms('all')
    # atoms=0
    for i in atomindex:
        atoms += u.select_atoms('index ' + str(int(i)))
    if info:
        try:
            print(len(atoms))
            print(atoms.dimensions)
        except:
            print('Error! given index is empty!----fabricate_u_from_atom_index')
    return atoms


def atoms_index_withincutoff(atomgroup, ref, cutoff):
    dist_arr = distances.distance_array(atomgroup.positions, ref.positions, box=ref.dimensions)
    index = []
    for i in range(len(atomgroup)):
        for j in range(len(ref)):
            if 0 < dist_arr[i][j] < cutoff:  # 0<说明不是原子本身 排除原子自己  < cutoff  说明与al发生了psy/chem反应
                index.append(atomgroup.atoms[i].index)
    index = list(dict.fromkeys(index))  # remove duplicates!!!
    return index


def cross_cut(u, cut='xyz0', info=False, selection=False):
    # get an approxi centerid of ANP cluster by selecting all the Al atoms
    if selection:
        sys = u.select_atoms(selection)
    else:
        sys = u.select_atoms('all')

    centerids = [sys.center_of_mass(), sys.center_of_geometry(), sys.centroid()]
    if info:
        print("sys.center_of_mass(),sys.center_of_geometry(),sys.centroid()")
        print(centerids)
    try:
        [cx, cy, cz] = centerids[1]
    except:
        print('Center of mass not detected, is the system even contains any atom?')
        [cx, cy, cz]=[0,0,0]
    index = []
    atoms = sys
    # x
    if len(cut) == 1:
        if cut == 'x':
            val = cx
            axis = 0
        elif cut == 'y':
            val = cy
            axis = 1
        elif cut == 'z':
            val = cz
            axis = 2
        else:
            val = cx
            axis = 0
            print('input error, use default value')
        for i in range(len(atoms)):
            if atoms[i].position[axis] > val:
                index.append(atoms[i].index)
        index = list(dict.fromkeys(index))

    # xy
    if len(cut) == 2:
        if '-' in cut:
            if cut == '-x':
                val = cx
                axis = 0
            elif cut == '-y':
                val = cy
                axis = 1
            elif cut == '-z':
                val = cz
                axis = 2
            else:
                val = cx
                axis = 0
                print('input error, use default value')
            index = []
            for i in range(len(atoms)):
                if atoms[i].position[axis] < val:
                    index.append(atoms[i].index)
            index = list(dict.fromkeys(index))
        else:
            if 'x' not in cut:  # yz
                val0 = cy
                axis0 = 1
                val1 = cz
                axis1 = 2
            elif 'y' not in cut:  # xz
                val0 = cx
                axis0 = 0
                val1 = cz
                axis1 = 2
            elif 'z' not in cut:  # xy
                val0 = cx
                axis0 = 0
                val1 = cy
                axis1 = 1
            else:
                val0 = cy
                axis0 = 1
                val1 = cz
                axis1 = 2
                print('input error, use default value')
            for i in range(len(atoms)):
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] < val1:
                    index.append(atoms[i].index)
            index = list(dict.fromkeys(index))
    # xyz
    if len(cut) == 3:
        val0 = cx
        axis0 = 0
        val1 = cy
        axis1 = 1
        val2 = cy
        axis2 = 2
        print('plz use xyz[0-7] instead!\n' * 4)
        for i in range(len(atoms)):
            if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                axis2] > val2:
                index.append(atoms[i].index)
        index = list(dict.fromkeys(index))

    if len(cut) == 4:
        val0 = cx
        axis0 = 0
        val1 = cy
        axis1 = 1
        val2 = cy
        axis2 = 2
        method = int(cut[-1])
        for i in range(len(atoms)):
            if method == 0:
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                    axis2] > val2:
                    index.append(atoms[i].index)
            elif method == 1:
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                    axis2] < val2:
                    index.append(atoms[i].index)
            elif method == 2:
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] < val1 or atoms[i].position[
                    axis2] > val2:
                    index.append(atoms[i].index)
            elif method == 3:
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] < val1 or atoms[i].position[
                    axis2] < val2:
                    index.append(atoms[i].index)
            elif method == 4:
                if not atoms[i].position[axis0] > val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                    axis2] > val2:
                    index.append(atoms[i].index)
            elif method == 5:
                if not atoms[i].position[axis0] > val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                    axis2] < val2:
                    index.append(atoms[i].index)
            elif method == 6:
                if not atoms[i].position[axis0] > val0 or atoms[i].position[axis1] < val1 or atoms[i].position[
                    axis2] > val2:
                    index.append(atoms[i].index)
            elif method == 7:
                if not atoms[i].position[axis0] > val0 or atoms[i].position[axis1] < val1 or atoms[i].position[
                    axis2] < val2:
                    index.append(atoms[i].index)
            else:
                if not atoms[i].position[axis0] < val0 or atoms[i].position[axis1] > val1 or atoms[i].position[
                    axis2] > val2:
                    index.append(atoms[i].index)
                print('input error, use default value')
        index = list(dict.fromkeys(index))

    atoms = fabricate_u_from_atom_index(u, atomindex=index, info=False)
    return atoms, index


# plot charge part
def rewrite_lammpsdump(lammpsdata, lammpsdump):
    # get the start index of  data row
    print('reading lammpsdata')
    with open(lammpsdata, 'r') as dataread:
        lines = dataread.readlines()
        print('generate atom typedict')
        typedic = {}
        for line in lines:
            if len(line.split()) == 9:
                typedic[line.split()[0]] = line.split()[1]
    # write the mod lammpsdump
    with open(lammpsdump, 'r') as origindump:
        print('reading lammpsdump')
        lines = origindump.readlines()
        print('write the mod lammpsdump')
        with open(lammpsdump + '_mod', 'w') as dumpwriter:
            for index, line in enumerate(lines):
                line_l = line.split(' ')
                if 'ITEM: ATOMS' in line:
                    line_l = 'ITEM: ATOMS id type x y z q'.split()
                    newline = ' '.join(line_l) + '\n'
                elif line_l[0].isnumeric():
                    if len(line_l) == 6:
                        line_l[1] = typedic[line_l[0]]
                        newline = '\t'.join(line_l)
                    else:
                        newline = '\t'.join(line_l)
                else:
                    newline = ' '.join(line.split()) + '\n'
                dumpwriter.write(newline)


def vw(u):
    v = nv.show_mdanalysis(u)
    return v


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
        N.write('CRYST1  180.000  180.000  180.000  90.00  90.00  90.00 P 1           1\n')
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
