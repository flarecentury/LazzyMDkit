import fileinput
import os
import pickle
import re
from enum import Enum, auto
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from LazzyMDkit.Utils import run_mp


class ReadLAMMPSdump:
    def __init__(self, atomname):
        self.atomname = atomname

    class LineType(Enum):
        """Line type in the LAMMPS dump files."""
        TIMESTEP = auto()
        ATOMS = auto()
        NUMBER = auto()
        BOX = auto()
        OTHER = auto()

        @classmethod
        def linecontent(cls, line):
            """Return line content."""
            if line.startswith("ITEM: TIMESTEP"):
                return cls.TIMESTEP
            if line.startswith("ITEM: ATOMS"):
                return cls.ATOMS
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                return cls.NUMBER
            if line.startswith("ITEM: BOX"):
                return cls.BOX
            return cls.OTHER

    def _readNfunc(self, f):
        iscompleted = False
        for index, line in enumerate(f):
            if line.startswith("ITEM:"):
                linecontent = self.LineType.linecontent(line)
                if linecontent == self.LineType.ATOMS:
                    self.keys = line.split()[2:]
                    keys = self.keys
                    self.id_idx = keys.index('id')
                    self.tidx = keys.index('type')
                    self.xidx = keys.index('x')
                    self.yidx = keys.index('y')
                    self.zidx = keys.index('z')
            else:
                if linecontent == self.LineType.NUMBER:
                    if iscompleted:
                        steplinenum = index - stepaindex
                        break
                    else:
                        iscompleted = True
                        stepaindex = index
                    N = int(line.split()[0])
                    atomtype = np.zeros(N, dtype=int)
                elif linecontent == self.LineType.ATOMS:
                    s = line.split()
                    atomtype[int(s[self.id_idx]) - 1] = int(s[self.tidx]) - 1
        else:
            steplinenum = index + 1
        self.N = N
        self.atomtype = atomtype
        return steplinenum, N, atomtype, keys

    def _readstepfunc(self, item):
        step, lines = item
        step_atoms = []
        ss = []
        atomname = self.atomname
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    linecontent = self.LineType.linecontent(line)
                else:
                    if linecontent == self.LineType.ATOMS:
                        s = line.split()
                        step_atoms.append(s)
                    elif linecontent == self.LineType.TIMESTEP:
                        timestep = step, int(line.split()[0])
                    elif linecontent == self.LineType.BOX:
                        s = line.split()
                        ss.append(list(map(float, s)))
        ss = np.array(ss)
        if ss.shape[1] > 2:
            xy = ss[0][2]
            xz = ss[1][2]
            yz = ss[2][2]
        else:
            xy, xz, yz = 0., 0., 0.
        xlo = ss[0][0] - min(0., xy, xz, xy + xz)
        xhi = ss[0][1] - max(0., xy, xz, xy + xz)
        ylo = ss[1][0] - min(0., yz)
        yhi = ss[1][1] - max(0., yz)
        zlo = ss[2][0]
        zhi = ss[2][1]
        boxsize = np.array([[xhi - xlo, 0., 0.],
                            [xy, yhi - ylo, 0.],
                            [xz, yz, zhi - zlo]])

        step_atoms = np.asarray(step_atoms)
        step_atoms = pd.DataFrame(step_atoms, columns=self.keys)
        # change colum data type
        keys = self.keys
        for key in keys:
            if key in ['id', 'type', 'proc']:
                step_atoms[key] = step_atoms[key].astype(int)
            else:
                try:
                    step_atoms[key] = step_atoms[key].astype(float)
                except:
                    step_atoms[key] = step_atoms[key]
        # sort dataframe by atom id
        step_atoms = step_atoms.sort_values(by=['id'], ignore_index=True)
        for k in range(len(atomname)):
            step_atoms.loc[(step_atoms.type == k + 1), 'element'] = atomname[k]
        return timestep, step_atoms, boxsize


def save_to_file(v_lists, v_names, counter):
    try:
        os.system('mkdir tmp_trj')
    except:
        print('dir exist')
    for idx, i in enumerate(v_lists):
        pickle.dump(i, open(f'tmp_trj/{v_names[idx]}_{str(counter)}.pickle', 'wb'))


def read_lammpsdump_to_pickle(trj, elements, interval=1, nproc=1, picklize_size=64):
    atomname = np.asarray(elements)
    trj = trj
    nproc = nproc
    interval = interval

    os.system('rm tmp_trj/*.pickle')
    FF = ReadLAMMPSdump(atomname)
    # obtain _steplinenum N atomtype
    f = fileinput.input(files=trj)
    _steplinenum, N, atomtype, keys = FF._readNfunc(f)
    print(f'_steplinenum: {_steplinenum},N: {N},atomtype: {atomtype}')
    f.close()

    # read atoms
    f = fileinput.input(files=trj)
    results = run_mp(nproc=nproc, func=FF._readstepfunc, l=f, nlines=_steplinenum, unordered=False, return_num=True,
                     interval=interval,
                     desc="Read trj information", unit="timestep")

    timestep_s = []
    step_atoms_s = []
    boxsize_s = []
    v_names = ['timestep_s', 'step_atoms_s', 'boxsize_s']

    counter = 0
    for timestep, step_atoms, boxsize in results:
        timestep_s.append(timestep)
        step_atoms_s.append(step_atoms)
        boxsize_s.append(boxsize)
        if len(timestep_s) == picklize_size:
            print(f'total steps is larger than {picklize_size}, flush and write to file: {counter}')
            v_lists = [timestep_s, step_atoms_s, boxsize_s]

            save_to_file(v_lists, v_names, counter)
            timestep_s = []
            step_atoms_s = []
            boxsize_s = []
            counter += 1

    print(f'write last buffer to file: {counter}')
    v_lists = [timestep_s, step_atoms_s, boxsize_s]
    save_to_file(v_lists, v_names, counter)
    f.close()
    os.system('ls -alh tmp_trj/*.pickle $pwd')


def read_lammpsdump_from_pickle():
    def num_sort(test_string):
        try:
            number = list(map(int, re.findall(r'\d+', test_string)))[0]
            return number
        except:
            print(f'skip {test_string}')
            return 9999999

    fileslist = [f for f in listdir('tmp_trj') if isfile(join('tmp_trj', f))]
    fileslist.sort(key=num_sort)
    # load all binary file into memory
    step_atoms_s = []
    boxsize_s = []
    timestep_s = []
    for f in fileslist:
        if f.startswith('timestep_s') and 'pickle' in f:
            timestep_s.extend(pickle.load(open('tmp_trj/' + f, 'rb')))
        elif f.startswith('boxsize_s') and 'pickle' in f:
            boxsize_s.extend(pickle.load(open('tmp_trj/' + f, 'rb')))
        elif f.startswith('step_atoms_s') and 'pickle' in f and 'all' not in f:
            print('tmp_trj/' + f)
            step_atoms_s.extend(pickle.load(open('tmp_trj/' + f, 'rb')))
    print('total frames:', len(step_atoms_s))
    df_s = step_atoms_s
    steps_all = [i[1] for i in timestep_s]
    return df_s, steps_all, boxsize_s, fileslist
