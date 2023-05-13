import os
import ipynbname
from LazzyMDkit.PlotCustomizer import *

class IntEnv:
    def __init__(self):
        # nb_fname = ipynbname.name()
        nb_path = ipynbname.path()
        PJdir = '/'.join(str(nb_path).split('/')[:-3]) + '/'
        PJname = str(nb_path).split('/')[-3]

        self.PJname = PJname
        self.inputfile = 'in.' + PJname
        self.trajs = 'dcd.' + PJname + '.dcd'
        self.lammpstrj = 'lammpstrj.' + PJname + '.lammpstrj'
        self.lammpslog = 'log.' + PJname
        self.slurmlog = 'Output.' + self.inputfile
        self.slurmerrorlog = 'Error.' + self.inputfile

        self.R_workdir = PJdir.replace('MD_domain', 'MD-analysis') + PJname + '/'
        self.L_workdir = PJdir + PJname + '/'
        self.L_input = self.L_workdir + '#input/'
        self.L_tmp = self.L_workdir + 'tmp/'
        self.L_traj = self.L_workdir + 'tmp/'
        self.L_log = self.L_workdir + 'tmp/'
        self.L_reax = self.L_workdir + 'tmp/'
        self.L_analysis = self.L_workdir + '#analysis/'

        self.L_folders = [self.L_workdir, self.L_input, self.L_tmp, self.L_traj, self.L_log, self.L_reax,
                          self.L_analysis]
        self.L_foldernames = [i.rsplit('/', 2)[-2] for i in self.L_folders]
        self.R_foldername = self.R_workdir.rsplit('/', 2)[-2]

        os.system('mkdir -p images')

        plt.text(0.6, 0.7, "Initialize", size=30, rotation=30.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )

        plt.text(0.3, 0.6, "Done", size=30, rotation=-25.,
                 ha="right", va="top",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )
        # fun
        plt.gca().yaxis.set_major_formatter(axis_num_formatter)
        plt.show()

    def getlmpdatafile(self):
        L_input = self.L_input
        inputfile = self.inputfile
        with open(L_input + inputfile, encoding="utf8", errors='ignore') as inp:
            f = inp.readlines()
        fname = 0
        for i in f:
            if i.startswith('read_data'):
                fname = i.split()[-1]
        if fname == 0:
            print('Error, not found lmp file from local input')
        else:
            return fname
