import math
import numpy as np
import pandas as pd
import tqdm
from sympy import Matrix
import plotly.graph_objects as go


def get_x_data_per_atom_type(df_s, elements, dfcolumn=None):
    if dfcolumn is None:
        dfcolumn = ['id', 'c_peratom[1]', 'c_peratom[2]', 'c_peratom[3]', 'c_peratom[4]', 'c_peratom[5]',
                    'c_peratom[6]']
    x_s_all = []
    # get charge / velocity /....
    for df in df_s:
        x_s = []
        for i in range(len(elements)):
            try:
                x_i = df.loc[df['type'].astype(float) == i + 1][dfcolumn].astype(float)
            except:
                print('data is not float_able!check if u are using right lammpsdump file!')
                x_i = df.loc[df['type'].astype(float) == i + 1][dfcolumn]
            x_s.append(x_i)
        x_s_all.append(x_s)
    return x_s_all


def filter_x_para_s_all(x_s_all, labeled_index):
    x_s_all_filtered = []
    for x_s in x_s_all:
        x_s_filtered = []
        for x_s_i in x_s:
            x_s_i_filtered = x_s_i[x_s_i.index.isin(labeled_index)]
            x_s_filtered.append(x_s_i_filtered)
        x_s_all_filtered.append(x_s_filtered)
    return x_s_all_filtered


def merge_dfs(x_s_all):
    x_s_all_combined = []
    for f in range(len(x_s_all)):
        combineddf = pd.concat(x_s_all[f]).drop_duplicates()
        x_s_all_combined.append(combineddf)
    return x_s_all_combined


def cal_stress_s(p_s_all, elements, centreid_s, calPrincipal_stress=False):
    """

    :param p_s_all:
    :param elements:
    :param centreid_s:
    :param calPrincipal_stress:
    :return:
    # lammps里面的应力是能量的量纲，但是不是一个标量，也是有方向性的，其实就是对应各个应力分量，应力的量纲是能量
    """

    # 六个stree的张量，只有x y z 方向的数值较大，因此可以忽略掉另外3个数值，默认忽略，节省算力
    if not calPrincipal_stress:
        print('简化stress计算')
    molar_volumes = {'Al': 10.00, 'O': 17.36, 'C': 5.29, 'H': 11.42}  # 10-6 m3/mol

    def convert_to_atom_volumes(molar_volume_s=None):
        if molar_volume_s is None:
            molar_volume_s = {'Al': 10.00, 'O': 17.36, 'C': 5.29, 'H': 11.42}
        keys = list(molar_volume_s.keys())
        volumes_angstrom = [(k / 6.02214076) * 10 for k in molar_volume_s.values()]  # A^3 per atom
        atom_volumes_in_angstrom = molar_volume_s.copy()
        for i in range(len(molar_volume_s)):
            atom_volumes_in_angstrom.update({keys[i]: volumes_angstrom[i]})
        return atom_volumes_in_angstrom

    # def radial_stress(centre_id, pos_atom_i, stress_atom_i, debug=False):
    #     # 注意，这里是根据输入的xx yy zz 方向应力和质心 计算xx yy zz三个方向的径向应力的合力
    #     x0, y0, z0 = centre_id  # centre of mass
    #     x1, y1, z1 = pos_atom_i  # pos_atom_i
    #     v_x, v_y, v_z = stress_atom_i  # stresses
    #     dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    #     # 相对质心的位置
    #     L_yz = abs(x1 - x0)
    #     L_xz = abs(y1 - y0)
    #     L_xy = abs(z1 - z0)
    #     # cos夹角
    #     cos_yz = L_yz / dist
    #     cos_xz = L_xz / dist
    #     cos_xy = L_xy / dist
    #     # xyz  方向主应力在径向的分量
    #     v_x_radial = v_x * cos_yz
    #     v_y_radial = v_y * cos_xz
    #     v_z_radial = v_z * cos_xy
    #     # print(cos_yz,cos_xz,cos_xy)
    #     # 径向力
    #     v_radial = v_x_radial + v_y_radial + v_z_radial
    #     # print('cos_yz,cos_xz,cos_xy',cos_yz,cos_xz,cos_xy)
    #     if debug:
    #         print('centre_id', centre_id)
    #         print('pos_atom_i',pos_atom_i)
    #         print('stress_atom_i',stress_atom_i)
    #         print('x1,y1,z1', x1, y1, z1)
    #         print('L_yz,L_xz,L_xy,dist', L_yz, L_xz, L_xy, dist)
    #         print('cos_yz,cos_xz,cos_xy', cos_yz, cos_xz, cos_xy)
    #         print('v_x_radial,v_y_radial,v_z_radial', v_x_radial, v_y_radial, v_z_radial)
    #     return v_radial

    def radial_stress(centre_id, pos_atom_i, stress_atom_i, debug=False):
        # Extract coordinates from the input
        xc, yc, zc = centre_id
        xp, yp, zp = pos_atom_i
        sigma_x, sigma_y, sigma_z = stress_atom_i

        # Compute the direction vector from Centre to Atom_i
        r = [xp - xc, yp - yc, zp - zc]

        # Compute the magnitude of r
        magnitude_r = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)

        # Compute the unit direction vector
        u = [r[0]/magnitude_r, r[1]/magnitude_r, r[2]/magnitude_r]

        # Compute the radial stress
        sigma_r = sigma_x * u[0] + sigma_y * u[1] + sigma_z * u[2]

        return sigma_r


    atom_volumes_angstrom = convert_to_atom_volumes(molar_volumes)
    # get atom volume from above dict according to the elements
    atom_volumes_s = [atom_volumes_angstrom[j] for j in elements]  # generate a list from dict
    print('atom_volumes_s', atom_volumes_s)
    print('elements', elements)
    if len(elements) != len(p_s_all[0]):
        print('The atom list you provided does not correspond to the pressure data')

    radial_stress_s_all = []
    stress_s_all = []

    for frame in tqdm.tqdm(range(len(p_s_all))):
        centreid = centreid_s[frame]
        radial_stress_s = []
        # print('frame:',frame)
        for index in range(len(elements)):
            atom_volumes_i = atom_volumes_s[index]
            ids = p_s_all[frame][index]['id']
            datalist = p_s_all[frame][index][
                ['id', 'type', 'x', 'y', 'z', 'c_peratom[1]', 'c_peratom[2]', 'c_peratom[3]', 'c_peratom[4]',
                 'c_peratom[5]', 'c_peratom[6]', ]].values.tolist()
            # step 1 calPrincipal_stress
            principal_stress_s_type_i = []
            principal_stress_x_type_i = []
            principal_stress_y_type_i = []
            principal_stress_z_type_i = []
            pos_s = []
            for data in datalist:  # cal stress row by row
                atomid, atomtype, posi_x, posi_y, posi_z, a, b, c, d, e, f = [float(k) for k in data]
                if calPrincipal_stress:
                    # 注意，这里计算得到的时主应力，但主应力的方向并非是xyz坐标轴方向，而radial_stress函数未考虑此问题，因此，需要考虑主方向，以正确计算径向应力
                    # cal principal_stress ######################
                    sigma = Matrix([[a, d, e], [d, b, f], [e, f, c]])
                    principal_stress_per_atom_type_i = [float(i) for i in [str(i).split('.')[0] for i in
                                                                           list(sigma.eigenvals().keys())]]  # 计算主应力
                    principal_stress_s_type_i.append(principal_stress_per_atom_type_i)
                else:
                    # dont cal principal_stress, use pxx yy zz as input for calculating the radial stress
                    # 因 d e f 数值较小，可简化为[a, 0, 0], [0, b, 0], [0, 0, c]
                    principal_stress_per_atom_type_i = [a, b, c]
                    principal_stress_s_type_i.append([a, b, c])  # 主应力
                # create 3 list containing the principal_stress at x y z direction
                # 主应力在x y z 方向分量
                principal_stress_x_type_i.append(principal_stress_per_atom_type_i[0])
                principal_stress_y_type_i.append(principal_stress_per_atom_type_i[1])
                principal_stress_z_type_i.append(principal_stress_per_atom_type_i[2])
                posi = [posi_x, posi_y, posi_z]
                pos_s.append(posi)

            # step 2 cal radial_stress_s
            # 注意 以上的stress 并非真正的stress 其包含了体积项  本步将除去原子体积
            radial_stress_s_type_i = []
            for i in range(len(principal_stress_s_type_i)):
                posi = pos_s[i]
                stress_i = principal_stress_s_type_i[i]
                radial_stress_i = radial_stress(centreid, posi, stress_i, debug=False)
                radial_stress_s_type_i.append(- radial_stress_i / (atom_volumes_i * 1) * (
                    0.0001))  # 0.0001 # atomespheres --> Gpa ：1 GPa? The answer is 9869.2326671601.
                # if -radial_stress_i/(atom_volumes_i*1)*(0.0001) > 20:
                # print(elements[index])
                # print(atom_volumes_s[index])
                # print(atom_volumes_i)
            # print(radial_stress_s_type_i)
            radial_stress_s_type_i_df = pd.DataFrame(ids, columns=['id']).astype(int)

            radial_stress_s_type_i_df['type'] = p_s_all[frame][index]['type'].astype(int)
            radial_stress_s_type_i_df['Radial stress'] = radial_stress_s_type_i
            radial_stress_s_type_i_df['principal_stress_x'] = principal_stress_x_type_i
            radial_stress_s_type_i_df['principal_stress_y'] = principal_stress_y_type_i
            radial_stress_s_type_i_df['principal_stress_z'] = principal_stress_z_type_i
            radial_stress_s_type_i_df[
                ['x', 'y', 'z', 'c_peratom[1]', 'c_peratom[2]', 'c_peratom[3]', 'c_peratom[4]', 'c_peratom[5]',
                 'c_peratom[6]']] = p_s_all[frame][index][
                ['x', 'y', 'z', 'c_peratom[1]', 'c_peratom[2]', 'c_peratom[3]', 'c_peratom[4]', 'c_peratom[5]',
                 'c_peratom[6]']].astype(float)
            radial_stress_s.append(radial_stress_s_type_i_df)
        radial_stress_s_all.append(radial_stress_s)

        # step 3 cal xx yy zz 方向应力及三个方向应力合力的模
        p_s = []
        for index in range(len(elements)):
            ids = p_s_all[frame][index]['id']
            px = p_s_all[frame][index]['c_peratom[1]']
            py = p_s_all[frame][index]['c_peratom[2]']
            pz = p_s_all[frame][index]['c_peratom[3]']
            # cal the results
            atom_volumes_i = atom_volumes_s[index]
            stressxyz_per_atom_type_i = np.sqrt((px ** 2 + py ** 2 + pz ** 2)) / (atom_volumes_i * 1) * (0.0001)
            # xyz三个方向上的合应力模，注意，方向各不相同
            stressx_per_atom_type_i = px / (atom_volumes_i * 1) * 0.0001
            # x方向上的应力的模，注意，方向各不相同
            stressy_per_atom_type_i = py / (atom_volumes_i * 1) * 0.0001
            stressz_per_atom_type_i = pz / (atom_volumes_i * 1) * 0.0001
            # create a df to store the results
            stress_per_atom_type_i = pd.DataFrame(ids, columns=['id']).astype(int)
            stress_per_atom_type_i['type'] = p_s_all[frame][index]['type'].astype(int)
            stress_per_atom_type_i['stress_xyz'] = stressxyz_per_atom_type_i.astype(float)
            stress_per_atom_type_i['stress_x'] = stressx_per_atom_type_i.astype(float)
            stress_per_atom_type_i['stress_y'] = stressy_per_atom_type_i.astype(float)
            stress_per_atom_type_i['stress_z'] = stressz_per_atom_type_i.astype(float)
            p_s.append(stress_per_atom_type_i)
        stress_s_all.append(p_s)

    radial_stress_s_all_combined = merge_dfs(radial_stress_s_all)  # radial fangxiang
    stress_s_all_combined = merge_dfs(stress_s_all)  # xx yy zz 方向应力及三个方向应力合力的模，未考虑xy xz yz 方向

    return radial_stress_s_all_combined, stress_s_all_combined


# cal temp
def cal_temp_s(df_s, lammpsdata, out=False):
    v_s_all = []

    # get atom type and mass
    with open(lammpsdata, 'r') as lmpreader:
        # dump al1 all custom 100 lammpstrj.${name}.lammpstrj  id type x y z q xu yu zu vx vy vz proc element
        lmpreader = lmpreader.readlines()
    atom_masses = {}
    for line_lmp in lmpreader:
        line_lmp_l = line_lmp.split()
        if len(line_lmp_l) == 2 and line_lmp_l[0].isnumeric():
            try:
                type(float(line_lmp_l[-1]))  # 如果不可以被float 说明不是mass数据所在行!
                atom_masses[int(line_lmp_l[0])] = float(line_lmp_l[1])
            except:
                yyy = 1
    atom_types = len(atom_masses)
    print('atom_types', atom_types)
    # get velocity
    for df in df_s:
        # print(df)
        v_s = []
        for i in range(atom_types):
            v_i = df[i].loc[df[i]['type'].astype(float) == i + 1][['id', 'type', 'vx', 'vy', 'vz']].astype(float)
            v_s.append(v_i)
        v_s_all.append(v_s)

    # k_B = 1.380649 * math.pow(10, -23)  # J/K
    # Na = 6.02214076 * math.pow(10, 23)  # mol-1
    # 指数相反,这里直接去掉z
    k_B = 1.380649  # J/K
    Na = 6.02214076  # mol-1
    Ct = Na * k_B

    T_s_all = []
    c_labels = list(v_s_all[0][0].columns)
    if out:
        print('using colum:', c_labels[2], c_labels[3], c_labels[4], 'as data source!')
    for f in range(len(v_s_all)):
        T_s = []
        for index in range(atom_types):
            vx = v_s_all[f][index][c_labels[2]]
            vy = v_s_all[f][index][c_labels[3]]
            vz = v_s_all[f][index][c_labels[4]]
            V_atom_type_i = vx ** 2 + vy ** 2 + vz ** 2

            # cal T from v**2
            m_atom_type_i = atom_masses[index + 1]
            m = m_atom_type_i
            T_atom_type_i = [(1 / 2 * (2 / 3) * math.pow(10, 7) * m * v2 / Ct / 1) for v2 in
                             V_atom_type_i]  # 3 dimention, 最后一项为N, 原子数目，此处算的是单个原子
            # https://zhuanlan.zhihu.com/p/412880682
            ids = v_s_all[f][index]['id']
            types = v_s_all[f][index]['type']

            T_atom_type_i_df = pd.DataFrame(ids, columns=['id']).astype(int)
            T_atom_type_i_df['type'] = types.astype(int)
            T_atom_type_i_df['Temp'] = T_atom_type_i
            T_s.append(T_atom_type_i_df)
        T_s_all.append(T_s)

    T_s_all_combined = merge_dfs(T_s_all)
    return T_s_all_combined


def plot_data(pos_s, para_s, labeled_index=None, atomnames=['cAl', 'sO', 'sAl', 'eO'], colors=['silver','red','black','green'], auto_colorscale=False, cmin=0, cmax=0,
              filter=None,
              out=True, legendstr='legend', hide_background=False, hide_axis=False, hide_legend=False):
    global cMax, cMin, para_name, info
    if labeled_index is None:
        labeled_index = []
    fontsize = 11
    # https://plotly.com/python/3d-mesh/
    # https://plotly.com/python/builtin-colorscales/
    # import time
    # from plotly import graph_objs as go
    # make sure para_s is float
    para_s_float = []
    for para in para_s:
        para_s_float.append(para.astype(float))
    para_s = para_s_float

    def max_min_value(param_s):
        maxs = []
        mins = []
        for paraValue in param_s:
            if float(paraValue.max()) < 0.1:  # 对较小数字,不round
                maxs.append(float(paraValue.max()))
                mins.append(float(paraValue.min()))
            else:
                maxs.append(round(float(paraValue.max()), 4))
                mins.append(round(float(paraValue.min()), 4))
        # p_max = float(round(np.p_max(np.asarray(maxs)),4))
        # p_min = float(round(np.p_min(np.asarray(mins)),4))
        if np.max(np.asarray(maxs)) < 0.1:  # 对较小数字,不round
            p_max = np.max(np.asarray(maxs))
            p_min = np.min(np.asarray(mins))
        else:
            p_max = round(np.max(np.asarray(maxs)), 4)
            p_min = round(np.min(np.asarray(mins)), 4)
        return p_min, p_max, mins, maxs

    Min, Max, Mins, Maxs = max_min_value(para_s)

    # set color scale
    if not auto_colorscale:
        if cmin != 0 or cmax != 0:
            cMax = cmax
            cMin = cmin
            if out:
                print('color scaling using user defined cMin,cMax', cMin, cMax)
        else:
            cMax = None
            cMin = None
            if out:
                print('disable color scaling cMin,cMax not provided!')
    elif auto_colorscale:
        cMax = Max
        cMin = Min
        if out:
            print('color scaling using detected limit', cMin, cMax)

    # colorscale = 'Spectral'
    # colorscale = 'magma'
    colorscale = 'turbo'
    traces = []
    # colors = ['silver','red','black','green']

    for i in range(len(atomnames)):
        # para_name
        para_name = para_s[i].columns[0]
        Mean = round(float(para_s[i].mean()), 4)
        if out:
            print('Type ' + str(i) + ' ' + atomnames[i] + ', mean ' + para_name + ' ' + str(Mean) + ' p_max:' + str(
                Maxs[i]) + ' p_min:' + str(Mins[i]))
        para_s_i = para_s[i].copy()

        # filter noise
        if filter is not None:
            print('warning, filter is enabled! triming data..')
            print('color scaling with cmax,cmin vaules from filter')

            def clean(x, filter):
                if float(x) > float(filter[1]):
                    x = float(filter[1])
                elif float(x) < float(filter[0]):
                    x = float(filter[0])
                else:
                    x = x
                return float(x)

            para_s_i[para_name] = para_s_i[para_name].apply(clean)
            cMin = float(filter[0])
            cMax = float(filter[1])

        para_s_i_c = para_s_i.copy()

        # get paraValue text info
        def info(x, prefix=para_name):  # name: charge/velocity/...
            if x > 0.1:
                x = prefix + ': ' + str(round(x, 4))
            else:  # 对较小数字,不round
                x = prefix + ': ' + str(x)
            return x

        textinfo = para_s_i[para_name].apply(info)

        # atom position
        pos = pos_s[i]
        ##################################

        # atoms
        # color0 = colors[i]
        color0 = colors[i]
        data = go.Scatter3d(
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            text=textinfo,
            mode='markers',
            name=atomnames[i],
            opacity=1.0,
            marker=dict(
                sizemode='diameter',
                sizeref=50,
                size=5,
                color=color0,
                opacity=1.0,
                colorscale=colorscale,
                # colorbar=dict(thickness=20, title=cmap_title,orientation='h'),
                line=dict(width=0.5, color='#455A64')
            )
        )
        traces.append(data)

        ###########################
        # atoms's para_s
        color1 = para_s_i_c[para_name]
        data = go.Scatter3d(
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            text=textinfo,
            mode='markers',
            name=para_name + '_' + atomnames[i],
            opacity=1.0,
            marker=dict(
                sizemode='diameter',
                sizeref=50,
                size=14,
                color=color1,
                opacity=0.7,
                colorscale=colorscale,
                cmin=cMin,
                cmax=cMax,
                colorbar=dict(thickness=12,
                              ticklen=3,
                              tickcolor='black',
                              tickfont=dict(size=fontsize, family='Times New Roman', color='black'),
                              orientation='h',
                              # name='dddd'
                              # ticktext='ffffffff',
                              ),
                # line=dict(width=0.5, color='#455A64')
            )
        )
        traces.append(data)

    if len(labeled_index) > 0:

        def filter_x_para_s(x_s, labeledIndex):
            x_s_filtered = []
            for x_s_i in x_s:
                x_s_i_filtered = x_s_i[x_s_i.index.isin(labeledIndex)]
                x_s_filtered.append(x_s_i_filtered)
            return x_s_filtered

        def merge_df(x_s):
            combined_df = pd.concat(x_s)
            x_s_combined = combined_df.drop_duplicates()
            return x_s_combined

        pos_s_combined_filtered = merge_df(filter_x_para_s(pos_s, labeled_index))
        para_s_combined_filtered = merge_df(filter_x_para_s(para_s, labeled_index))

        line_opicity = '0.5'
        color_opicity = '0.5'

        data = go.Scatter3d(
            x=pos_s_combined_filtered['x'],
            y=pos_s_combined_filtered['y'],
            z=pos_s_combined_filtered['z'],
            text=para_s_combined_filtered[para_name].apply(info),
            mode='markers',
            name='labeled atoms',
            opacity=1.0,
            marker=dict(
                sizemode='diameter',
                sizeref=50,
                size=8,
                color='rgba(255,255,0,' + color_opicity + ')',
                line=dict(color='rgba(255,255,0,' + line_opicity + ')', width=1.0),
            )
        )
        traces.append(data)

    fig = go.Figure(data=traces)

    # scene_backgroundcolor='rgba(0, 0, 0, 0.1)'
    # scene_grid_color='black'

    if hide_background:
        scene_backgroundcolor = 'white'
    else:
        scene_backgroundcolor = None

    if hide_axis:
        # legend
        visibility = False
    else:
        visibility = True
    scene_grid_color = None
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        legend={'title': {'font': {'color': 'black', 'family': 'Times New Roman'}, 'text': legendstr}},
        # font=dict(family="Times New Roman",size=18,color="RebeccaPurple") # method1
        font_size=fontsize,
        font_color="black",
        font_family="Times New Roman",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black",
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        # paper_bgcolor="LightSteelBlue",
        paper_bgcolor="white",

        scene=dict(
            xaxis=dict(
                title=dict(font=dict(size=fontsize), text='X (Å)'),
                backgroundcolor=scene_backgroundcolor,
                gridcolor=scene_grid_color,
                showbackground=True,
                showticklabels=True,
                visible=visibility,
                zerolinecolor=scene_grid_color, ),
            yaxis=dict(
                title=dict(font=dict(size=fontsize), text='Y (Å)'),
                backgroundcolor=scene_backgroundcolor,
                gridcolor=scene_grid_color,
                showbackground=True,
                showticklabels=True,
                visible=visibility,
                zerolinecolor=scene_grid_color),
            zaxis=dict(
                title=dict(font=dict(size=fontsize), text='Z (Å)'),
                backgroundcolor=scene_backgroundcolor,
                gridcolor=scene_grid_color,
                showbackground=True,
                showticklabels=True,
                visible=visibility,
                zerolinecolor=scene_grid_color, ),
        ),

    )
    fig.layout.coloraxis.colorbar.title = 'another title'
    fig.update_xaxes(title_font_family="Times New Roman")
    if hide_legend:
        fig.update_layout(showlegend=False)
    return fig
