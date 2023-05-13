import uuid

import nglview as nv
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import pyvista as pv
import tqdm
from scipy.spatial import distance

from .PlotCustomizer import *

pv.set_jupyter_backend('pythreejs')


class SurfaceTools:
    def position_s(self, positions_df):
        # pos of all atom for selected frame
        positions = []
        for df in positions_df:
            pos = df[['x', 'y', 'z']].to_numpy()
            positions.append(pos)
        return positions

    def test_alpha_outlier_removal(self, positions, frame=0, nb_points=20, radius=10, info=False):
        areas = []
        volumes = []
        alphas = range(120)[::1]
        # print(times[frame],'ps')
        for alpha in tqdm.tqdm(alphas):
            pos = positions[frame]
            pv.PolyData(pos).save('/tmp/pointcloud.ply', binary=False, texture=None)
            pcd = o3d.io.read_point_cloud('/tmp/pointcloud.ply')
            # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
            # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.5)

            # #### Alpha shapes (convex hull method)
            # pcd=voxel_down_pcd
            alpha = alpha
            # ################### outlier removal #############################################################
            # filtering the outsider atoms  ### important！！！！！
            cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)  # Radius outlier removal
            # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0) # Statistical outlier removal
            pcd = cl

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.compute_vertex_normals()
            try:
                o3d.io.write_triangle_mesh('/tmp/tmp.ply', mesh, write_ascii=True, compressed=False,
                                           write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True,
                                           print_progress=False)
                meshx = pv.read('/tmp/tmp.ply')
                areas.append(meshx.area)
                volumes.append(meshx.volume)
            except:
                if info:
                    print('write ply failed, ot enough points to create a tetrahedral mesh?frame: ', frame)
                areas.append(0)
                volumes.append(0)

        fig = plt.figure(figsize=(8, 4))
        ax = plt.subplot(121)
        ax.plot(alphas, areas)
        ax.set_xlabel('alphas')
        ax.set_ylabel('surface area(Å$^2$)')

        ax = plt.subplot(122)
        ax.plot(alphas, volumes)
        ax.set_xlabel('alphas')
        ax.set_ylabel('volumes(Å$^3$)')
        plt.suptitle('xxxx')
        fig.show()

    def test_nb_points_radius_outlier_removal(self, positions, frame=0, alpha=75, info=False):
        areas = []
        volumes = []
        radius_s = range(20)[1::1]
        nb_points_s = range(100)[1::1]

        volume_map = []
        surfacearea_map = []
        # print(times[frame],'ps')
        for radius in tqdm.tqdm(radius_s):
            for nb_points in nb_points_s:
                pos = positions[frame]
                pv.PolyData(pos).save('/tmp/pointcloud.ply', binary=False, texture=None)
                pcd = o3d.io.read_point_cloud('/tmp/pointcloud.ply')
                # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
                # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.5)

                # #### Alpha shapes (convex hull method)
                # pcd=voxel_down_pcd
                alpha = alpha
                # ################### outlier removal #############################################################
                # filtering the outsider atoms  ### important！！！！！
                cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)  # Radius outlier removal
                # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0) # Statistical outlier removal
                pcd = cl

                try:
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                    mesh.compute_vertex_normals()

                    o3d.io.write_triangle_mesh('/tmp/tmp.ply', mesh, write_ascii=True, compressed=False,
                                               write_vertex_normals=True, write_vertex_colors=True,
                                               write_triangle_uvs=True, print_progress=False)
                    meshx = pv.read('/tmp/tmp.ply')
                    areas.append(meshx.area)
                    volumes.append(meshx.volume)
                    volume_map.append([radius, nb_points, meshx.volume])
                    surfacearea_map.append([radius, nb_points, meshx.area])
                except:
                    if info:
                        print('write ply failed, not enough points to create a tetrahedral mesh?frame: ', frame)
                    volume_map.append([radius, nb_points, 0])
                    surfacearea_map.append([radius, nb_points, 0])

        z = np.asarray(surfacearea_map)[:, -1]
        x = np.asarray(surfacearea_map)[:, 0]
        y = np.asarray(surfacearea_map)[:, 1]

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))])

        # Set plot title and axes labels
        fig.update_layout(title=f"alpha={alpha}",
                          scene=dict(xaxis_title='radius', yaxis_title='nb_points', zaxis_title='surfacearea_map'))

        fig.update_layout(autosize=False,
                          width=800, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        # Show the plot
        # fig.show()

        z = np.asarray(volume_map)[:, -1]
        x = np.asarray(volume_map)[:, 0]
        y = np.asarray(volume_map)[:, 1]

        # Create a 3D scatter plot
        fig2 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))])

        # Set plot title and axes labels
        fig2.update_layout(title=f"alpha={alpha}",
                           scene=dict(xaxis_title='radius', yaxis_title='nb_points', zaxis_title='volume_map'))

        fig2.update_layout(autosize=False,
                           width=800, height=800,
                           margin=dict(l=65, r=50, b=65, t=90))
        # Show the plot
        # fig2.show()
        return fig, fig2

    def surface_mesh_s(self, positions, alpha=75, nb_points=20, radius=10):
        # pos of surface atom by alpha_shape algriithm
        surface_pos = []  # pos of surface atom by alpha_shape algriithm
        mesh_s = []

        # for pos in tqdm(positions):
        for pos in tqdm.tqdm(positions):
            uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
            alpha = alpha
            pv.PolyData(pos).save(uuidname, binary=False, texture=None)
            pcd = o3d.io.read_point_cloud(uuidname)

            # voxel_size=1
            # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
            # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            #         ##### Ball pivoting (surface reconstruction method)
            #         distances = pcd1.compute_nearest_neighbor_distance()
            #         avg_dist = np.mean(distances)
            #         radius = 3 * avg_dist
            #         radii = [radius, radius * 2, radius * 4, radius * 8]

            # pcd.estimate_normals() mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
            # o3d.utility.DoubleVector(radii)) mesh.compute_vertex_normals()

            # ##### Poisson (surface reconstruction method) pcd.estimate_normals() ## estimate pcd mesh, densities =
            # o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=1, width=1, scale=0.5,
            # linear_fit=False)

            # ################### outlier removal #################
            # filtering the outsider atoms 去除飞散出的单原子/小原子簇
            cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)  # Radius outlier removal
            # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0) # Statistical outlier removal
            pcd = cl

            # #### Alpha shapes (convex hull method)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
            mesh.compute_vertex_normals()
            # print(uuidname)
            surfacepos_i = np.asarray(mesh.vertices)
            # result = [surfacepos_i,SerializableMesh(mesh)]
            surface_pos.append(surfacepos_i)
            mesh_s.append(mesh)
        return surface_pos, mesh_s

    def surface_indice(self, positions_df, surface_pos):
        # inbdices of every surface atom
        surface_indices = []
        for index, df_i in enumerate(tqdm.tqdm(positions_df)):
            rslt_df = df_i.loc[df_i['x'].isin(surface_pos[index][:, 0])]
            surface_indices_i = rslt_df['id'].to_list()
            surface_indices.append(surface_indices_i)
        return surface_indices

    def mesh_area_volume_s(self, mesh_s):
        # mesh area/volume of every frame
        mesh_area_s = []
        mesh_volume_s = []
        for mesh_i in tqdm.tqdm(mesh_s):
            vv, meshx = self.visual_mesh(mesh_i, opacity=0.8)
            mesh_area_s.append(meshx.area)
            mesh_volume_s.append(meshx.volume)
        return mesh_area_s, mesh_volume_s

    def visual_mesh_with_corner(self, mesh, opacity=0.8):
        o3d.io.write_triangle_mesh('/tmp/tmp.ply', mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read('/tmp/tmp.ply')

        pl = pv.Plotter(window_size=(800, 800))
        meshx.texture_map_to_plane(inplace=True)

        # plot mesh
        pts = meshx.points.copy()
        pts -= pts.min()
        rgba_sphere = (255 * pts).astype(np.uint8)
        # pl.add_mesh(meshx,show_edges=True,color='purple',style='wireframe', render_points_as_spheres=True,
        # point_size=100,opacity=0.1,)
        pl.add_mesh(meshx, scalars=rgba_sphere, rgba=True, smooth_shading=True, show_edges=True, style='surface',
                    pbr=True, opacity=opacity, roughness=0.2, metallic=0.3)
        pl.add_mesh(meshx, show_edges=True, color='purple', style='points', render_points_as_spheres=True,
                    point_size=100, opacity=0.1, )

        # plot corner
        corners = meshx.outline_corners()
        pts = corners.points.copy()
        pts -= pts.min()
        print(len(pts))
        pts = 255 * (pts / pts.max())  # Make 0-255 RGBA values
        corners['rgba_values'] = pts.astype(np.uint8)
        edges = corners.tube(radius=0.1).triangulate()
        pl.add_mesh(edges, rgba=True, smooth_shading=True, color='indigo')
        pl.add_mesh(edges, rgba=True, smooth_shading=True, color='indigo')  # 可以指定颜色，也可以使用生成的rgb values
        pl.background_color = 'white'
        return pl, meshx

    # without corner
    def visual_mesh(self, mesh, opacity=0.8):
        o3d.io.write_triangle_mesh('/tmp/tmp.ply', mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read('/tmp/tmp.ply')

        pl = pv.Plotter()
        meshx.texture_map_to_plane(inplace=True)

        meshxc = meshx.copy()
        meshxc.point_data['myscalars'] = range(len(meshxc.points))

        pl.add_mesh(meshxc, cmap='viridis', show_scalar_bar=True, show_edges=True, pbr=True, opacity=opacity,
                    roughness=0.2, metallic=0.3)  # cmap='viridis  coolwarm
        pl.add_mesh(meshx, show_edges=True, color='purple', style='points', render_points_as_spheres=True,
                    point_size=100)
        # pl.add_mesh(meshx,show_edges=True,color='black',style='wireframe', render_points_as_spheres=True,
        # point_size=100)
        pl.background_color = 'white'
        return pl, meshx

    def visual_mesh_2(self, mesh_s, positions, frame=0, opacity=0.8, pointcolor='black', whirecolor='orange'):
        mesh = mesh_s[frame]

        o3d.io.write_triangle_mesh('/tmp/tmp.ply', mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read('/tmp/tmp.ply')

        pl = pv.Plotter()
        meshx.texture_map_to_plane(inplace=True)

        meshxc = meshx.copy()
        meshxc.point_data['myscalars'] = range(len(meshxc.points))

        # pl.add_mesh(meshxc,cmap='viridis',pbr=True, opacity=0.9,roughness=0.2,metallic=0.1) # cmap='viridis  coolwarm
        pl.add_mesh(meshxc, cmap='gist_stern_r', show_edges=True, opacity=opacity, pbr=True, roughness=0.1,
                    metallic=0.3)  # cmap='viridis  coolwarm
        pl.add_mesh(meshx, style='wireframe', line_width=1, color=whirecolor)

        point_cloud = pv.PolyData(positions[frame])
        pl.add_points(point_cloud, point_size=100.0, color=pointcolor)
        # pl.add_mesh(meshx,style='points',point_size=200,color='blue')
        pl.background_color = 'white'
        return pl, meshx

    # Open the output file for writing
    def visualize_3d_numpy_array(self, points):
        with open('/tmp/1.xyz', 'w') as f:
            num_points = len(points)
            # Write the number of points to the first line of the file
            f.write(str(num_points) + '\n')
            f.write('comments' + '\n')
            # Write each point to a separate line in the file
            for i in range(num_points):
                x, y, z = points[i]
                f.write('c {:.6f} {:.6f} {:.6f}\n'.format(x, y, z))
        import MDAnalysis as mda
        v = nv.show_file(mda.Universe('/tmp/1.xyz'))
        return v

    def indicing_shell_atoms(self, atomgroup, shell_thickness=8, alpha=75, nb_points=20, radius=10):
        # ################## 使用alpha shhape 找到表面锚点，并标记
        atoms = atomgroup

        points_pos = atoms.positions
        indices_all = atoms.indices
        positions = [points_pos, ]

        labeled_surface_points, mesh_s = self.surface_mesh_s(positions, alpha=alpha, nb_points=nb_points, radius=radius)
        # surface_indices = surface_indice(positions_df,surface_pos)
        mesh_area_s, mesh_volume_s = self.mesh_area_volume_s(mesh_s)

        # clearn_up_tmp_dir(patens=['*.dcd','*.ply',])
        # !ls /tmp
        print(f'标记的表面锚点数目:{len(labeled_surface_points[0])},表面锚点mesh覆盖区域:{mesh_area_s}')

        # ################### 依据标记的锚点，找到距其一定范围内的点
        surface_points = np.asarray(labeled_surface_points[0])  # 表面锚点

        # Compute the distances between each point in the cloud and the shell points
        distances = distance.cdist(surface_points, points_pos)

        # Find the minimum distance for each point
        min_distances = np.min(distances, axis=0)

        # Label all points within the shell with the shell thickness of 5
        points_within_shell = np.where(min_distances <= shell_thickness, True, False)

        labeled_shell_points = []
        for i, x in enumerate(points_within_shell):
            if x:
                labeled_shell_points.append(points_pos[i])
        labeled_shell_points = np.asarray(labeled_shell_points)

        # ########### 匹配到maanalysis Universe对象中的坐标，
        print('匹配表面锚点和shell原子indices...')
        idx_s_all = []
        for s_points in [surface_points, labeled_shell_points]:
            # 匹配表面锚点和shell原子indices
            idx_s = []
            # slower
            # for idx,pos_i in enumerate(points_pos):
            #     for labeled_pos_i in labeled_points:
            #         vec1=labeled_pos_i
            #         vec2=pos_i
            #         dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
            #         if dist <= 0.2:
            #             idx_s.append(idx)
            # faster
            # 匹配
            for idx, pos_i in enumerate(tqdm.tqdm(points_pos)):
                for labeled_pos_i in s_points:
                    x1, y1, z1 = labeled_pos_i
                    x, y, z = pos_i
                    delta_limit = 0.01
                    if abs(x - x1) <= delta_limit:
                        if abs(y - y1) <= delta_limit:
                            if abs(z - z1) <= delta_limit:
                                idx_s.append(idx)

            print(f'共遍历原子{len(points_pos)}个，匹配到表面锚点{len(idx_s)}个')
            if len(s_points) == len(idx_s):
                print('匹配无误')
            else:
                print('匹配出错')
                print(f'需匹配原子{len(s_points)}个，匹配到{len(idx_s)}个')
            idx_s_all.append(idx_s)

        # ############# 找到对应的真实indice
        idx_surface, idx_shell = idx_s_all

        indice_surface = []
        indice_shell = []
        for idx, indice in enumerate(indices_all):
            if idx in idx_shell:
                indice_shell.append(indice)
            if idx in idx_surface:
                indice_surface.append(indice)
        return indice_surface, indice_shell
