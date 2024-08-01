import uuid
import os
import nglview as nv
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import pyvista as pv
import tqdm
from sklearn.cluster import KMeans,DBSCAN
from scipy.spatial import distance,cKDTree
from .PlotCustomizer import *


# Isolating Open3D operations: Import Open3D within each function to allow for normal parallel operations
# This ensures that each process does its independent import, avoiding issues caused by shared state between processes.
class SurfaceTools:
    def surface_indice(self, positions_df, surface_pos):
        # inbdices of every surface atom
        surface_indices = []
        for index, df_i in enumerate(tqdm.tqdm(positions_df)):
            rslt_df = df_i.loc[df_i['x'].isin(surface_pos[index][:, 0])]
            surface_indices_i = rslt_df['id'].to_list()
            surface_indices.append(surface_indices_i)
        return surface_indices

    def visual_mesh_with_corner(self, mesh, opacity=0.8):
        pv.set_jupyter_backend('pythreejs')
        uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
        o3d.io.write_triangle_mesh(uuidname, mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read(uuidname)
        os.remove(uuidname)

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

    def visual_mesh_2(self, mesh_s, positions, frame=0, opacity=0.8, pointcolor='black', whirecolor='orange'):
        pv.set_jupyter_backend('pythreejs')
        mesh = mesh_s[frame]
        uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'

        o3d.io.write_triangle_mesh(uuidname, mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read(uuidname)
        os.remove(uuidname)

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

    def position_s(self, positions_df):
        # pos of all atom for selected frame
        positions = []
        for df in positions_df:
            pos = df[['x', 'y', 'z']].to_numpy()
            positions.append(pos)
        return positions

    def test_alpha_outlier_removal(self, positions, frame=0, nb_points=20, radius=10, info=False):
        """
        :param positions:
        :param frame:
        :param nb_points: pcd.remove_radius_outlier which lets you pick the minimum amount of points that the sphere should contain.
        :param radius: pcd.remove_radius_outlier  which defines the radius of the sphere that will be used for counting the neighbors.
        :param info:
        :return:
        """
        areas = []
        volumes = []
        alphas = range(120)[::1]
        # print(times[frame],'ps')
        for alpha in tqdm.tqdm(alphas):
            pos = positions[frame]
            uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
            pv.PolyData(pos).save(uuidname, binary=False, texture=None)
            pcd = o3d.io.read_point_cloud(uuidname)
            os.remove(uuidname)

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
                uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
                o3d.io.write_triangle_mesh(uuidname, mesh, write_ascii=True, compressed=False,
                                           write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True,
                                           print_progress=False)
                meshx = pv.read(uuidname)
                os.remove(uuidname)
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
        """
        上方地带dou'sh
        :param positions:
        :param frame:
        :param alpha: o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape A smaller alpha value will result in a more detailed and complex shape, while a larger alpha value will result in a simpler shape with fewer edges and vertices.
        :param info:
        :return:
        """
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
                uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
                pv.PolyData(pos).save(uuidname, binary=False, texture=None)
                pcd = o3d.io.read_point_cloud(uuidname)
                os.remove(uuidname)
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
                    uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
                    o3d.io.write_triangle_mesh(uuidname, mesh, write_ascii=True, compressed=False,
                                               write_vertex_normals=True, write_vertex_colors=True,
                                               write_triangle_uvs=True, print_progress=False)
                    meshx = pv.read(uuidname)
                    os.remove(uuidname)
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

    # ############################################################################################################
    def surface_mesh_s(self, positions, alpha=75, nb_points=20, radius=10, debug=False):
        # pos of surface atom by alpha_shape algriithm
        surface_pos = []  # pos of surface atom by alpha_shape algriithm
        mesh_s = []

        # for pos in tqdm(positions):
        for pos in positions:
            uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
            # if debug:
            #     print(f'save pos to {uuidname}')
            pv.PolyData(pos).save(uuidname, binary=False, texture=None)
            # if debug:
            #     print(f'reading pcd from ply {uuidname}')
            pcd = o3d.io.read_point_cloud(uuidname)
            # if debug:
            #     print(f'rm ply file')
            os.remove(uuidname)

            # voxel_size=1
            # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
            # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            #         ##### Ball pivoting (surface reconstruction method)
            #         distancess = pcd1.compute_nearest_neighbor_distance()
            #         avg_dist = np.mean(distancess)
            #         radius = 3 * avg_dist
            #         radii = [radius, radius * 2, radius * 4, radius * 8]

            # pcd.estimate_normals() mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
            # o3d.utility.DoubleVector(radii)) mesh.compute_vertex_normals()

            # ##### Poisson (surface reconstruction method) pcd.estimate_normals() ## estimate pcd mesh, densities =
            # o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=1, width=1, scale=0.5,
            # linear_fit=False)

            # ################### outlier removal #################
            # filtering the outsider atoms 去除飞散出的单原子/小原子簇
            if debug:
                print(f'points in the point cloud{len(pcd.points)}, nb_points{nb_points}, radius{radius}/n')
            if len(pcd.points)<nb_points:
                print(f'Error: points in the point cloud{len(pcd.points)}, nb_points{nb_points}, radius{radius}/n')

            # pcd_new, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)  # Radius outlier removal
            ##################
            import threading

            def remove_radius_outlier_with_timeout(pcd, nb_points, radius, timeout):
                def remove_outliers():
                    nonlocal pcd
                    pcd_new, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
                    pcd = pcd_new

                # Start the command in a separate thread
                thread = threading.Thread(target=remove_outliers)
                thread.start()
                thread.join(timeout)  # Wait for the thread to finish or timeout
                if thread.is_alive():
                    print('kill due to timeout')
                    # If the thread is still alive, it means the command has exceeded the timeout
                    thread.terminate()  # You need to implement the 'terminate' method for the thread
                    thread.join()  # Clean up the terminated thread

            pcd_new = pcd  # Replace 'pcd' with your point cloud data
            nb_points = 100  # Replace 'nb_points' with the desired value
            radius = 0.1  # Replace 'radius' with the desired value
            timeout = 5  # Set the timeout in seconds
            remove_radius_outlier_with_timeout(pcd_new, nb_points, radius, timeout)
            ##################

            # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0) # Statistical outlier removal

            if debug:
                print(f'computing_vertex_normals')
            # #### Alpha shapes (convex hull method)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_new, alpha=alpha)
            # print('TriangleMeshdone ')
            mesh.compute_vertex_normals()

            surfacepos_i = np.asarray(mesh.vertices)
            # result = [surfacepos_i,SerializableMesh(mesh)]
            surface_pos.append(surfacepos_i)
            mesh_s.append(mesh)

        return surface_pos, mesh_s

    def visual_mesh(self, mesh, opacity=0.8):
        pv.set_jupyter_backend('pythreejs')
        uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
        o3d.io.write_triangle_mesh(uuidname, mesh, write_ascii=True, compressed=False, write_vertex_normals=True,
                                   write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        meshx = pv.read(uuidname)
        os.remove(uuidname)

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

    def mesh_area_volume_s(self, mesh_s):
        # mesh area/volume of every frame
        mesh_area_s = []
        mesh_volume_s = []
        for mesh_i in mesh_s:
            vv, meshx = self.visual_mesh(mesh_i, opacity=0.8)
            mesh_area_s.append(meshx.area)
            mesh_volume_s.append(meshx.volume)
        return mesh_area_s, mesh_volume_s

    def indicing_shell_atoms(self, atomgroup, shell_ranges=None, eps=3, min_samples=3, alpha=75, nb_points=20, radius=19, debug=False, frame=None):

        def refine_surface_points(surface_points, masscenter, eps=eps, min_samples=min_samples):
            def dbscan_classification(eps, min_samples):
                distances = np.linalg.norm(surface_points - masscenter, axis=1)
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(distances.reshape(-1, 1))
                labels = db.labels_
                # Assuming the largest cluster is the outer surface
                unique, counts = np.unique(labels, return_counts=True)
                largest_cluster = unique[np.argmax(counts)]
                return labels, largest_cluster

            labels, largest_cluster = dbscan_classification(eps, min_samples)
            MultiSurface = False

            surface_points_refined = surface_points[labels == largest_cluster]
            if len(surface_points) != len(surface_points_refined):
                MultiSurface = True
                print('smllest_cluster',len(surface_points)-len(surface_points_refined))
                print(f'Warning! Multiple surfaces were detected, refining the surface_points...{frame}')
                if frame:
                    print(f'frame {frame}, from {surface_points.shape} to {surface_points_refined.shape}')
                else:
                    print(f'from {surface_points.shape} to {surface_points_refined.shape}')
            return surface_points_refined, MultiSurface

        def surface_mesh_s(positions, alpha=75, nb_points=20, radius=10, debug=False):
            # pos of surface atom by alpha_shape algorithm
            surface_pos = []  # pos of surface atom by alpha_shape algorithm
            mesh_s = []

            for pos in positions:
                uuidname = '/tmp/pick_' + str(uuid.uuid4()) + '.ply'
                pv.PolyData(pos).save(uuidname, binary=False, texture=None)
                pcd = o3d.io.read_point_cloud(uuidname)
                os.remove(uuidname)

                if len(pcd.points) < nb_points:
                    print(f'Error: points in the point cloud {len(pcd.points)}, nb_points {nb_points}, radius {radius}')

                pcd_new = pcd

                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_new, alpha=alpha)
                mesh.compute_vertex_normals()

                surfacepos_i = np.asarray(mesh.vertices)
                surface_pos.append(surfacepos_i)
                mesh_s.append(mesh)

            return surface_pos, mesh_s

        # Main body of the indicing_shell_atoms function
        if shell_ranges is None:
            print('using default shell_ranges = [[0, 3], [1, 3], [3, 5]]')
            shell_ranges = [[0, 3], [1, 3], [3, 5]]
        atoms = atomgroup

        atoms.universe.trajectory[frame]
        points_pos = atoms.positions
        indices_all = atoms.indices
        positions = [points_pos, ]
        masscenter = atoms.center_of_mass()

        labeled_surface_points, mesh_s = surface_mesh_s(positions, alpha, nb_points, radius, debug)

        surface_points = np.asarray(labeled_surface_points[0])  # surface anchor points coordinates

        surface_points, MultiSurface = refine_surface_points(surface_points, masscenter, eps=eps, min_samples=min_samples)

        # Compute the distances between each point in the cloud and the shell points
        distancess = distance.cdist(surface_points, points_pos)

        # Find the minimum distance for each point
        min_distances = np.min(distancess, axis=0)

        indice_shell_s = []
        surfaceAtomSearched = False

        #     for ii, shell_range in enumerate(shell_ranges):
        #         shell_thickness_range_min, shell_thickness_range_max = shell_range
        #         points_within_shell = np.logical_and(min_distances >= shell_thickness_range_min, min_distances <= shell_thickness_range_max)

        #         labeled_shell_points = []
        #         for i, x in enumerate(points_within_shell):
        #             if x:
        #                 labeled_shell_points.append(points_pos[i])
        #         labeled_shell_points = np.asarray(labeled_shell_points)  # positions

        #         # Match to real indices in the atomgroup
        #         if not surfaceAtomSearched:  # search anchor atom only once
        #             idx_surface = [i for i, pos in enumerate(points_pos) if any(np.linalg.norm(pos - pt) < 0.01 for pt in surface_points)]
        #             indice_surface = [indices_all[i] for i in idx_surface]
        #         surfaceAtomSearched = True

        #         idx_shell = [i for i, pos in enumerate(points_pos) if any(np.linalg.norm(pos - pt) < 0.01 for pt in labeled_shell_points)]
        #         indice_shell = [indices_all[i] for i in idx_shell]

        #         indice_shell_s.append(indice_shell)

        kdtree = cKDTree(points_pos)
        indice_shell_s = []
        surfaceAtomSearched = False
        for shell_range in shell_ranges:
            shell_thickness_range_min, shell_thickness_range_max = shell_range
            points_within_shell = np.logical_and(min_distances >= shell_thickness_range_min, min_distances <= shell_thickness_range_max)

            # Using boolean indexing to get labeled_shell_points
            labeled_shell_points = points_pos[points_within_shell]

            if not surfaceAtomSearched:  # search anchor atom only once
                _, idx_surface = kdtree.query(surface_points, distance_upper_bound=0.01)
                # Filter out invalid indices
                idx_surface = idx_surface[idx_surface != len(points_pos)]
                # Convert to real atom indices
                indice_surface = [indices_all[i] for i in idx_surface]
                surfaceAtomSearched = True

            _, idx_shell = kdtree.query(labeled_shell_points, distance_upper_bound=0.01)
            # Filter out invalid indices
            idx_shell = idx_shell[idx_shell != len(points_pos)]
            # Convert to real atom indices
            indice_shell = [indices_all[i] for i in idx_shell]

            indice_shell_s.append(indice_shell)

        return indice_surface, indice_shell_s, MultiSurface

    def indicing_shell_atoms_old(self, atomgroup, shell_ranges=None, alpha=75, nb_points=20, radius=19, debug=False,
                             frame=None, threshold=None):
        """
        :param threshold: max distance difference to determine if there is multisurface in detected surface points
        :param atomgroup:
        :param shell_ranges:[[0, 3], [1, 3], [3, 5]]
        :param alpha: o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape A smaller alpha value will result in a more detailed and complex shape, while a larger alpha value will result in a simpler shape with fewer edges and vertices.
        :param nb_points: pcd.remove_radius_outlier which lets you pick the minimum amount of points that the sphere should contain.
        :param radius: pcd.remove_radius_outlier  which defines the radius of the sphere that will be used for counting the neighbors.
        :return:
        """
        # ################## 使用alpha shhape 找到表面锚点，并标记
        if shell_ranges is None:
            print('using default shell_ranges = [[0, 3], [1, 3], [3, 5]]')
            shell_ranges = [[0, 3], [1, 3], [3, 5]]
        atoms = atomgroup

        points_pos = atoms.positions
        indices_all = atoms.indices
        positions = [points_pos, ]
        masscenter = atoms.center_of_mass()
        if debug:
            print('Start meshing')
            labeled_surface_points, mesh_s = self.surface_mesh_s(positions, alpha, nb_points, radius, debug=True)
        else:
            labeled_surface_points, mesh_s = self.surface_mesh_s(positions, alpha, nb_points, radius, debug=False)
        if debug:
            print(f'get mesh area and vol')
        # surface_indices = surface_indice(positions_df,surface_pos)
        mesh_area_s, mesh_volume_s = self.mesh_area_volume_s(mesh_s)

        # clearn_up_tmp_dir(patens=['*.dcd','*.ply',])
        # !ls /tmp
        if debug:
            print(f'标记的表面锚点数目:{len(labeled_surface_points[0])},表面锚点mesh覆盖区域:{mesh_area_s}')

        # ################### 依据标记的锚点，找到距其一定范围内的点
        surface_points = np.asarray(labeled_surface_points[0])  # 表面锚点坐标
        # 以最外层厚度的1/2作为阈值
        if not threshold:
            # threshold = (shell_ranges[0][0] - shell_ranges[0][1]) / 2
            threshold = shell_ranges[0][1] - shell_ranges[0][0]
        # 防止出现多个surface的情况: 如ANP空腔的表面
        def refine_surface_points(surface_points, masscenter, threshold):
            # 注意，以下代码只针对两个surface做了处理
            def kmean_classification(n_clusters=2):
                # Calculate the distances
                distances = np.linalg.norm(surface_points - masscenter, axis=1)
                # Create a k-means object
                kmeans = KMeans(n_clusters, n_init='auto')
                # Fit the data
                kmeans.fit(distances.reshape(-1, 1))
                # Get the cluster labels
                if debug:
                    print(f'Get the cluster labels')
                labels = kmeans.labels_
                # 将surface points分为两类, 如果两类原子的平均径向距离相差较大，说明识别到了多个表面
                # Find the cluster label that has the most occurrences
                largest_cluster = np.bincount(labels).argmax()
                smallest_cluster = np.bincount(labels).argmin()
                largest_distances_points = distances[labels == largest_cluster]
                smallest_distances_points = distances[labels == smallest_cluster]
                return labels, largest_cluster, smallest_cluster, largest_distances_points, smallest_distances_points

            labels, largest_cluster, smallest_cluster, largest_distances_points, smallest_distances_points = kmean_classification(n_clusters=2)

            if debug:
                print(f'MultiSurface')
            MultiSurface = False
            if largest_distances_points.mean() - smallest_distances_points.mean() > threshold:
                MultiSurface = True
                print(f'Warning! Multiple surfaces were detected, current threshold {threshold}, refining the surface_points...')
                surface_points_refined = surface_points[labels == largest_cluster]
                if frame:
                    print(f'frame {frame}, from {surface_points.shape} to {surface_points_refined.shape}')
                else:
                    print(f'from {surface_points.shape} to {surface_points_refined.shape}')
            else:
                surface_points_refined = surface_points
            return surface_points_refined, MultiSurface

        surface_points, MultiSurface = refine_surface_points(surface_points, masscenter, threshold)
        if debug:
            print(f'surface_points, MultiSurface')
        # Compute the distances between each point in the cloud and the shell points
        distancess = distance.cdist(surface_points, points_pos)

        # Find the minimum distance for each point
        min_distances = np.min(distancess, axis=0)  # 每个点与离他最近的锚点的距离

        # Label all points within the shell with the shell thickness
        # points_within_shell = np.where(shell_thickness_range_min <= min_distances <= shell_thickness_range_max, True, False)
        indice_shell_s = []
        surfaceAtomSearched = False
        if debug:
            print(f'for ii, shell_range in enumerate(shell_ranges):')
        for ii, shell_range in enumerate(shell_ranges):
            shell_thickness_range_min, shell_thickness_range_max = shell_range
            points_within_shell = np.logical_and(min_distances >= shell_thickness_range_min,
                                                 min_distances <= shell_thickness_range_max)

            labeled_shell_points = []
            for i, x in enumerate(points_within_shell):
                if x:
                    labeled_shell_points.append(points_pos[i])
            labeled_shell_points = np.asarray(labeled_shell_points)  # positions

            # ########### 匹配到mdanalysis Universe对象中的坐标，
            def match_atom_indices(s_points):
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
                for idx, pos_i in enumerate(points_pos):
                    for labeled_pos_i in s_points:
                        x1, y1, z1 = labeled_pos_i
                        x, y, z = pos_i
                        delta_limit = 0.01
                        if abs(x - x1) <= delta_limit:
                            if abs(y - y1) <= delta_limit:
                                if abs(z - z1) <= delta_limit:
                                    idx_s.append(idx)

                if len(s_points) == len(idx_s):
                    if debug:
                        print(f'共遍历原子{len(points_pos)}个，匹配到{len(idx_s)}个原子')
                else:
                    print('匹配出错')
                    print(f'需匹配原子{len(s_points)}个，匹配到{len(idx_s)}个')
                return idx_s

            # ############# 找到对应的真实indice
            if not surfaceAtomSearched:  # 只搜索一次锚点原子
                if debug:
                    print('# 匹配 表面锚点原子 indices...')
                idx_surface = match_atom_indices(surface_points)
                indice_surface = []
                for idx, indice in enumerate(indices_all):
                    if idx in idx_surface:
                        indice_surface.append(indice)
            surfaceAtomSearched = True

            indice_shell = []
            if debug:
                print(f'# 匹配 第 {ii} 层shell原子 indices, Current shell range: {shell_range}')
            idx_shell = match_atom_indices(labeled_shell_points)
            for idx, indice in enumerate(indices_all):
                if idx in idx_shell:
                    indice_shell.append(indice)

            indice_shell_s.append(indice_shell)
        return indice_surface, indice_shell_s, MultiSurface
