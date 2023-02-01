import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
from open3d.visualization import O3DVisualizer
import open3d as o3d
import numpy as np
from open3d import*
import pdb

def Simplest():
    # source_data = np.load('curtain_0088.npy')[:,0:3]  #10000x3
    source_data = np.random.randint(-100, high=100, size=(1000, 3))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(source_data)
    o3d.visualization.draw_geometries([point_cloud])


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = o3d.visualization.O3DVisualizer("Open3D1 - 3D Text", 1920, 1080)
    app.add_window(vis)
    # vis.show_settings = True
    vis.add_geometry("Points", points)
    vis.enable_raw_mode(True)
    # vis.add_geometry("Points", points)
    for idx in range(0, len(points.points)):
        vis.add_3d_label(points.points[idx], "{}".format(idx))
    vis.setup_camera(180.0, np.array([0.0, 0.0, 0.0]).reshape(3, 1),
                     np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    vis.show_skybox(False)
    vis.ground_plane = rendering.Scene.GroundPlane.XY
    vis.show_axes = True
    # vis.post_redraw()
    # vis.reset_camera_to_default()
    vis.show(True)

    # vis2
    # vis2 = o3d.visualization.O3DVisualizer("Open3D2 - 3D Text", 1024, 768)
    # vis2.show_settings = True
    # vis2.add_geometry("Points", points)
    # for idx in range(0, len(points.points)):
    #     vis2.add_3d_label(points.points[idx], "{}".format(idx))
    # vis2.reset_camera_to_default()
    # vis2.show(True)
    # app.add_window(vis2)

    app.run()


def low_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()

    widget3d.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, len(points.points)):
        l = widget3d.add_3d_label(points.points[idx], "{}".format(idx))
        l.color = gui.Color(points.colors[idx][0], points.colors[idx][1],
                            points.colors[idx][2])
        l.scale = np.random.uniform(0.5, 3.0)
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()


if __name__ == "__main__":
    # pdb.set_trace()
    print("This is high level")
    high_level()
    # print("This is low level")
    # low_level()
    # print("This is simplest")
    # Simplest()
