import numpy as np
import pybullet as p
from function_robot.Jtool_fanuc_function import Jtool_fanuc_function
from function_robot.Robot5_observer import Robot5_observer
from function_robot.Robot5_capsuls import Robot5_capsuls

class Robot_5_Dof:
    def __init__(self, urdf_path, base_position, base_orientation_euler, use_fixed_base=True):
        self.robot_id = p.loadURDF(
            urdf_path,
            base_position,
            p.getQuaternionFromEuler(base_orientation_euler),
            useFixedBase=use_fixed_base
        )
        # Link tool để lấy pose (ở code cũ là linkIndex = 5)
        self.tool_link_index = 4
        self.joint_vel_limit = np.pi / 10.0
        self.base_position = base_position
        # Các biến dùng cho luật điều khiển
        self.flag_take_longest_distance = False
        self.att_position_max = 0.0
        self.att_Rx = 0.0
        self.att_Rz = 0.0

        # Góc khởi tạo cho robot 6 bậc
        initial_joint_angles = [0, np.pi / 2, -np.pi/2, np.pi/2, 0]
        for i, angle in enumerate(initial_joint_angles):
            p.resetJointState(self.robot_id, i, targetValue=angle)

        # ====== TẠO MARKER TẠI CÁC LINK (cho observer) ======
        self.link_markers = [] # Lưu ID của các marker
        self.marker_radii = [] # Lưu bán kính của từng marker
        # Tự động xác định số điểm observer bằng cách gọi hàm với giá trị giả
        dummy_observer_points = Robot5_observer(0, 0, 0, 0)
        self.num_observer = len(dummy_observer_points)

        for i in range(self.num_observer):
            # Tạo visual shape riêng cho mỗi marker để có thể có bán kính khác nhau
            radius = dummy_observer_points[i][1] # Lấy bán kính từ dữ liệu observer
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 1, 0.5])
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0, 0, 0]
            )
            self.link_markers.append(marker_id)
            self.marker_radii.append(radius) # Lưu bán kính

        dummy_capsule_points = Robot5_capsuls(0, 0, 0, 0)
        self.num_capsule = len(dummy_capsule_points)
        
        # Tính độ dài trung bình các đoạn nối
        dummy_pts = np.array(dummy_capsule_points)
        if len(dummy_pts) >= 2:
            lengths = [np.linalg.norm(dummy_pts[i+1] - dummy_pts[i]) for i in range(len(dummy_pts)-1)]
            self.capsule_segment_length = float(np.mean(lengths)) if lengths else 1.0
        else:
            self.capsule_segment_length = 1.0
        
        # configurable radius for the connecting cylinders (capsules)
        self.capsule_radius = [0.08, 0.08, 0.05, 0.05, 0.04]
        self.capsule_points_radius = [self.capsule_radius[0]] + self.capsule_radius
        
        # Màu sắc cho từng segment capsule (RGBA): [r, g, b, alpha]
        # Mỗi segment sẽ có 1 màu, segment i sẽ dùng capsule_colors[i]
        self.capsule_colors = [
            [1.0, 0.0, 0.0, 1],    # segment 0: Đỏ
            [0.0, 1.0, 0.0, 1],    # segment 1: Xanh lá
            [0.0, 0.0, 1.0, 1],    # segment 2: Xanh dương
            [1.0, 0.5, 0, 1],    # segment 3: Vàng
            [1.0, 0.0, 1.0, 1],    # segment 4: Tím
        ]
        
        for i in range(self.num_capsule):   # Chỉ có 5 capsule thôi nhưng sẽ có 6 điểm nên bị thiếu 1
            capsul_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.capsule_points_radius[i], rgbaColor=[1, 0, 0, 0.5])
            capsul_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=capsul_visual,
                basePosition=[0, 0, 0]
            )
            self.link_markers.append(capsul_id)## Maker_points

        self.marker_points = []
        self.capsule_points = []
        # cylinder markers connecting capsule points
        self.capsule_cylinders = []

        
    def update_link_markers(self):
        """
        Cập nhật vị trí các marker theo các điểm observer.
        """
        if self.marker_points is not None and len(self.marker_points) > 0:
            # Đảm bảo số marker khớp với số điểm
            num_points_to_draw = min(len(self.marker_points), len(self.link_markers))
            for i in range(num_points_to_draw): # marker_points giờ là list of (pos, radius)
                marker_id = self.link_markers[i]
                link_pos = self.marker_points[i]
                p.resetBasePositionAndOrientation(
                    marker_id,
                    link_pos,
                    [0, 0, 0, 1]
                )
    def update_capsule_markers(self):
        """
        Cập nhật vị trí các marker theo các điểm capsule.
        Vẽ sphere tại mỗi điểm và cylinder kết nối các điểm liên tiếp.
        Tái sử dụng (reuse) các bodies — chỉ tạo một lần, sau đó update vị trí.
        """
        if self.capsule_points is None or len(self.capsule_points) == 0:
            return

        # Cập nhật vị trí sphere tại mỗi điểm capsule
        start_index = self.num_observer  # Bắt đầu từ sau các marker observer
        num_points_to_draw = min(len(self.capsule_points), self.num_capsule)
        for i in range(num_points_to_draw):
            marker_id = self.link_markers[start_index + i]
            link_pos = self.capsule_points[i]
            p.resetBasePositionAndOrientation(marker_id, link_pos, [0, 0, 0, 1])

        # Chuẩn bị dữ liệu capsule
        pts = np.array(self.capsule_points)
        num_segments = max(0, len(pts) - 1)
        num_components = num_segments * 3  # mỗi segment: 1 cylinder + 2 sphere (trừ overlap)

        # Lần đầu tiên: tạo tất cả capsule bodies
        if len(self.capsule_cylinders) != num_components:
            # Xóa các bodies cũ
            for cid in self.capsule_cylinders:
                try:
                    p.removeBody(cid)
                except Exception:
                    pass
            self.capsule_cylinders = []

            # Tạo mới
            for i in range(num_segments):
                # Lấy màu cho segment này (với boundary check)
                color = self.capsule_colors[i] if i < len(self.capsule_colors) else [0.5, 0.5, 0.5, 0.7]
                
                # Cylinder (dùng capsule_segment_length cố định)
                try:
                    vis_cyl = p.createVisualShape(p.GEOM_CYLINDER, radius=self.capsule_radius[i], 
                                                   length=self.capsule_segment_length, rgbaColor=color)
                    cyl_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_cyl, 
                                                basePosition=[0, 0, 0])
                    self.capsule_cylinders.append(cyl_id)
                except Exception:
                    self.capsule_cylinders.append(None)

                # Sphere A
                try:
                    vis_sph = p.createVisualShape(p.GEOM_SPHERE, radius=self.capsule_radius[i], 
                                                   rgbaColor=color)
                    sph_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_sph, 
                                                basePosition=[0, 0, 0])
                    self.capsule_cylinders.append(sph_id)
                except Exception:
                    self.capsule_cylinders.append(None)

                # Sphere B (cuối cùng)
                try:
                    vis_sph = p.createVisualShape(p.GEOM_SPHERE, radius=self.capsule_radius[i], 
                                                   rgbaColor=color)
                    sph_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_sph, 
                                                basePosition=[0, 0, 0])
                    self.capsule_cylinders.append(sph_id)
                except Exception:
                    self.capsule_cylinders.append(None)

        # Cập nhật vị trí/hướng các bodies
        for i in range(num_segments):
            a = pts[i]
            b = pts[i + 1]
            vec = b - a
            length = float(np.linalg.norm(vec))
            
            if length > 1e-6:
                mid = (a + b) / 2.0
                dir_vec = vec / length

                # Tính quaternion
                z = np.array([0.0, 0.0, 1.0])
                dot = float(np.dot(z, dir_vec))
                dot = max(min(dot, 1.0), -1.0)
                angle = np.arccos(dot)
                axis = np.cross(z, dir_vec)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    quat = [0.0, 0.0, 0.0, 1.0] if dot > 0.0 else p.getQuaternionFromEuler([np.pi, 0, 0])
                else:
                    axis = axis / axis_norm
                    s = np.sin(angle / 2.0)
                    quat = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)]

                # Cập nhật cylinder
                cyl_idx = i * 3
                if cyl_idx < len(self.capsule_cylinders) and self.capsule_cylinders[cyl_idx] is not None:
                    p.resetBasePositionAndOrientation(self.capsule_cylinders[cyl_idx], mid.tolist(), quat)

                # Cập nhật sphere A
                sph_a_idx = i * 3 + 1
                if sph_a_idx < len(self.capsule_cylinders) and self.capsule_cylinders[sph_a_idx] is not None:
                    p.resetBasePositionAndOrientation(self.capsule_cylinders[sph_a_idx], a.tolist(), [0, 0, 0, 1])

                # Cập nhật sphere B
                sph_b_idx = i * 3 + 2
                if sph_b_idx < len(self.capsule_cylinders) and self.capsule_cylinders[sph_b_idx] is not None:
                    p.resetBasePositionAndOrientation(self.capsule_cylinders[sph_b_idx], b.tolist(), [0, 0, 0, 1])

    def take_joint_position(self):
        """Lấy vector góc khớp hiện tại của robot 5 bậc."""
        positions = []
        for i in range(5):
            joint_value = p.getJointState(self.robot_id, i)[0]
            positions.append(joint_value)
        return np.array(positions)

    def set_joint_velocity(self, theta_v_sixdof):
        """Set vận tốc cho từng khớp robot 6 bậc."""
        for i in range(5):
            velocity = theta_v_sixdof[i]
            p.setJointMotorControl2(
                self.robot_id,
                i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=500
            )
    
    def calculate_capsule_distance(self):
        """
        Tính khoảng cashc giữa các capsule với nhau để kiểm tra tự va chạm. 
        """
        # If not enough points, nothing to compute
        pts = np.array(self.capsule_points)
        if pts is None or pts.size == 0 or pts.shape[0] < 2:
            return float('inf'), None

        # Helper: shortest distance between two segments [p1,p2] and [q1,q2]
        def seg_seg_dist(p1, p2, q1, q2):
            u = p2 - p1
            v = q2 - q1
            w0 = p1 - q1
            a = np.dot(u, u)
            b = np.dot(u, v)
            c = np.dot(v, v)
            d = np.dot(u, w0)
            e = np.dot(v, w0)

            D = a * c - b * b
            sc, sN, sD = 0.0, D, D  # sc = sN / sD
            tc, tN, tD = 0.0, D, D

            SMALL_NUM = 1e-9

            # compute the line parameters of the two closest points
            if D < SMALL_NUM:
                # the lines are almost parallel
                sN = 0.0        # force using s = 0
                sD = 1.0        # to avoid division by 0
                tN = e
                tD = c
            else:
                # get the closest points on the infinite lines
                sN = (b * e - c * d)
                tN = (a * e - b * d)
                if sN < 0.0:
                    sN = 0.0
                    tN = e
                    tD = c
                elif sN > sD:
                    sN = sD
                    tN = e + b
                    tD = c

            if tN < 0.0:
                tN = 0.0
                if -d < 0.0:
                    sN = 0.0
                elif -d > a:
                    sN = sD
                else:
                    sN = -d
                    sD = a
            elif tN > tD:
                tN = tD
                if (-d + b) < 0.0:
                    sN = 0
                elif (-d + b) > a:
                    sN = sD
                else:
                    sN = (-d + b)
                    sD = a

            sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
            tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD

            # closest points
            cp = p1 + sc * u
            cq = q1 + tc * v
            dvec = cp - cq
            dist = np.linalg.norm(dvec)
            return dist, cp, cq

        n_pts = pts.shape[0]
        n_seg = n_pts - 1
        min_surface_dist = float('inf')
        best = None

        # For each pair of non-adjacent segments, compute segment-segment distance
        for i in range(n_seg):
            p1 = pts[i]
            p2 = pts[i + 1]
            for j in range(i + 2, n_seg):
                # skip adjacent segments (they share a point) — j starts at i+2
                q1 = pts[j]
                q2 = pts[j + 1]
                d_center, cp, cq = seg_seg_dist(p1, p2, q1, q2)
                # subtract radii
                r_sum = 0.0
                # if per-segment radii are available, use them; otherwise use single radius
                if hasattr(self, 'capsule_radius'):
                    r_sum = 2.0 * float(self.capsule_radius)
                else:
                    r_sum = 0.0
                surface_dist = d_center - r_sum
                if surface_dist < min_surface_dist:
                    min_surface_dist = surface_dist
                    best = {
                        'seg_i': (i, i + 1),
                        'seg_j': (j, j + 1),
                        'center_dist': d_center,
                        'surface_dist': surface_dist,
                        'closest_point_seg_i': cp,
                        'closest_point_seg_j': cq
                    }

        return min_surface_dist, best
     
    def get_theta_dot(self, x_target, y_target, z_target, vmax, joint_position_local):
        """
        Tính theta_dot dựa trên:
        - lỗi vị trí end-effector so với (x_target, y_target, z_target)
        - lỗi orientation Rx, Rz
        """
        t1, t2, t3, t4, t5 = joint_position_local
    
        # Lấy các điểm observer và bán kính của chúng
        observer_data = Robot5_observer(t1, t2, t3, t4)
        # Jacobian tại tool
        Jtool_fivedof = Jtool_fanuc_function(t1, t2, t3, t4)
        # Lưu chỉ tọa độ điểm cho marker_points
        self.marker_points = np.array([data[0] for data in observer_data]) + self.base_position
        self.capsule_points = Robot5_capsuls(t1, t2, t3, t4) + self.base_position

        # Tọa độ đích (goal)
        goal = np.array([float(x_target), float(y_target), float(z_target)])

        # Lấy vị trí tool từ PyBullet
        pos = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.tool_link_index)[0]
        x, y, z = pos
        att = np.array([x, y, z]) - goal

        # Lưu khoảng cách lớn nhất để chuẩn hóa việc tăng/giảm tốc
        if not self.flag_take_longest_distance:
            self.att_position_max = np.linalg.norm(att)
            self.flag_take_longest_distance = True

        # Hướng mong muốn cho Rx, Rz (ở đây đặt = 0)
        Rx_des = 0.0
        Rz_des = 0.0

        Rx_now = t2 + t3 + t4
        Rz_now = -t1 + t5
        self.att_Rx = Rx_now - Rx_des
        self.att_Rz = Rz_now - Rz_des

        # Tính phần trăm quãng đường đã đi để điều chỉnh vận tốc
        percent = 1.0 - np.linalg.norm(att) / max(self.att_position_max, 1e-6)

        if percent <= 0.1:
            v_position_sixdof = max(vmax * percent / 0.1, vmax * 0.5)
        elif percent < 0.9:
            v_position_sixdof = vmax
        else:
            v_position_sixdof = vmax * (1.0 - percent) / 0.1

        # Vận tốc hấp dẫn theo vị trí
        v_att_tool_position = -v_position_sixdof * (att / np.linalg.norm(att))
        # Vận tốc hấp dẫn theo orientation
        v_att_tool_orientation = -50.0 * np.array([self.att_Rx, self.att_Rz])

        # Vector tốc độ mong muốn ở không gian task
        c_tool = np.hstack((v_att_tool_position, v_att_tool_orientation))

        # Tính tốc độ khớp bằng pseudo-inverse Jacobian
        theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_fivedof), c_tool.T)

        for i in range(5):
            theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -self.joint_vel_limit, self.joint_vel_limit)

        # Ở code gốc: khớp 4 bị set 0, khớp 5,6 map lại
        theta_dot = np.array([
            theta_v_sixdof[0],
            theta_v_sixdof[1],
            theta_v_sixdof[2],
            theta_v_sixdof[3],
            theta_v_sixdof[4]
        ])

        return theta_dot, np.linalg.norm(att), self.att_Rx, self.att_Rz

    def calculate_capsule_pair_distances(self, pair_indices=None):
        pts = np.array(self.capsule_points)
        num_segments = len(pts) - 1
        segments = [(pts[i], pts[i + 1]) for i in range(num_segments)]

        # Assume uniform capsule_radius exists
        radii = self.capsule_radius
        if pair_indices is None:
            pair_indices = [(i, j) for i in range(num_segments) for j in range(i + 2, num_segments)]

        def seg_seg_dist(p1, q1, p2, q2): # This function should return 3 values
            # Note: p1, q1 define segment 1; p2, q2 define segment 2
            u = q1 - p1 # Direction vector of segment 1
            v = q2 - p2 # Direction vector of segment 2
            w0 = p1 - p2 # Vector between start points
            a = np.dot(u, u)
            b = np.dot(u, v)
            c = np.dot(v, v)
            d = np.dot(u, w0)
            e = np.dot(v, w0)
            D = a * c - b * b
            SMALL_NUM = 1e-7

            # compute the line parameters of the two closest points
            if D < SMALL_NUM: # the lines are almost parallel
                sc = 0.0
                if c > SMALL_NUM:
                    tc = e / c
                else:
                    tc = 0.0
            else:
                sc = (b * e - c * d) / D
                tc = (a * e - b * d) / D
            sc = np.clip(sc, 0.0, 1.0)
            tc = np.clip(tc, 0.0, 1.0)
            c1 = p1 + sc * u
            c2 = p2 + tc * v
            return np.linalg.norm(c1 - c2), c1, c2

        d_list = []
        for (i, j) in pair_indices:
            p1, q1 = segments[i]
            p2, q2 = segments[j]
            d_raw, c1, c2 = seg_seg_dist(np.array(p1), np.array(q1), np.array(p2), np.array(q2))
            surf = d_raw - (radii[i] + radii[j])
            d_list.append(float(surf))

        return np.array(d_list, dtype=np.float32)

    def calculate_capsule_pair_distances_with_points(self, pair_indices=None):
        """
        Tính khoảng cách bề mặt giữa các cặp đoạn capsule và trả cả điểm gần nhất trên mỗi đoạn.
        Trả về 3 mảng: distances (surface), closest_point_on_seg1, closest_point_on_seg2
        """
        pts = np.array(self.capsule_points)
        num_segments = len(pts) - 1
        if num_segments < 2:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        segments = [(pts[i], pts[i + 1]) for i in range(num_segments)]
        radii = self.capsule_radius # Use the list of radii

        if pair_indices is None:
            pair_indices = [(i, j) for i in range(num_segments) for j in range(i + 2, num_segments)]

        def seg_seg_dist_points(p1, q1, p2, q2): # p1-q1 is seg1, p2-q2 is seg2
            u = q1 - p1
            v = q2 - p2
            w0 = p1 - p2
            a = np.dot(u, u)
            b = np.dot(u, v)
            c = np.dot(v, v)
            d = np.dot(u, w0)
            e = np.dot(v, w0)
            D = a * c - b * b
            SMALL = 1e-7
            if D < SMALL:
                sc = 0.0
                tc = e / c if c > SMALL else 0.0
            else:
                sc = (b * e - c * d) / D
                tc = (a * e - b * d) / D
            sc = np.clip(sc, 0.0, 1.0)
            tc = np.clip(tc, 0.0, 1.0)
            c1 = p1 + sc * u
            c2 = p2 + tc * v
            return float(np.linalg.norm(c1 - c2)), c1, c2

        d_list = []
        c1_list = []
        c2_list = []
        for (i, j) in pair_indices:
            p1, q1 = segments[i]
            p2, q2 = segments[j]
            d_raw, c1, c2 = seg_seg_dist_points(np.array(p1), np.array(q1), np.array(p2), np.array(q2))
            surf = d_raw - (radii[i] + radii[j])
            d_list.append(float(surf))
            c1_list.append(c1)
            c2_list.append(c2)

        return np.array(d_list, dtype=np.float32), np.array(c1_list, dtype=np.float32), np.array(c2_list, dtype=np.float32)
    
if __name__ == "__main__":
    # Test robot 5 dof
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath("D:/Data_save_obsidian/Hoc_tap/Lab/Master thesis/Code/Train_fanuc/urdf/")
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0]
    startOrientation = [0, 0, 0]
    robot5 = Robot_5_Dof("robot5dof.urdf", startPos, startOrientation)
    robot6 = Robot_5_Dof("robot6dof.urdf", [2,0,0], startOrientation)
