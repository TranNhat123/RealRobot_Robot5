import numpy as np
import pybullet as p
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
from function_robot.Robot6_observer import Robot6_observer

class Robot_6_Dof:
    def __init__(self, urdf_path, base_position, base_orientation_euler, use_fixed_base=True):
        """Khởi tạo robot 6 bậc trong PyBullet."""
        self.robot_id = p.loadURDF(
            urdf_path,
            base_position,
            p.getQuaternionFromEuler(base_orientation_euler),
            useFixedBase=use_fixed_base
        )

        # Link tool để lấy pose (ở code cũ là linkIndex = 5)
        self.tool_link_index = 5
        self.base_position = base_position
        self.link_markers = [] # Lưu ID của các marker
        self.marker_radii = [] # Lưu bán kính của từng marker
        self.robot_capsuls = []
        # Tự động xác định số điểm observer bằng cách gọi hàm với giá trị giả
        # Điều này giúp code đồng bộ khi file Robot6_observer.py thay đổi
        dummy_observer_points = Robot6_observer(0, 0, 0, 0, 0)

        # Các biến dùng cho luật điều khiển
        self.flag_take_longest_distance = False
        self.att_position_max = 0.0
        self.att_Rx = 0.0
        self.att_Rz = 0.0

        self.theta1 = 0
        self.theta2 = np.pi/2
        self.theta3 = 0
        self.theta4 = 0
        self.theta5 = 0
        self.theta6 = 0

        # Giới hạn tốc độ cho 5 khớp đầu
        self.joint_vel_limit = np.pi / 10.0
        # Góc khởi tạo cho robot 6 bậc
        initial_joint_angles = [0, np.pi / 2, 0, 0, 0, 0]
        for i, angle in enumerate(initial_joint_angles):
            p.resetJointState(self.robot_id, i, targetValue=angle)

        # ====== TẠO MARKER TẠI CÁC LINK ======
        self.num_observer = len(dummy_observer_points) # Cập nhật num_observer sau khi lấy dummy data
        for i in range(self.num_observer):
            # Tạo visual shape riêng cho mỗi marker để có thể có bán kính khác nhau
            radius = dummy_observer_points[i][1] # Lấy bán kính từ dữ liệu observer
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.5])
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0, 0, 0]
            )
            self.link_markers.append(marker_id)
            self.marker_radii.append(radius) # Lưu bán kính
        # =======================================

        self.marker_points = []
    # ------------------ Hàm xử lý robot ------------------ #

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

    def take_joint_position(self, client_socket):
        """Lấy vector góc khớp hiện tại của robot 6 bậc."""
        if client_socket is not None:
            data_1 = client_socket.recv(1024).decode()
            data_1 = data_1[:-1]
            numbers_list_1 = [float(num) for num in data_1.split(':')]
            t_1 = numbers_list_1[0]
            t_2 = numbers_list_1[1]
            t_3 = numbers_list_1[2]
            t_4 = numbers_list_1[3]
            t_5 = numbers_list_1[4]
            t_6 = numbers_list_1[5]
            positions = [t_1, t_2, t_3, t_4, t_5, t_6]
            self.theta1 = t_1
            self.theta2 = t_2
            self.theta3 = t_3   
            self.theta4 = t_4
            self.theta5 = t_5
            self.theta6 = t_6
        else:
            positions = [self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6]

        return np.array(positions)

    def set_joint_velocity(self, theta_v_sixdof, client_socket):
        if client_socket is not None:
            v1 = theta_v_sixdof[0]
            v2 = theta_v_sixdof[1]
            v3 = theta_v_sixdof[2]
            v4 = theta_v_sixdof[3]
            v5 = theta_v_sixdof[4]
            v6 = theta_v_sixdof[5]
            data_send_1 = str(v1) + ":" + str(v2) + ":" + str(v3) + ":" + str(v4) + ":" + str(v5) + ":" + str(v6)
            client_socket.send(data_send_1.encode())
        else: 
            self.theta1 = self.theta1 + theta_v_sixdof[0]*0.01
            self.theta2 = self.theta2 + theta_v_sixdof[1]*0.01  
            self.theta3 = self.theta3 + theta_v_sixdof[2]*0.01
            self.theta4 = self.theta4 + theta_v_sixdof[3]*0.01
            self.theta5 = self.theta5 + theta_v_sixdof[4]*0.01
            self.theta6 = self.theta6 + theta_v_sixdof[5]*0.01


    def get_theta_dot(self, x_target, y_target, z_target,v_target, Rx_target, Rz_target, vmax, joint_position_local):
        """
        Tính theta_dot dựa trên:
        - lỗi vị trí end-effector so với (x_target, y_target, z_target)
        - lỗi orientation Rx, Rz
        - Đơn vị là m /s cho vị trí và rad/s cho orientation
        - joint_position_local: vị trí khớp hiện tại của robot 6 bậc    
        - vmax: vận tốc tối đa ở không gian task (m/s)
        Trả về: theta_dot (vận tốc khớp), norm lỗi vị trí, lỗi Rx, lỗi Rz
                """
        K_1 = 10
        t1, t2, t3, t4, t5, t6 = joint_position_local
        # Jacobian tại tool
        Jtool_sixdof = Jtool_sixdof_function(t1, t2, t3, t4, t5, t6)

        # Tọa độ đích (goal)
        goal = np.array([float(x_target), float(y_target), float(z_target)]) # Base position
        
        observer_data = Robot6_observer(t1, t2, t3, t4, t5)
        self.marker_points = np.array([data[0] for data in observer_data]) + self.base_position
        # Lấy vị trí tool từ PyBullet
        pos = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.tool_link_index)[0]
        x, y, z = pos
        att_sixdof = np.array([x, y, z]) - goal

        v_att_tool_position_sixdof = - K_1*(att_sixdof) + v_target 
        v_att_tool_position_sixdof = np.clip(v_att_tool_position_sixdof, -vmax, vmax)


        # Hướng mong muốn cho Rx, Rz (ở đây đặt = 0)
        Rx_des = float(Rx_target)
        Rz_des = float(Rz_target)
        t4_des = 0
        Rx_now = t2 + t3 + t5
        Rz_now = -t1 + t6
        self.att_Rx = Rx_now - Rx_des
        self.att_Rz = Rz_now - Rz_des
        att_t4 = t4 - t4_des
        theta_v_t4 = -np.pi/20*np.sign(att_t4)
        if abs(att_t4) < 1e-3:
            theta_v_t4 = 0

         # Vận tốc hấp dẫn theo orientation
        v_att_tool_orientation = -50.0 * np.array([self.att_Rx, self.att_Rz])

        # Vector tốc độ mong muốn ở không gian task
        c_tool = np.hstack((v_att_tool_position_sixdof , v_att_tool_orientation))

        # Tính tốc độ khớp bằng pseudo-inverse Jacobian
        theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_sixdof), c_tool.T)

        for i in range(5):
            theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -self.joint_vel_limit, self.joint_vel_limit)

        # Ở code gốc: khớp 4 bị set 0, khớp 5,6 map lại
        theta_dot = np.array([
            theta_v_sixdof[0],
            theta_v_sixdof[1],
            theta_v_sixdof[2],
            theta_v_t4,
            theta_v_sixdof[3],
            theta_v_sixdof[4]
        ])

        return theta_dot, np.linalg.norm(att_sixdof), self.att_Rx, self.att_Rz
    
    def Update_visualization(self):
        p.resetJointState(self.robot_id, 0, targetValue=self.theta1)
        p.resetJointState(self.robot_id, 1, targetValue=self.theta2)
        p.resetJointState(self.robot_id, 2, targetValue=self.theta3)
        p.resetJointState(self.robot_id, 3, targetValue=self.theta4)
        p.resetJointState(self.robot_id, 4, targetValue= -self.theta5)
        p.resetJointState(self.robot_id, 5, targetValue=self.theta6)    

        # for i in range(6):
        #     joint_value = p.getJointState(self.robot_id, i)[0]
        #     if i == 4:
        #         joint_value = -joint_value  # đổi dấu nếu là khớp số 4
        #     positions.append(joint_value)

        ## Vì trục 5 bị ngược nên cả khi đẩy vị trí sang thì chỉnh lại dấu