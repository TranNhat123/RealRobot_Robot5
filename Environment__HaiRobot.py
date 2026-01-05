import numpy as np
import pybullet as p
import pybullet_data
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from Robot5 import Robot_5_Dof
from Robot6 import Robot_6_Dof
from collections import deque
from function_robot.Robot5_capsuls import Robot5_capsuls
from function_robot.Robot6_observer import Robot6_observer
from function_robot.Robot5_observer import Robot5_observer

"""
Môi trường này, chứa hai robot và không có vật cản. 
Hai robot tránh va chạm lẫn nhau. Biểu đồ chỉ cần d_min, d_self của robot5 là đủ. (Thí nghiệm 3)
"""
class Environment:
    def __init__(self, use_gui=True, use_matplotlib=True, ax=None, draw_callback=None):
        """Khởi tạo PyBullet, mặt phẳng, robot và quả cầu target."""
        """Đây là môi trường để sau này deploy chương trình"""
        self.use_gui = use_gui
        self.use_matplotlib = use_matplotlib
        self.draw_callback = draw_callback

        # Kết nối PyBullet
        if use_gui:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Tắt tính năng kéo thả chuột (Mouse Picking) để ngăn tương tác vật lý
            # p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Mặt phẳng
        self.plane_id = p.loadURDF("plane.urdf")

        self.base6 = np.array([0.0, -0.75, 0.0])
        # Robot 6 bậc
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=self.base6,
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        self.base5 = np.array([0.0, 0, 0.0])
        # Robot 5 bậc
        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=self.base5,
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        # Quả cầu hiển thị mục tiêu
        self.sphere_robot6 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[1, 0, 0, 1]
            ),
            basePosition=[0, 0, 0]
        )

                # Quả cầu hiển thị mục tiêu
        self.sphere_robot5 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[0, 0, 1, 1]
            ),
            basePosition=[0, 0, 0]
        )

        self.workspace_data = np.array([0, 0, 0.2, 0.7]) # x, y, z, radius

        # Tham số mô phỏng và điều khiển
        self.vmax_robot6 = 0.05 # (m/s)
        self.vmax_robot5 = 0.05  # (m/s)
        self.dt = 1.0 / 240.0

        # Target mặc định
        self.Px_robot6 = 0.0
        self.Py_robot6 = 0.0
        self.Pz_robot6 = 0.0
        self.Rx_robot6 = 0.0
        self.Rz_robot6 = 0.0

                # Target mặc định
        self.Px_robot5 = 0.0
        self.Py_robot5 = 0.0
        self.Pz_robot5 = 0.0
        self.Rx_robot5 = 0.0
        self.Rz_robot5 = 0.0
        # 0 - real time, 1 - non-real time
        self.flag_mode_simulation = 1

        # --- Tham số cho CBF ---
        # Robot6 sẽ được coi là chướng ngại vật
        self.use_cbf = False # Mặc định tắt, sẽ bật khi người dùng yêu cầu
        self.d_safe = 0.05   # Ngưỡng an toàn (m)
        self.cbf_gamma = 10.0 # Hệ số gamma cho CBF
        self.last_debug_lines = []

        # --- Logging ---
        self.log_filename = "joint_log.txt"
        self.log_file = open(self.log_filename, "w")
        # Ghi tiêu đề cho file log
        header = "time_step,t1,t2,t3,t4,t5,time_step,t6,t7,t8,t9,t10,t11,Case_fanuc,Case_sixdof\n"
        self.log_file.write(header)
        self.time_step_counter = 0

        # --- Plotting Setup ---
        if self.use_matplotlib:
            if ax:
                self.ax = ax
                self.fig = ax.figure
            else:
                self.fig, self.ax = plt.subplots()
                plt.ion()
                plt.show()

            self.ax.set_title("Real-time Distance Monitoring")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Distance (m)")
            self.line_dmin, = self.ax.plot([], [], 'r-', label='d_inter (Robot-Robot)')
            self.line_dself, = self.ax.plot([], [], 'b-', label='d_self (Self-Collision)')
            self.line_safe, = self.ax.plot([], [], 'g--', label='d_safe')
            self.ax.legend(loc='upper right')
            self.ax.grid(True)
        # Sử dụng deque để tối ưu hóa hiệu năng (O(1) cho popleft)
        self.x_data = deque(maxlen=500)
        self.y_dmin = deque(maxlen=500)
        self.y_dself = deque(maxlen=500)
        
        # Cấu hình hiển thị đồ thị
        self.plot_update_freq = 20  # Giảm tần suất vẽ đồ thị để tiết kiệm tài nguyên (mỗi 20 bước)
        self.plot_window_time = 10.0 # Thời gian hiển thị trên đồ thị (giây)
        
        self.start_time = None

        self.time_step = 0.0

    # ------------------ Cài đặt target & mode ------------------ #

    def set_target_robot6(self, Px, Py, Pz, Rx, Rz):
        self.Px_robot6 = float(Px) # Bỏ chia 1000, giả định đầu vào là mét
        self.Py_robot6 = float(Py)
        self.Pz_robot6 = float(Pz)
        self.Rx_robot6 = float(Rx)
        self.Rz_robot6 = float(Rz)
    def set_target_robot5(self, Px, Py, Pz, Rx, Rz):
        self.Px_robot5 = float(Px) # Bỏ chia 1000, giả định đầu vào là mét
        self.Py_robot5 = float(Py)
        self.Pz_robot5 = float(Pz)
        self.Rx_robot5 = float(Rx)
        self.Rz_robot5 = float(Rz)

    def set_cbf_status(self, use_cbf):
        self.use_cbf = bool(use_cbf)

    def set_mode(self, flag_mode_simulation):
        """
        0 - real time
        1 - non-real time (mình sẽ tự stepSimulation)
        """
        self.flag_mode_simulation = int(flag_mode_simulation)
        if self.flag_mode_simulation == 0:
            # real-time
            p.setRealTimeSimulation(1)
        else:
            # non-real-time
            p.setRealTimeSimulation(0)

    # --- Các hàm cho CBF ---
    def _get_min_dist_between_robots(self, q5_angles, q6_angles):
        """Tính khoảng cách bề mặt tối thiểu giữa các observer sphere của robot5 và robot6."""
        # Lấy thông tin các điểm observer của robot5 và robot6
        Robot5_observer_data = Robot5_observer(q5_angles[0], q5_angles[1], q5_angles[2], q5_angles[3])
        pts5 = np.array([point for point, R in Robot5_observer_data]) + self.robot5.base_position
        R_pts5 = np.array([R for point, R in Robot5_observer_data])

        Robot6_observer_data = Robot6_observer(q6_angles[0], q6_angles[1], q6_angles[2], q6_angles[3], q6_angles[4])
        pts6 = np.array([point for point, R in Robot6_observer_data]) + self.robot6.base_position
        R_pts6 = np.array([R for point, R in Robot6_observer_data])

        min_dist_overall = float('inf')
        closest_points_info = {}

        # --- Tối ưu hóa bằng Vectorization (NumPy Broadcasting) ---
        # Tính khoảng cách giữa mọi cặp điểm (N x M) cùng lúc
        # pts5: (N, 3), pts6: (M, 3) -> diff: (N, M, 3)
        diff = pts5[:, np.newaxis, :] - pts6[np.newaxis, :, :]
        center_dists = np.linalg.norm(diff, axis=2)
        
        # Tính khoảng cách bề mặt: dist - (r1 + r2)
        radii_sum = R_pts5[:, np.newaxis] + R_pts6[np.newaxis, :]
        surface_dists = center_dists - radii_sum
        
        # Tìm giá trị nhỏ nhất
        min_dist_overall = np.min(surface_dists)

        # Chỉ tính toán chi tiết điểm va chạm nếu cần thiết (khi khoảng cách nhỏ)
        # để tiết kiệm thời gian tính toán cho các trường hợp an toàn
        if min_dist_overall < 1.0: # Ngưỡng tối ưu, chỉ tính chi tiết khi gần va chạm
            idx_flat = np.argmin(surface_dists)
            i, j = np.unravel_index(idx_flat, surface_dists.shape)
            
            p5_c, p6_c = pts5[i], pts6[j]
            c_dist = center_dists[i, j]
            
            if c_dist > 1e-6:
                direction_vec = (p5_c - p6_c) / c_dist
                p_r6_surface = p6_c + direction_vec * R_pts6[j]
                p_r5_surface = p5_c - direction_vec * R_pts5[i]
            else:
                p_r6_surface, p_r5_surface = p6_c, p5_c

            closest_points_info = {
                'p_robot': p_r5_surface,
                'p_obstacle': p_r6_surface,
                'dist': min_dist_overall
            }

        return min_dist_overall, closest_points_info

    def _get_dist_jacobian_numerical(self, q5_current, q6_current, epsilon=1e-5):
        """Tính Jacobian của khoảng cách tối thiểu theo góc khớp robot5 bằng phương pháp số."""
        d_current, _ = self._get_min_dist_between_robots(q5_current, q6_current)
        jacobian = np.zeros(len(q5_current))

        for i in range(len(q5_current)):
            q_perturbed = q5_current.copy()
            q_perturbed[i] += epsilon
            d_perturbed, _ = self._get_min_dist_between_robots(q_perturbed, q6_current)
            jacobian[i] = (d_perturbed - d_current) / epsilon

        return jacobian, d_current
    
    def solve_cbf_qp(self, u_nom, d_min, jacobian):
        """Hàm cũ, giải QP cho 1 ràng buộc. Sẽ được thay thế bằng _solve_cbf_qp."""
        act_dim = len(u_nom)
        u = cp.Variable(act_dim)
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

        h = d_min - self.d_safe
        constraints = [jacobian @ u >= -self.cbf_gamma * h]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            print("[Warning] CBF-QP solver failed. Using clipped nominal action.")
            return np.clip(u_nom, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)
        
        if u.value is not None:
            return u.value
        else:
            print("[Warning] CBF-QP infeasible. Using clipped nominal action.")
            return np.clip(u_nom, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)

    def _solve_cbf_qp(self, u_nom, constraints, u_var):
        """Giải bài toán QP của CBF với các ràng buộc cho trước để tìm hành động an toàn."""
        act_dim = len(u_nom)
        # u_var là biến cvxpy được truyền từ bên ngoài
        objective = cp.Minimize(cp.sum_squares(u_var - u_nom))

        # Thêm ràng buộc vận tốc khớp
        # constraints.append(u_var >= -self.robot5.joint_vel_limit)
        # constraints.append(u_var <= self.robot5.joint_vel_limit)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            print("[Warning] CBF-QP solver failed. Using clipped nominal action.")
            return np.clip(u_nom, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)

        if u_var.value is not None:
            return u_var.value
        else:
            print("[Warning] CBF-QP infeasible. Using clipped nominal action.")
            return np.clip(u_nom, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)
        
    def draw_safety_line(self, closest_points_info, d_min):
        """Vẽ đường debug thể hiện khoảng cách an toàn."""
        if not closest_points_info:
            return

        p1 = closest_points_info['p_robot']
        p2 = closest_points_info['p_obstacle']

        # Xanh: an toàn, Đỏ: nguy hiểm
        color = [1, 0, 0] if d_min < self.d_safe else [0, 1, 0]
        width = 10

        # Xóa các đường cũ và vẽ đường mới
        for line_id in self.last_debug_lines:
            p.removeUserDebugItem(line_id)
        self.last_debug_lines.clear()

        line_id = p.addUserDebugLine(p1.tolist(), p2.tolist(), lineColorRGB=color, lineWidth=width)
        self.last_debug_lines.append(line_id)

    def _get_self_dist_jacobian_numerical_robot5(self, q5_current, epsilon=1e-5):
        """Tính Jacobian của khoảng cách tự va chạm của Robot5."""
        d_current = self._get_min_self_dist_robot5(q5_current)
        jacobian = np.zeros(len(q5_current))

        for i in range(len(q5_current)):
            q_perturbed = q5_current.copy()
            q_perturbed[i] += epsilon
            d_perturbed = self._get_min_self_dist_robot5(q_perturbed)
            jacobian[i] = (d_perturbed - d_current) / epsilon
        
        return jacobian, d_current

    def _get_min_self_dist_robot5(self, q5_angles):
        """Tính khoảng cách bề mặt tối thiểu giữa các capsule của chính Robot5."""
        # Cập nhật các điểm capsule bên trong đối tượng robot5 cho cấu hình khớp q5_angles
        # Ta cần gọi một phần của logic `get_theta_dot` để cập nhật `self.robot5.capsule_points`
        # Chuyển đổi kết quả của Robot5_capsuls thành numpy array trước khi cộng với base5
        self.robot5.capsule_points = np.array(Robot5_capsuls(q5_angles[0], q5_angles[1], q5_angles[2], q5_angles[3])) + self.base5
        
        # SỬA LỖI: Gọi hàm calculate_capsule_pair_distances thay vì hàm cũ bị lỗi.
        # Hàm này trả về một danh sách các khoảng cách bề mặt cho các cặp capsule không liền kề.
        surface_distances = self.robot5.calculate_capsule_pair_distances()

        # Tìm khoảng cách nhỏ nhất từ danh sách. Nếu danh sách rỗng, trả về vô cùng.
        if surface_distances.size > 0:
            min_dist = np.min(surface_distances)
        else:
            min_dist = float('inf')
        return min_dist
    
    # ------------------ Một bước mô phỏng ------------------ #
    def step(self, client_socket_robot6, client_socket_robot5):
        if self.start_time is None:
            self.start_time = time.time()

        # Xóa các đường debug cũ của CBF, các đường khác (nếu có) sẽ không bị ảnh hưởng
        for line_id in self.last_debug_lines:
            p.removeUserDebugItem(line_id)
        self.last_debug_lines.clear()

        #### Robot6 (coi như chướng ngại vật động) ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot6 = self.robot6.take_joint_position(client_socket = client_socket_robot6)
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi

        # Controller cho robot 6
        theta_dot_robot6, pos_err_robot6, att_Rx_robot6, att_Rz_robot6 = self.robot6.get_theta_dot(
            x_target=self.Px_robot6,
            y_target=self.Py_robot6,
            z_target=self.Pz_robot6,
            v_target = 0,
            vmax=self.vmax_robot6,
            Rx_target=self.Rx_robot6,
            Rz_target=self.Rz_robot6,
            joint_position_local=joint_position_robot6, 
        )

        # Điều kiện dừng cho robot 6
        if pos_err_robot6 < 1e-3 and abs(att_Rx_robot6) < 1e-3 and abs(att_Rz_robot6) < 1e-3:
            theta_dot_robot6 = np.zeros(6)

        # Gửi lệnh vận tốc cho robot 6
        self.robot6.set_joint_velocity(theta_dot_robot6, client_socket = client_socket_robot6)
        self.robot6.Update_visualization()
        # Cập nhật vị trí quả cầu mục tiêu của robot 6
        p.resetBasePositionAndOrientation(
            self.sphere_robot6,
            [self.Px_robot6, self.Py_robot6, self.Pz_robot6],
            [0, 0, 0, 1]
        )

        #### Robot5 ##########
        #### Robot5 ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot5 = self.robot5.take_joint_position(client_socket = client_socket_robot5)
        joint_position_robot5 = (joint_position_robot5 + np.pi) % (2 * np.pi) - np.pi

        # Controller
        # 1. Tính hành động danh nghĩa (nominal) từ controller gốc
        theta_dot_robot5, pos_err_robot5, att_Rx_robot5, att_Rz_robot5 = self.robot5.get_theta_dot(
            x_target=self.Px_robot5,
            y_target=self.Py_robot5,
            z_target=self.Pz_robot5,
            v_target= 0, 
            Rx_target=self.Rx_robot5,
            Rz_target=self.Rz_robot5,
            vmax=self.vmax_robot5,
            joint_position_local=joint_position_robot5
        )

        # Điều kiện dừng cho robot 5: Nếu đã đến đích, vận tốc danh nghĩa là 0
        # Nhưng vẫn cho phép CBF điều chỉnh nếu cần tránh va chạm
        if pos_err_robot5 < 1e-3 and abs(att_Rx_robot5) < 1e-3 and abs(att_Rz_robot5) < 1e-3:
            theta_dot_robot5 = np.zeros(5)

        # --- Update Plot every 10 steps to save performance ---
        if self.use_matplotlib and self.time_step_counter % self.plot_update_freq == 0:
            # Calculate distances for plotting
            d_inter_plot, _ = self._get_min_dist_between_robots(joint_position_robot5, joint_position_robot6)
            d_self_plot = self._get_min_self_dist_robot5(joint_position_robot5)
            if d_self_plot == float('inf'): d_self_plot = 1.0 # Clip if no collision detected

            current_time = time.time() - self.start_time
            self.x_data.append(current_time)
            self.y_dmin.append(d_inter_plot)
            self.y_dself.append(d_self_plot)
            
            # Keep window size
            while self.x_data and (self.x_data[-1] - self.x_data[0] > self.plot_window_time):
                self.x_data.popleft()
                self.y_dmin.popleft()
                self.y_dself.popleft()
            
            self.line_dmin.set_data(self.x_data, self.y_dmin)
            self.line_dself.set_data(self.x_data, self.y_dself)
            
            if self.x_data:
                self.line_safe.set_data([self.x_data[0], self.x_data[-1]], [self.d_safe, self.d_safe])
                self.ax.set_xlim(self.x_data[0], self.x_data[-1])
                self.ax.set_ylim(0, max(max(self.y_dmin), max(self.y_dself), self.d_safe) + 0.1)
            
            if self.draw_callback:
                self.draw_callback()
            else:
                plt.pause(0.00001)

       # 2. Áp dụng CBF nếu được bật
        if self.use_cbf:
            # --- Ràng buộc 1: Va chạm giữa 2 robot (inter-collision) ---
            jacobian_inter, d_min_inter = self._get_dist_jacobian_numerical(joint_position_robot5, joint_position_robot6) # Tính toán d_min_inter
            h_inter = d_min_inter - self.d_safe

            # --- Ràng buộc 2: Tự va chạm của Robot5 (self-collision) ---
            # Có thể dùng một ngưỡng an toàn khác cho tự va chạm nếu muốn, ví dụ: self.d_safe_self
            jacobian_self, d_min_self = self._get_self_dist_jacobian_numerical_robot5(joint_position_robot5) # Tính toán d_min_self
            h_self = d_min_self - self.d_safe 

            # --- Xây dựng và giải bài toán QP với nhiều ràng buộc ---
            # Định nghĩa biến tối ưu hóa CHO CẢ HAI ràng buộc
            u_var = cp.Variable(len(theta_dot_robot5))
            constraints = []

            # Chỉ thêm ràng buộc khi robot ở gần vùng nguy hiểm
            if h_inter < 0.1: # Vùng ảnh hưởng của CBF va chạm ngoài
                # SỬA LỖI: Sử dụng cùng một biến u_var cho tất cả các ràng buộc
                constraints.append(jacobian_inter @ u_var >= -self.cbf_gamma * h_inter)
            if h_self < 0.1: # Vùng ảnh hưởng của CBF tự va chạm
                constraints.append(jacobian_self @ u_var >= -self.cbf_gamma * h_self)

            # Giải QP để có hành động an toàn
            safe_theta_dot = self._solve_cbf_qp(theta_dot_robot5, constraints, u_var)

            # Lấy thông tin điểm gần nhất để vẽ đường debug cho va chạm ngoài
            _, closest_points_info = self._get_min_dist_between_robots(joint_position_robot5, joint_position_robot6)
            # self.draw_safety_line(closest_points_info, d_min_inter)
            print(f"\rCBF Active | d_inter: {d_min_inter:.3f}m | d_self: {d_min_self:.3f}m", end="")

        else:
            safe_theta_dot = theta_dot_robot5

        # 3. Gửi lệnh vận tốc (an toàn)
        safe_theta_dot = np.clip(safe_theta_dot, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)
        self.robot5.set_joint_velocity(safe_theta_dot, client_socket=client_socket_robot5)

        ###########################################################################
        #################################################################################
        self.robot5.Update_visualization()
        #################################################################################
        #################################################################################
        # Cập nhật vị trí quả cầu mục tiêu
        # Cập nhật vị trí quả cầu mục tiêu của robot5
        p.resetBasePositionAndOrientation(
            self.sphere_robot5,
            [self.Px_robot5, self.Py_robot5, self.Pz_robot5],
            [0, 0, 0, 1]
        )
        #################################################################################
        #################################################################################

        # >>> Cập nhật vị trí 6 điểm quan sát trên link
        # self.robot6.update_link_markers()
        # self.robot5.update_link_markers()

        # --- Ghi dữ liệu góc khớp ra file ---
        # Định dạng theo yêu cầu: time_step,t1..t5,time_step,t6..t11,case_fanuc,case_sixdof
        log_data = []
        log_data.append(f"{self.time_step_counter * self.dt:.6f}") # time_step
        log_data.extend([f"{q:.6f}" for q in joint_position_robot5]) # t1-t5
        log_data.append(f"{self.time_step_counter * self.dt:.6f}") # time_step (lặp lại)
        log_data.extend([f"{q:.6f}" for q in joint_position_robot6]) # t6-t11
        log_data.append("1") # Case_fanuc
        log_data.append("1") # Case_sixdof

        self.log_file.write(",".join(log_data) + "\n")
        self.time_step_counter += 1

        
        # Nếu non-real-time thì phải tự step
        if self.flag_mode_simulation == 1:
            p.stepSimulation()
            # time.sleep(self.dt)  # nếu muốn chậm lại
            if self.use_gui:
                pass # time.sleep(self.dt)  # Đã bỏ sleep để tăng tốc độ tối đa

        return [pos_err_robot5, att_Rx_robot5, att_Rz_robot5], [pos_err_robot6, att_Rx_robot6, att_Rz_robot6]


    def step_circle(self, client_socket_robot6, client_socket_robot5):
        if self.start_time is None:
            self.start_time = time.time()

        self.time_step += 0.01
        # Tham số quỹ đạo tròn cho robot5
        self.R_target_robot5 = 0.3  # Bán kính quỹ đạo (m)
        self.omega_target_robot5 = 0.1  # Tốc độ góc (rad/s)
        self.x_0_target_robot5 = 0.25  # Tọa độ x tâm quỹ đạo (m)
        self.y_0_target_robot5 = 0  # Tọa độ y tâm quỹ đạo (m)   

        # Tham số quỹ đạo tròn cho robot5
        self.R_target_robot6 = 0.15  # Bán kính quỹ đạo (m)
        self.omega_target_robot6 = 0.3  # Tốc độ góc (rad/s)
        self.x_0_target_robot6 = 0  # Tọa độ x tâm quỹ đạo (m)
        self.y_0_target_robot6 = -0.4  # Tọa độ y tâm quỹ đạo (m)   

        # Xóa các đường debug cũ của CBF, các đường khác (nếu có) sẽ không bị ảnh hưởng
        for line_id in self.last_debug_lines:
            p.removeUserDebugItem(line_id)
        self.last_debug_lines.clear()

        #### Robot6 (coi như chướng ngại vật động) ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot6 = self.robot6.take_joint_position(client_socket = client_socket_robot6)
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi

        ## Quỹ đạo target
        x_target_robot6 = self.R_target_robot6*np.sin(self.omega_target_robot6*self.time_step) + self.x_0_target_robot6  
        y_target_robot6 = self.R_target_robot6*np.cos(self.omega_target_robot6*self.time_step) + self.y_0_target_robot6
        v_target_robot6 = [self.R_target_robot6*self.omega_target_robot6*np.cos(self.omega_target_robot6*self.time_step), 
                    -self.R_target_robot6*self.omega_target_robot6*np.sin(self.omega_target_robot6*self.time_step), 0]
        z_target_robot6 = 0.3
        Rx_target_robot6 = 0.0
        Rz_target_robot6 = 0.0
        # Controller cho robot 6
        self.set_target_robot6(x_target_robot6, y_target_robot6, z_target_robot6, Rx_target_robot6, Rz_target_robot6)
        theta_dot_robot6, pos_err_robot6, att_Rx_robot6, att_Rz_robot6 = self.robot6.get_theta_dot(
            x_target=self.Px_robot6,
            y_target=self.Py_robot6,
            z_target=self.Pz_robot6,
            v_target = v_target_robot6,
            vmax=self.vmax_robot6,
            Rx_target=self.Rx_robot6,
            Rz_target=self.Rz_robot6,
            joint_position_local=joint_position_robot6, 
        )

        # Gửi lệnh vận tốc cho robot 6
        self.robot6.set_joint_velocity(theta_dot_robot6, client_socket = client_socket_robot6)
        self.robot6.Update_visualization()
        # Cập nhật vị trí quả cầu mục tiêu của robot 6
        p.resetBasePositionAndOrientation(
            self.sphere_robot6,
            [self.Px_robot6, self.Py_robot6, self.Pz_robot6],
            [0, 0, 0, 1]
        )

        #### Robot5 ##########
        #### Robot5 ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot5 = self.robot5.take_joint_position(client_socket = client_socket_robot5)
        joint_position_robot5 = (joint_position_robot5 + np.pi) % (2 * np.pi) - np.pi

        ## Quỹ đạo target
        x_target_robot5 = self.R_target_robot5*np.sin(self.omega_target_robot5*self.time_step) + self.x_0_target_robot5  
        y_target_robot5 = self.R_target_robot5*np.cos(self.omega_target_robot5*self.time_step) + self.y_0_target_robot5
        v_target_robot5 = [self.R_target_robot5*self.omega_target_robot5*np.cos(self.omega_target_robot5*self.time_step), 
                    -self.R_target_robot5*self.omega_target_robot5*np.sin(self.omega_target_robot5*self.time_step), 0]
        z_target_robot5 = 0.3
        Rx_target_robot5 = 0.0
        Rz_target_robot5 = 0.0
        self.set_target_robot5(x_target_robot5, y_target_robot5, z_target_robot5, Rx_target_robot5, Rz_target_robot5)
        # 1. Tính hành động danh nghĩa (nominal) từ controller gốc
        theta_dot_robot5, pos_err_robot5, att_Rx_robot5, att_Rz_robot5 = self.robot5.get_theta_dot(
            x_target=self.Px_robot5,
            y_target=self.Py_robot5,
            z_target=self.Pz_robot5,
            v_target= v_target_robot5,  
            Rx_target=self.Rx_robot5,
            Rz_target=self.Rz_robot5,
            vmax=self.vmax_robot5,
            joint_position_local=joint_position_robot5
        )

        # Điều kiện dừng cho robot 5: Nếu đã đến đích, vận tốc danh nghĩa là 0
        # --- Update Plot every 10 steps to save performance ---
        if self.use_matplotlib and self.time_step_counter % self.plot_update_freq == 0:
            # Calculate distances for plotting
            d_inter_plot, _ = self._get_min_dist_between_robots(joint_position_robot5, joint_position_robot6)
            d_self_plot = self._get_min_self_dist_robot5(joint_position_robot5)
            if d_self_plot == float('inf'): d_self_plot = 1.0 # Clip if no collision detected

            current_time = time.time() - self.start_time
            self.x_data.append(current_time)
            self.y_dmin.append(d_inter_plot)
            self.y_dself.append(d_self_plot)
            
            # Keep window size
            while self.x_data and (self.x_data[-1] - self.x_data[0] > self.plot_window_time):
                self.x_data.popleft()
                self.y_dmin.popleft()
                self.y_dself.popleft()
            
            self.line_dmin.set_data(self.x_data, self.y_dmin)
            self.line_dself.set_data(self.x_data, self.y_dself)
            
            if self.x_data:
                self.line_safe.set_data([self.x_data[0], self.x_data[-1]], [self.d_safe, self.d_safe])
                self.ax.set_xlim(self.x_data[0], self.x_data[-1])
                self.ax.set_ylim(0, max(max(self.y_dmin), max(self.y_dself), self.d_safe) + 0.1)
            
            if self.draw_callback:
                self.draw_callback()
            else:
                plt.pause(0.00001)

       # 2. Áp dụng CBF nếu được bật
        if self.use_cbf:
            # --- Ràng buộc 1: Va chạm giữa 2 robot (inter-collision) ---
            jacobian_inter, d_min_inter = self._get_dist_jacobian_numerical(joint_position_robot5, joint_position_robot6) # Tính toán d_min_inter
            h_inter = d_min_inter - self.d_safe

            # --- Ràng buộc 2: Tự va chạm của Robot5 (self-collision) ---
            # Có thể dùng một ngưỡng an toàn khác cho tự va chạm nếu muốn, ví dụ: self.d_safe_self
            jacobian_self, d_min_self = self._get_self_dist_jacobian_numerical_robot5(joint_position_robot5) # Tính toán d_min_self
            h_self = d_min_self - self.d_safe 

            # --- Xây dựng và giải bài toán QP với nhiều ràng buộc ---
            # Định nghĩa biến tối ưu hóa CHO CẢ HAI ràng buộc
            u_var = cp.Variable(len(theta_dot_robot5))
            constraints = []

            # Chỉ thêm ràng buộc khi robot ở gần vùng nguy hiểm
            if h_inter < 0.1: # Vùng ảnh hưởng của CBF va chạm ngoài
                # SỬA LỖI: Sử dụng cùng một biến u_var cho tất cả các ràng buộc
                constraints.append(jacobian_inter @ u_var >= -self.cbf_gamma * h_inter)
            if h_self < 0.1: # Vùng ảnh hưởng của CBF tự va chạm
                constraints.append(jacobian_self @ u_var >= -self.cbf_gamma * h_self)

            # Giải QP để có hành động an toàn
            safe_theta_dot = self._solve_cbf_qp(theta_dot_robot5, constraints, u_var)

            # Lấy thông tin điểm gần nhất để vẽ đường debug cho va chạm ngoài
            _, closest_points_info = self._get_min_dist_between_robots(joint_position_robot5, joint_position_robot6)
            # self.draw_safety_line(closest_points_info, d_min_inter)
            print(f"\rCBF Active | d_inter: {d_min_inter:.3f}m | d_self: {d_min_self:.3f}m", end="")

        else:
            safe_theta_dot = theta_dot_robot5

        # 3. Gửi lệnh vận tốc (an toàn)
        safe_theta_dot = np.clip(safe_theta_dot, -self.robot5.joint_vel_limit, self.robot5.joint_vel_limit)
        self.robot5.set_joint_velocity(safe_theta_dot, client_socket=client_socket_robot5)

        ###########################################################################
        #################################################################################
        self.robot5.Update_visualization()
        #################################################################################
        #################################################################################
        # Cập nhật vị trí quả cầu mục tiêu
        # Cập nhật vị trí quả cầu mục tiêu của robot5
        p.resetBasePositionAndOrientation(
            self.sphere_robot5,
            [self.Px_robot5, self.Py_robot5, self.Pz_robot5],
            [0, 0, 0, 1]
        )
        #################################################################################
        #################################################################################

        # >>> Cập nhật vị trí 6 điểm quan sát trên link
        # self.robot6.update_link_markers()
        # self.robot5.update_link_markers()

        # --- Ghi dữ liệu góc khớp ra file ---
        # Định dạng theo yêu cầu: time_step,t1..t5,time_step,t6..t11,case_fanuc,case_sixdof
        log_data = []
        log_data.append(f"{self.time_step_counter * self.dt:.6f}") # time_step
        log_data.extend([f"{q:.6f}" for q in joint_position_robot5]) # t1-t5
        log_data.append(f"{self.time_step_counter * self.dt:.6f}") # time_step (lặp lại)
        log_data.extend([f"{q:.6f}" for q in joint_position_robot6]) # t6-t11
        log_data.append("1") # Case_fanuc
        log_data.append("1") # Case_sixdof

        self.log_file.write(",".join(log_data) + "\n")
        self.time_step_counter += 1

        
        # Nếu non-real-time thì phải tự step
        if self.flag_mode_simulation == 1:
            p.stepSimulation()
            # time.sleep(self.dt)  # nếu muốn chậm lại
            if self.use_gui:
                pass # time.sleep(self.dt)  # Đã bỏ sleep để tăng tốc độ tối đa

        return [pos_err_robot5, att_Rx_robot5, att_Rz_robot5], [pos_err_robot6, att_Rx_robot6, att_Rz_robot6]
    
    def run(self):
        """Vòng lặp vô hạn."""
        try:
            while True:
                self.step()
        finally:
            print(f"\nClosing log file: {self.log_filename}")
            self.log_file.close()

if __name__ == "__main__":
    env = Environment(use_gui=True)

    # Nhập mục tiêu
    Px = float(input("Nhap gia tri cua Px (m): "))
    Py = float(input("Nhap gia tri cua Py (m): "))
    Pz = float(input("Nhap gia tri cua Pz (m): "))
    env.set_target_robot6(Px, Py, Pz)
    env.set_target_robot5(Px, Py, Pz)

    # Hỏi người dùng có muốn sử dụng CBF không
    use_cbf_input = input("Ban co muon su dung CBF de tranh va cham khong? (y/n): ").lower()
    if use_cbf_input == 'y':
        env.set_cbf_status(True)
    # Chọn mode
    flag_mode = int(input("0 - real time | 1 - non-real time: "))
    env.set_mode(flag_mode)

    # Chạy mô phỏng
    env.run()
