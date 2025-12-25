import numpy as np
import pybullet as p
import pybullet_data
import cvxpy as cp
import time
from Robot5 import Robot_5_Dof
from Robot6 import Robot_6_Dof
from function_robot.Robot6_observer import Robot6_observer
from function_robot.Robot5_observer import Robot5_observer

class Environment:
    def __init__(self, use_gui=True):
        """Khởi tạo PyBullet, mặt phẳng, robot và quả cầu target."""
        """Đây là môi trường để sau này deploy chương trình"""
        self.use_gui = use_gui

        # Kết nối PyBullet
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Mặt phẳng
        self.plane_id = p.loadURDF("plane.urdf")

        # Robot 6 bậc
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        self.base5 = np.array([0.0, -0.75, 0.0])
        # Robot 5 bậc
        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=[0, -0.75, 0],
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
        # Quả cầu hiển thị mục tiêu
        # self.workspace = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=p.createVisualShape(
        #         p.GEOM_SPHERE,
        #         radius= self.workspace_data[3],
        #         rgbaColor=[0, 1, 0, 0.2]
        #     ),
        #     basePosition=[self.workspace_data[0], self.workspace_data[1], self.workspace_data[2]]
        # )

        # Tham số mô phỏng và điều khiển
        self.vmax_robot6 = 0.05 # (m/s)
        self.vmax_robot5 = 0.05  # (m/s)
        self.dt = 1.0 / 240.0

        # Target mặc định
        self.Px_robot6 = 0.0
        self.Py_robot6 = 0.0
        self.Pz_robot6 = 0.0

                # Target mặc định
        self.Px_robot5 = 0.0
        self.Py_robot5 = 0.0
        self.Pz_robot5 = 0.0
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


    # ------------------ Cài đặt target & mode ------------------ #

    def set_target_robot6(self, Px, Py, Pz):
        self.Px_robot6 = float(Px) # Bỏ chia 1000, giả định đầu vào là mét
        self.Py_robot6 = float(Py)
        self.Pz_robot6 = float(Pz)

    def set_target_robot5(self, Px, Py, Pz):
        self.Px_robot5 = float(Px) # Bỏ chia 1000, giả định đầu vào là mét
        self.Py_robot5 = float(Py)
        self.Pz_robot5 = float(Pz)

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
        pts5 = np.array([point for point, R in Robot5_observer_data]) + self.base5
        R_pts5 = np.array([R for point, R in Robot5_observer_data])

        Robot6_observer_data = Robot6_observer(q6_angles[0], q6_angles[1], q6_angles[2], q6_angles[3], q6_angles[4])
        pts6 = np.array([point for point, R in Robot6_observer_data])
        R_pts6 = np.array([R for point, R in Robot6_observer_data])

        min_dist_overall = float('inf')
        closest_points_info = {}

        # Duyệt qua tất cả các cặp observer sphere (một từ robot5, một từ robot6)
        for i, p5_center in enumerate(pts5):
            for j, p6_center in enumerate(pts6):
                diff = p5_center - p6_center
                center_dist = np.linalg.norm(diff)
                surface_dist = center_dist - (R_pts5[i] + R_pts6[j])

                if surface_dist < min_dist_overall:
                    min_dist_overall = surface_dist
                    # Để vẽ đường debug, ta cần điểm trên bề mặt của mỗi sphere
                    if center_dist > 1e-6:
                        direction_vec = diff / center_dist
                        p_r6_surface = p6_center + direction_vec * R_pts6[j]
                        p_r5_surface = p5_center - direction_vec * R_pts5[i]
                    else: # Tâm trùng nhau
                        p_r6_surface = p6_center
                        p_r5_surface = p5_center

                    closest_points_info = {
                        'p_robot': p_r5_surface,    # Điểm trên robot5 (agent)
                        'p_obstacle': p_r6_surface, # Điểm trên robot6 (obstacle)
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
        """Giải bài toán QP của CBF để tìm hành động an toàn."""
        act_dim = len(u_nom)
        u = cp.Variable(act_dim)
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

        h = d_min - self.d_safe
        constraints = [jacobian @ u >= -self.cbf_gamma * h]

        # Ràng buộc vận tốc khớp
        # constraints.append(u >= -2*self.robot5.joint_vel_limit)
        # constraints.append(u <= 2*self.robot5.joint_vel_limit)

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

    def draw_safety_line(self, closest_points_info, d_min):
        """Vẽ đường debug thể hiện khoảng cách an toàn."""
        if not closest_points_info:
            return

        p1 = closest_points_info['p_robot']
        p2 = closest_points_info['p_obstacle']

        # Xanh: an toàn, Đỏ: nguy hiểm
        color = [1, 0, 0] if d_min < self.d_safe else [0, 1, 0]
        width = 3

        # Xóa các đường cũ và vẽ đường mới
        for line_id in self.last_debug_lines:
            p.removeUserDebugItem(line_id)
        self.last_debug_lines.clear()

        line_id = p.addUserDebugLine(p1.tolist(), p2.tolist(), lineColorRGB=color, lineWidth=width)
        self.last_debug_lines.append(line_id)

    # ------------------ Một bước mô phỏng ------------------ #
    def step(self):
        # Xóa các đường debug cũ của CBF, các đường khác (nếu có) sẽ không bị ảnh hưởng
        for line_id in self.last_debug_lines:
            p.removeUserDebugItem(line_id)
        self.last_debug_lines.clear()

        #### Robot6 (coi như chướng ngại vật động) ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot6 = self.robot6.take_joint_position()
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi

        # Controller cho robot 6
        theta_dot_robot6, pos_err_robot6, att_Rx_robot6, att_Rz_robot6 = self.robot6.get_theta_dot(
            x_target=self.Px_robot6,
            y_target=self.Py_robot6,
            z_target=self.Pz_robot6,
            vmax=self.vmax_robot6,
            joint_position_local=joint_position_robot6
        )

        # Điều kiện dừng
        if pos_err_robot6 < 1e-3 and abs(att_Rx_robot6) < 1e-3 and abs(att_Rz_robot6) < 1e-3:
            # theta_dot_robot6 = np.zeros(6)
            new_goal = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.2, -0.4), np.random.uniform(0.25, 0.45)])
            env.set_target_robot6(*new_goal)
            env.goal_robot6 = new_goal

        # Gửi lệnh vận tốc cho robot 6
        self.robot6.set_joint_velocity(theta_dot_robot6)

        # Cập nhật vị trí quả cầu mục tiêu của robot 6
        p.resetBasePositionAndOrientation(
            self.sphere_robot6,
            [self.Px_robot6, self.Py_robot6, self.Pz_robot6],
            [0, 0, 0, 1]
        )

        #### Robot5 ##########
        #### Robot5 ##########
        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot5 = self.robot5.take_joint_position()
        joint_position_robot5 = (joint_position_robot5 + np.pi) % (2 * np.pi) - np.pi

        # Controller
        # 1. Tính hành động danh nghĩa (nominal) từ controller gốc
        theta_dot_robot5, pos_err_robot5, att_Rx_robot5, att_Rz_robot5 = self.robot5.get_theta_dot(
            x_target=self.Px_robot5,
            y_target=self.Py_robot5,
            z_target=self.Pz_robot5,
            vmax=self.vmax_robot5,
            joint_position_local=joint_position_robot5
        )

        # 2. Áp dụng CBF nếu được bật
        if self.use_cbf:
            # Tính Jacobian và khoảng cách hiện tại
            jacobian, d_min = self._get_dist_jacobian_numerical(joint_position_robot5, joint_position_robot6)

            # Giải QP để có hành động an toàn
            safe_theta_dot = self.solve_cbf_qp(theta_dot_robot5, d_min, jacobian)

            # Lấy thông tin điểm gần nhất để vẽ
            _, closest_points_info = self._get_min_dist_between_robots(joint_position_robot5, joint_position_robot6)
            self.draw_safety_line(closest_points_info, d_min)
            print(f"\rCBF Active | d_min: {d_min:.3f}m | h: {d_min - self.d_safe:.3f}m", end="")

        else:
            safe_theta_dot = theta_dot_robot5

        # Điều kiện dừng cho robot 5
        if pos_err_robot5 < 1e-4 and abs(att_Rx_robot5) < 1e-3 and abs(att_Rz_robot5) < 1e-3:
            safe_theta_dot = np.zeros(5)
            

        # 3. Gửi lệnh vận tốc (an toàn)
        self.robot5.set_joint_velocity(safe_theta_dot)

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
                time.sleep(self.dt)  # nếu muốn chậm lại

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
