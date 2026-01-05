from Environment__HaiRobot import Environment
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
# from function.Robot_motion_function import Move_Robot, Connect_tcp_ip
import socket
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import random
import time 
# Biến cờ để quản lý thread
t_sample = 0.002  # Giảm thời gian mẫu để tăng tần số tối đa (0.002s = 500Hz)

class RobotThreadManager:
    def __init__(self):
        self.current_thread = None
        self.stop_event = threading.Event()

    def stop(self):
        """Gửi tín hiệu dừng và chờ luồng kết thúc an toàn."""
        self.stop_event.set()
        if self.current_thread and self.current_thread.is_alive():
            # Chờ tối đa 0.5s để tránh treo GUI nếu luồng bị kẹt
            self.current_thread.join(timeout=0.5)
        self.current_thread = None

    def start(self, target_func, args=()):
        """Dừng luồng cũ và khởi động luồng mới."""
        self.stop() # Đảm bảo luồng trước đã dừng
        self.stop_event.clear() # Reset trạng thái dừng
        self.current_thread = threading.Thread(target=target_func, args=args, daemon=True)
        self.current_thread.start()

# Khởi tạo quản lý luồng
manager = RobotThreadManager()

def Close_app():
    manager.stop()
    root.destroy()
    
def Move_Robot():

    global Px_data, Py_data, Pz_data, time_data
    Px_data, Py_data, Pz_data, time_data = [], [], [], []

    # while not Finish and not stop_flag and iteration < max_iterations:
    x_target_robot5 = float(entry_vars_position_robot5[0].get())
    y_target_robot5 = float(entry_vars_position_robot5[1].get())
    z_target_robot5 = float(entry_vars_position_robot5[2].get())
    Rx_robot5 = float(entry_vars_position_robot5[3].get())*np.pi/180
    Rz_robot5 = float(entry_vars_position_robot5[4].get())*np.pi/180

    x_target_robot6  = float(entry_vars_position_robot6 [0].get())
    y_target_robot6  = float(entry_vars_position_robot6 [1].get())
    z_target_robot6  = float(entry_vars_position_robot6 [2].get())
    Rx_robot6  = float(entry_vars_position_robot6 [3].get())*np.pi/180
    Rz_robot6  = float(entry_vars_position_robot6 [4].get())*np.pi/180

    env.set_target_robot5(x_target_robot5, y_target_robot5, z_target_robot5, Rx_robot5, Rz_robot5)
    env.set_target_robot6(x_target_robot6, y_target_robot6, z_target_robot6, Rx_robot6, Rz_robot6)
    
    last_time = time.time()
    frame_count = 0
    
    try:
        while not manager.stop_event.is_set():
            env.step(client_socket_robot5=client_socket_robot5, client_socket_robot6=client_socket_robot6)
            manager.stop_event.wait(timeout=t_sample)
            
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 0.5: # Cập nhật mỗi 0.5 giây
                freq = frame_count / (current_time - last_time)
                root.after(0, lambda f=freq: frequency_label_var.set(f"Freq: {f:.1f} Hz"))
                last_time = current_time
                frame_count = 0
    finally:
        # Dừng robot an toàn khi kết thúc thread và đọc phản hồi để đảm bảo tuần tự
        if client_socket_robot5: 
            env.robot5.set_joint_velocity([0]*5, client_socket_robot5)
            try: client_socket_robot5.recv(1024)
            except: pass
        if client_socket_robot6: 
            env.robot6.set_joint_velocity([0]*6, client_socket_robot6)
            try: client_socket_robot6.recv(1024)
            except: pass
        print("Dừng bởi người dùng.")

def start_move():
    manager.start(Move_Robot)

def Home_Robot():
    # while not Finish and not stop_flag and iteration < max_iterations:
    x_target_robot6 = 0.4
    y_target_robot6 = -0.75
    z_target_robot6 = 0.4
    Rx_target_robot5 = 90*np.pi/180
    Rz_target_robot5 = 0

    x_target_robot5 = 0.35
    y_target_robot5 = 0
    z_target_robot5 = 0.5
    Rx_target_robot6 = 90*np.pi/180
    Rz_target_robot6 = 0

    env.set_target_robot6(x_target_robot6, y_target_robot6, z_target_robot6, Rx_target_robot6, Rz_target_robot6)
    env.set_target_robot5(x_target_robot5, y_target_robot5, z_target_robot5, Rx_target_robot5, Rz_target_robot5)
    
    last_time = time.time()
    frame_count = 0
    
    try:
        while not manager.stop_event.is_set():
            env.step(client_socket_robot6, client_socket_robot5)
            manager.stop_event.wait(timeout=t_sample)
            
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 0.5: # Cập nhật mỗi 0.5 giây
                freq = frame_count / (current_time - last_time)
                root.after(0, lambda f=freq: frequency_label_var.set(f"Freq: {f:.1f} Hz"))
                last_time = current_time
                frame_count = 0
    finally:
        # Dừng robot an toàn khi kết thúc thread và đọc phản hồi để đảm bảo tuần tự
        if client_socket_robot5: 
            env.robot5.set_joint_velocity([0]*5, client_socket_robot5)
            try: client_socket_robot5.recv(1024)
            except: pass
        if client_socket_robot6: 
            env.robot6.set_joint_velocity([0]*6, client_socket_robot6)
            try: client_socket_robot6.recv(1024)
            except: pass
        print("Dừng bởi người dùng.")

# Hàm khởi chạy thread cho Home
def start_home():
    manager.start(Home_Robot)

def Stop_Robot():
    manager.stop()
    print("Tất cả hoạt động đã dừng.")

def Connect_tcp_ip():
    global client_socket_robot5, client_socket_robot6
    print('Connecting sixdof')
    server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_1_address = ('0.0.0.0', 12346)  # sixdof
    server_socket_1.bind(server_1_address)
    server_socket_1.listen(1)
    server_socket_1.settimeout(0.2)  # Giảm thời gian chờ để phản hồi dừng nhanh hơn
    # print(f"Đang lắng nghe tại {server_1_address[0]}:{server_1_address[1]}")
    while not manager.stop_event.is_set():  # Kiểm tra cờ
        try:
            client_socket_robot6, _ = server_socket_1.accept()
            # print('Kết nối sixdof thành công')
            break  # Thoát khi có kết nối
        except socket.timeout:
            # Không có kết nối, kiểm tra lại cờ
            continue
            
    if manager.stop_event.is_set():
        server_socket_1.close()
        return

    server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_2_address = ('0.0.0.0', 12345)  # robot5
    server_socket_2.bind(server_2_address)
    server_socket_2.listen(1)
    server_socket_2.settimeout(0.2)  # Giảm thời gian chờ
    # print(f"Đang lắng nghe tại {server_2_address[0]}:{server_2_address[1]}")
    while not manager.stop_event.is_set():  # Kiểm tra cờ
        try:
            client_socket_robot5, _ =  server_socket_2.accept()
            break  # Thoát khi có kết nối
        except socket.timeout:
            # Không có kết nối, kiểm tra lại cờ
            continue
            
    if manager.stop_event.is_set():
        server_socket_2.close()
        return
    
    print('Kết nối robot5 thành công')

def start_connect_tcp():
    manager.start(Connect_tcp_ip)

def start_tuong_tac():
    manager.start(Tuong_tac)

def start_circle_path():
    manager.start(Circle_path) 

def Tuong_tac():
    # while not Finish and not stop_flag and iteration < max_iterations:
    x_target_robot6_list = [0.3, 0.2, 0.1, 0]
    y_target_robot6_list = [-0.5, -0.5, -0.5, -0.5]
    z_target_robot6_list = [0.3, 0.3, 0.3, 0.3, 0.3]
    Rx_target_robot6 = 0
    Rz_target_robot6 = 0

    x_target_robot5 = 0
    y_target_robot5 = -0.5
    z_target_robot5 = 0.3
    Rx_target_robot5 = 0
    Rz_target_robot5 = 0

    robot6_task = 0
    env.set_target_robot6(x_target_robot6_list[robot6_task], y_target_robot6_list[robot6_task], 
                          z_target_robot6_list[robot6_task], Rx_target_robot6, Rz_target_robot6)
    env.set_target_robot5(x_target_robot5, y_target_robot5, z_target_robot5, Rx_target_robot5, Rz_target_robot5)
    
    last_time = time.time()
    frame_count = 0
    
    try:
        while not manager.stop_event.is_set():
            robot5_info, robot6_info = env.step(client_socket_robot6, client_socket_robot5)
            manager.stop_event.wait(timeout=t_sample)
            
            frame_count += 1
            current_time = time.time()

            pos_err_robot6, att_Rx_robot6, att_Rz_robot6 = robot6_info
            
            if pos_err_robot6 < 1e-3 and abs(att_Rx_robot6) < 0.1 and abs(att_Rz_robot6) < 0.1:
                robot6_task += 1
                if robot6_task >= len(x_target_robot6_list):
                    robot6_task = 0
                env.set_target_robot6(x_target_robot6_list[robot6_task], y_target_robot6_list[robot6_task], 
                          z_target_robot6_list[robot6_task], Rx_target_robot6, Rz_target_robot6)
            if current_time - last_time >= 0.5: # Cập nhật mỗi 0.5 giây
                freq = frame_count / (current_time - last_time)
                root.after(0, lambda f=freq: frequency_label_var.set(f"Freq: {f:.1f} Hz"))
                last_time = current_time
                frame_count = 0
    finally:
        # Dừng robot an toàn khi kết thúc thread và đọc phản hồi để đảm bảo tuần tự
        if client_socket_robot5: 
            env.robot5.set_joint_velocity([0]*5, client_socket_robot5)
            try: client_socket_robot5.recv(1024)
            except: pass
        if client_socket_robot6: 
            env.robot6.set_joint_velocity([0]*6, client_socket_robot6)
            try: client_socket_robot6.recv(1024)
            except: pass
        print("Dừng bởi người dùng.")

def Circle_path():
    frame_count = 0
    last_time = time.time()
    try:
        while not manager.stop_event.is_set():
            robot5_info, robot6_info = env.step_circle(client_socket_robot6, client_socket_robot5)
            manager.stop_event.wait(timeout=t_sample)
            
            frame_count += 1
            current_time = time.time()

            if current_time - last_time >= 0.5: # Cập nhật mỗi 0.5 giây
                freq = frame_count / (current_time - last_time)
                root.after(0, lambda f=freq: frequency_label_var.set(f"Freq: {f:.1f} Hz"))
                last_time = current_time
                frame_count = 0
    finally:
        # Dừng robot an toàn khi kết thúc thread và đọc phản hồi để đảm bảo tuần tự
        if client_socket_robot5: 
            env.robot5.set_joint_velocity([0]*5, client_socket_robot5)
            try: client_socket_robot5.recv(1024)
            except: pass
        if client_socket_robot6: 
            env.robot6.set_joint_velocity([0]*6, client_socket_robot6)
            try: client_socket_robot6.recv(1024)
            except: pass
        print("Dừng bởi người dùng.")

def move_home_stop(index):
    if index == 0:
        print('Move')
        start_move()
    elif index == 1:
        print('Home')
        start_home()

    elif index == 2:
        print('Stop')
        Stop_Robot()

    elif index == 3: 
        print('TPC connecting')
        start_connect_tcp()
    
    elif index == 4: 
        print('Tuong tac')
        start_tuong_tac()

    elif index == 5: 
        print('Circle path')
        start_circle_path()

    elif index == 6:
        Close_app()
        print('Close')

root = tk.Tk()
root.title("My Robot")
root.geometry('900x900')

# Tăng kích thước hình ảnh
plus_image = Image.open('./Icon_image/plus.jpg')  # Đọc ảnh
plus_image = plus_image.resize((20, 20))  # Thay đổi kích thước
plus_icon = ImageTk.PhotoImage(plus_image, master=root)  # Chuyển đổi thành PhotoImage

minus_image = Image.open('./Icon_image/minus.png')  # Đọc ảnh
minus_image = minus_image.resize((20, 20))  # Thay đổi kích thước
minus_icon = ImageTk.PhotoImage(minus_image, master=root)  # Chuyển đổi thành PhotoImage

# entry_var = tk.StringVar(value=0)
entry_vars_position_robot5 = [tk.StringVar(value=350), tk.StringVar(value =0), tk.StringVar(value =500), 
                       tk.StringVar(value =0), tk.StringVar(value =0), tk.StringVar(value =0)]
entry_vars_position_robot6 = [tk.StringVar(value=350), tk.StringVar(value =0), tk.StringVar(value =500), 
                       tk.StringVar(value =0), tk.StringVar(value =0), tk.StringVar(value =0)]

labels_robot5 = ['Px_robot5', 'Py_robot5', 'Pz_robot5', 'Rx_robot5', 'Ry_robot5', 'Rz_robot5']

for i in range(6):
    label = tk.Label(root, text= labels_robot5[i], 
                     font=('Arial', 10, 'bold'),
                     fg='#44AFB5',
                     bg='black',
                     bd=10,
                     padx=5,
                     pady=5)
    label.place(x=400, y=i*50)

    entry= tk.Entry(root, 
                    font=('Arial', 10),
                    textvariable=entry_vars_position_robot5[i]
                    )
    entry.place(x=550, y= 50*i + 20)

labels_robot6 = ['Px_robot6', 'Py_robot6', 'Pz_robot6', 'Rx_robot6', 'Ry_robot6', 'Rz_robot6']

for i in range(6):
    label = tk.Label(root, text= labels_robot6[i], 
                     font=('Arial', 10, 'bold'),
                     fg='#44AFB5',
                     bg='black',
                     bd=10,
                     padx=5,
                     pady=5)
    label.place(x=50, y=i*50)

    entry= tk.Entry(root, 
                    font=('Arial', 10),
                    textvariable=entry_vars_position_robot6[i]
                    )
    entry.place(x=200, y= 50*i + 20)

### Button move, home, stop 
labels_move_home_stop = ['Move', 'Home', 'Stop', 'TCP', 'TT', 'Circle', 'Close']
for i in range(len(labels_move_home_stop)):
    button_move_home_stop = tk.Button(root, 
                        text=labels_move_home_stop[i],
                        command=lambda idx=i: move_home_stop(idx))
    button_move_home_stop.place(x=50*i+50, y=350)

# Label hiển thị tần số
frequency_label_var = tk.StringVar(value="Freq: 0.0 Hz")
freq_label = tk.Label(root, textvariable=frequency_label_var, font=('Arial', 12, 'bold'), fg='red')
freq_label.place(x=450, y=355)

# Tạo khung Matplotlib
frame_graph = tk.Frame(root, width=800, height=400, bg="blue")
frame_graph.place(x=0, y=400)

fig = Figure(figsize=(8, 4), dpi=100)
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def draw_callback():
    root.after(0, canvas.draw_idle)

env = Environment(use_matplotlib=True, ax=ax, draw_callback=draw_callback, use_gui=True)
env.set_cbf_status(True)
env.set_mode(0)


# Khai báo biến toàn cục
client_socket_robot6 = None
client_socket_robot5 = None
root.mainloop()
