import numpy as np

def Robot6_observer(t1, t2, t3, t4, t5):
    a_2 = 0.26
    a_3 = 0.0305
    d_1 = 0.29
    d_4 = 0.27
    d_tool = 0.09
    Base = np.array([0, 0, 0])

    # joint_1
    joint_1 = np.array([0, 0, d_1])

    # Joint_2
    x_2 = a_2 * np.cos(t1) * np.cos(t2)
    y_2 = a_2 * np.sin(t1) * np.cos(t2)
    z_2 = d_1 + a_2 * np.sin(t2)
    joint_2 = np.array([x_2, y_2, z_2])

    # joint_3
    x_3 = a_2 * np.cos(t1) * np.cos(t2) + a_3 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.cos(t1) * np.sin(
        t2) * np.sin(t3)
    y_3 = a_2 * np.cos(t2) * np.sin(t1) + a_3 * np.cos(t2) * np.cos(t3) * np.sin(t1) - a_3 * np.sin(t1) * np.sin(
        t2) * np.sin(t3)
    z_3 = d_1 + a_2 * np.sin(t2) + a_3 * np.cos(t2) * np.sin(t3) + a_3 * np.cos(t3) * np.sin(t2)
    joint_3 = np.array([x_3, y_3, z_3])

    # joint_4
    x_4 = d_4 * (np.cos(t1) * np.cos(t2) * np.sin(t3) + np.cos(t1) * np.cos(t3) * np.sin(t2)) + a_2 * np.cos(
        t1) * np.cos(t2) + a_3 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.cos(t1) * np.sin(t2) * np.sin(t3)
    y_4 = d_4 * (np.cos(t2) * np.sin(t1) * np.sin(t3) + np.cos(t3) * np.sin(t1) * np.sin(t2)) + a_2 * np.cos(
        t2) * np.sin(t1) + a_3 * np.cos(t2) * np.cos(t3) * np.sin(t1) - a_3 * np.sin(t1) * np.sin(t2) * np.sin(t3)
    z_4 = d_1 - d_4 * (np.cos(t2) * np.cos(t3) - np.sin(t2) * np.sin(t3)) + a_2 * np.sin(t2) + a_3 * np.cos(
        t2) * np.sin(t3) + a_3 * np.cos(t3) * np.sin(t2)
    joint_4 = np.array([x_4, y_4, z_4])

    # Tính toán joint_tool
    x_tool = d_4 * np.sin((t2 + t3)) * np.cos(t1) + a_2 * np.cos(t1) * np.cos(t2) + d_tool * np.sin((t2 + t3)) * np.cos(
        t1) * np.cos(t5) + a_3 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.cos(t1) * np.sin(t2) * np.sin(
        t3) + d_tool * np.sin(t1) * np.sin(t4) * np.sin(t5) + d_tool * np.cos(t1) * np.cos(t2) * np.cos(t3) * np.cos(
        t4) * np.sin(t5) - d_tool * np.cos(t1) * np.cos(t4) * np.sin(t2) * np.sin(t3) * np.sin(t5)
    y_tool = d_4 * np.sin((t2 + t3)) * np.sin(t1) + a_2 * np.cos(t2) * np.sin(t1) + d_tool * np.sin((t2 + t3)) * np.cos(
        t5) * np.sin(t1) + a_3 * np.cos(t2) * np.cos(t3) * np.sin(t1) - d_tool * np.cos(t1) * np.sin(t4) * np.sin(
        t5) - a_3 * np.sin(t1) * np.sin(t2) * np.sin(t3) + d_tool * np.cos(t2) * np.cos(t3) * np.cos(t4) * np.sin(
        t1) * np.sin(t5) - d_tool * np.cos(t4) * np.sin(t1) * np.sin(t2) * np.sin(t3) * np.sin(t5)
    z_tool = d_1 - d_4 * np.cos((t2 + t3)) + a_3 * np.sin((t2 + t3)) + a_2 * np.sin(t2) + (
                d_tool * np.sin((t2 + t3)) * np.sin((t4 + t5))) / 2 - d_tool * np.cos((t2 + t3)) * np.cos(t5) - (
                         d_tool * np.sin((t2 + t3)) * np.sin((t4 - t5))) / 2
    joint_tool = np.array([x_tool, y_tool, z_tool])

    ###########################################3
    point_1 = np.array([0, 0, d_1/2])

    ## Giữa joint_1 và joint_2
    point_2 = joint_1 
    point_3 = 1/4 * joint_1 + 3/4 * joint_2
    point_4 = 1/2 * joint_1 + 1/2 * joint_2
    point_5 = 3/4 * joint_1 + 1/4 * joint_2

    ## Giữa joint_3 và joint_4
    point_6 = joint_2
    point_7 = 1/4 * joint_3 + 3/4 * joint_4
    point_8 = 1/2 * joint_3 + 1/2 * joint_4
    point_9 = 3/4 * joint_3 + 1/4 * joint_4
    ## joint tool   
    point_10= joint_4
    point_tool = joint_tool

    R_point_1 = 0.1
    R_point_2 = 0.1 
    R_point_3 = 0.1
    R_point_4 = 0.1
    R_point_5 = 0.1
    R_point_6 = 0.1
    R_point_7 = 0.1
    R_point_8 = 0.1
    R_point_9 = 0.1
    R_point_10= 0.1
    R_point_tool = 0.1  

    Arm_1 = [
        (point_1, R_point_1),
        (point_2, R_point_2),
        (point_3, R_point_3),
        (point_4, R_point_4),
        (point_5, R_point_5),
        (point_6, R_point_6),
        (point_7, R_point_7),
        (point_8, R_point_8),
        (point_9, R_point_9), 
        (point_10, R_point_10),
        (point_tool, R_point_tool)  
        ]

    return Arm_1