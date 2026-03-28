import numpy as np
ON = 1
OFF = 0
RU_array = [{"id":0, "sta":OFF, "pos":(0,0), "Km":[], "Pm":None},
            {'id': 1, 'sta': 1, 'pos': (1300, 600), 'Km': [], 'Pm': None},
            {'id': 2, 'sta': 1, 'pos': (1200.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 3, 'sta': 1, 'pos': (1000.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 4, 'sta': 1, 'pos': (900, 600), 'Km': [], 'Pm': None},
            {'id': 5, 'sta': 1, 'pos': (1000.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 6, 'sta': 1, 'pos': (1200.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 7, 'sta': 1, 'pos': (1100, 946), 'Km': [], 'Pm': None},
            {'id': 8, 'sta': 1, 'pos': (1000.0, 1119.0), 'Km': [], 'Pm': None},
            {'id': 9, 'sta': 1, 'pos': (800.0, 1119.0), 'Km': [], 'Pm': None},
            {'id': 10, 'sta': 1, 'pos': (700, 946), 'Km': [], 'Pm': None},
            {'id': 11, 'sta': 1, 'pos': (800.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 12, 'sta': 1, 'pos': (600.0, 1119.0), 'Km': [], 'Pm': None},
            {'id': 13, 'sta': 1, 'pos': (400.0, 1119.0), 'Km': [], 'Pm': None},
            {'id': 14, 'sta': 1, 'pos': (300, 946), 'Km': [], 'Pm': None},
            {'id': 15, 'sta': 1, 'pos': (400.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 16, 'sta': 1, 'pos': (600.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 17, 'sta': 1, 'pos': (500, 600), 'Km': [], 'Pm': None},
            {'id': 18, 'sta': 1, 'pos': (200.0, 773.0), 'Km': [], 'Pm': None},
            {'id': 19, 'sta': 1, 'pos': (100, 600), 'Km': [], 'Pm': None},
            {'id': 20, 'sta': 1, 'pos': (200.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 21, 'sta': 1, 'pos': (400.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 22, 'sta': 1, 'pos': (700, 254), 'Km': [], 'Pm': None},
            {'id': 23, 'sta': 1, 'pos': (600.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 24, 'sta': 1, 'pos': (300, 254), 'Km': [], 'Pm': None},
            {'id': 25, 'sta': 1, 'pos': (400.0, 81.0), 'Km': [], 'Pm': None},
            {'id': 26, 'sta': 1, 'pos': (600.0, 81.0), 'Km': [], 'Pm': None},
            {'id': 27, 'sta': 1, 'pos': (1100, 254), 'Km': [], 'Pm': None},
            {'id': 28, 'sta': 1, 'pos': (800.0, 427.0), 'Km': [], 'Pm': None},
            {'id': 29, 'sta': 1, 'pos': (800.0, 81.0), 'Km': [], 'Pm': None},
            {'id': 30, 'sta': 1, 'pos': (1000.0, 81.0), 'Km': [], 'Pm': None},
]
PN_array = [{"id":0, "sta":ON, "pos":(0,0), "Jm":[], "Kj":[], "C_ai":20, "C_bbu":20, "C_max":60, "Pj":0},
    {"id": 1, "sta": ON, "pos": (1100, 600), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
    {"id": 2, "sta": ON, "pos": (900, 946), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
    {"id": 3, "sta": ON, "pos": (500, 946), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
    {"id": 4, "sta": ON, "pos": (300, 600), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
    {"id": 5, "sta": ON, "pos": (500, 254), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
    {"id": 6, "sta": ON, "pos": (900, 254), "Jm": [], "Kj": [], "C_ai": 10, "C_bbu": 10, "C_max": 20, "Pj": None},
]

# 全局变量，将在外部被更新
User_array = []


def generate_user_data(num_users):
    """
    生成指定数量的随机用户位置，并初始化 User_array
    """
    global User_array

    # 清空旧数据，保留 id=0 的占位符（如果后续逻辑依赖 id=0）
    User_array = [{'id': 0, 'pos': (0, 0), 'dis': None, "Tr": 0, "Tp": 0, "Tw": 0, "Bk": None, "Rk": None, "Pn": -1}]

    # 生成随机位置
    # 假设 Random_pos 函数已经在 env_ee.py 中定义，或者在这里定义
    # 这里直接复用你提供的逻辑
    x_min, x_max, y_min, y_max = 0, 1400, 0, 1200
    x_cords = np.random.randint(low=x_min, high=x_max + 1, size=num_users).tolist()
    y_cords = np.random.randint(low=y_min, high=y_max + 1, size=num_users).tolist()
    user_pos = list(zip(x_cords, y_cords))

    for point_id, pos in enumerate(user_pos, start=1):
        User_array.append({
            "id": point_id,
            "pos": pos,
            "dis": None,
            "Tr": 0,
            "Tp": 0,
            "Tw": 0,
            "Bk": 0,
            "Rk": 0,
            "Pn": -1
        })
    # 将 User_array 写入 data.py 文件
    with open("data.py", "w", encoding="utf-8") as f:
        # 写入注释说明
        f.write("# 自动生成的用户数据\n")
        f.write("# 生成时间：自动生成\n")
        f.write("# 用户数量：{}（含id=0占位符）\n\n".format(len(User_array)))
        # 写入 User_array 变量（使用 repr 保证格式合法）
        f.write("User_array = {}\n".format(repr(User_array)))
    print(f"已生成 {num_users} 个用户数据。")
    for u in User_array:
        print(u,",")
    return User_array


# 初始化默认数据（例如 50 人），防止直接导入报错
if __name__ != "__main__":
    generate_user_data(100)