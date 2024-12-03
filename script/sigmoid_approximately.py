import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Định nghĩa hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Định nghĩa hàm mục tiêu: sai lệch giữa x * sigmoid(x) và 0.5 * x * e^(ax + b)
def objective(params, x):
    a, b, c, d, e, f = params
    left = x * sigmoid(x)  # x * sigmoid(x)
    right = 0.5 * x * (e * np.exp(a * x + b) + f * np.exp(c * x + d))  # 0.5 * x * e^(ax + b)
    return np.sum((left - right) ** 2)  # Tổng bình phương sai lệch

# Hàm tìm a và b
def find_parameters(x):
    # Khởi tạo giá trị ban đầu cho a và b
    initial_params = [0.1, 0.1, 0.1, 0.1, 0.8, 0.2]
    
    # Tối ưu hóa
    result = minimize(objective, initial_params, args=(x,), method='Powell')
    
    if result.success:
        a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = result.x
        return a_opt, b_opt, c_opt, d_opt, e_opt, f_opt
    else:
        raise ValueError("Optimization failed.")

# Tạo dãy giá trị x để tìm a và b
x_values = np.linspace(-5, 4, 100)

# Tìm các tham số a và b
a, b, c, d, e, f = find_parameters(x_values)

# print(f"Optimized a: {a}")
# print(f"Optimized b: {b}")
# print(f"Optimized b: {c}")
# print(f"Optimized b: {d}")
# print(f"Optimized b: {e}")

# Tính giá trị của hai hàm sau khi tìm được a và b
x_values = np.linspace(-10, 10, 100)
y1 = x_values * sigmoid(x_values)  # x * sigmoid(x)
y2 = 0.5 * x_values * (e * np.exp(a * x_values + b) - f * np.exp(c * x_values + d))

# Vẽ đồ thị của hai hàm
plt.figure(figsize=(8, 6))
plt.plot(x_values, y1, label='SiLU', color='blue')
plt.plot(x_values, y2, label='Approximate SiLU', color='red', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()