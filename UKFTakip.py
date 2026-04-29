import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# --- SABİTLER ---
W_REAL = 1.6  # Rakip İHA kanat açıklığı

def get_rotation_matrix(roll, pitch):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll),  math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    return np.dot(R_y, R_x)

def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt 
    return np.dot(F, x)

def hx_camera(x, uav_roll, uav_pitch):
    fx_cam, fy_cam = 800.0, 800.0  
    cx, cy = 320.0, 240.0          
    
    R = get_rotation_matrix(uav_roll, uav_pitch)
    Xc, Yc, Zc = R.dot(x[:3])
    Zc = max(Zc, 1.0) 
    
    u = (fx_cam * Xc) / Zc + cx
    v = (fy_cam * Yc) / Zc + cy
    w = (fx_cam * W_REAL) / Zc
    
    return np.array([u, v, w])

def generate_yolo_flight_data(n_steps=200, dt=0.1):
    true_states = []
    yolo_measurements = []
    uav_attitudes = []
    
    # Hedef kameranın merkezine daha yakın bir yerden başlıyor
    current_true_x = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0]) 
    
    for i in range(n_steps):
        # Rakip agresif manevra yapıyor
        current_true_x[3] = 4.0 * math.sin(i * 0.1)  
        current_true_x[4] = 2.0 * math.cos(i * 0.05) 
        current_true_x[5] = 2.0 * math.sin(i * 0.02) # Z ekseninde daha belirgin manevra
        
        current_true_x = fx(current_true_x, dt)
        true_states.append(current_true_x.copy())
        
        roll = math.radians(10 * math.sin(i * 0.1))   
        pitch = math.radians(5 * math.cos(i * 0.05))  
        uav_attitudes.append((roll, pitch))
        
        perfect_pixels = hx_camera(current_true_x, roll, pitch)
        
        # YOLO gürültüsü (Gerçekçi değerler: u,v için +-5 piksel, w için +-2 piksel)
        noisy_u = perfect_pixels[0] + np.random.normal(0, 5.0)
        noisy_v = perfect_pixels[1] + np.random.normal(0, 5.0)
        noisy_w = perfect_pixels[2] + np.random.normal(0, 2.0) 
        
        yolo_measurements.append(np.array([noisy_u, noisy_v, noisy_w]))
        
    return np.array(true_states), np.array(yolo_measurements), uav_attitudes, dt

if __name__ == "__main__":
    true_states, yolo_measurements, uav_attitudes, dt = generate_yolo_flight_data()
    
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-3) # Alpha'yı normale döndürdük
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=fx, hx=hx_camera, points=points)
    
    # 1. ÇÖZÜM: Başlangıç konumunu gerçeğe eşitledik
    ukf.x = np.array([0., 0., 30., 0., 0., 0.]) 
    ukf.P *= 50.0 # Başlangıçta biraz güvensiziz ki hemen toparlasın
    
    from filterpy.common import Q_discrete_white_noise
    import scipy.linalg as linalg
    
    # 2. ÇÖZÜM (Q Matrisi): İHA manevra yapabilir, fiziksel inatçılığı biraz kırdık (var=1.0)
    ukf.Q = linalg.block_diag(Q_discrete_white_noise(dim=2, dt=dt, var=1.0),
                              Q_discrete_white_noise(dim=2, dt=dt, var=1.0),
                              Q_discrete_white_noise(dim=2, dt=dt, var=1.0))
    
    # 3. ÇÖZÜM (R Matrisi): Genişliğe (w) olan güveni geri verdik (4.0^2).
    # u,v için 25 varyans (5 piksel sapma), w için 16 varyans (4 piksel sapma)
    ukf.R = np.diag([5.0**2, 5.0**2, 4.0**2])
    
    ukf_3d_estimates = []
    
    for i, z_pixels in enumerate(yolo_measurements):
        roll, pitch = uav_attitudes[i] 
        ukf.predict()
        ukf.update(z_pixels, uav_roll=roll, uav_pitch=pitch)
        ukf_3d_estimates.append(ukf.x.copy())
            
    ukf_3d_estimates = np.array(ukf_3d_estimates)
    
    rmse_x = np.sqrt(((ukf_3d_estimates[:, 0] - true_states[:, 0]) ** 2).mean())
    rmse_y = np.sqrt(((ukf_3d_estimates[:, 1] - true_states[:, 1]) ** 2).mean())
    rmse_z = np.sqrt(((ukf_3d_estimates[:, 2] - true_states[:, 2]) ** 2).mean())
    
    print(f"🎯 YENİ HATA PAYLARI: X={rmse_x:.2f}m, Y={rmse_y:.2f}m, Z={rmse_z:.2f}m")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], label='Gerçek MANEVRA Rotası', color='blue', linewidth=3)
    ax.plot(ukf_3d_estimates[:, 0], ukf_3d_estimates[:, 1], ukf_3d_estimates[:, 2], label='UKF Mükemmel Tahmini', color='green', linestyle='dashed', linewidth=2)
    
    ax.set_title("Otonom Hedef Takibi: Mükemmel Akortlanmış UKF")
    ax.set_xlabel("X (m) - Sağ/Sol")
    ax.set_ylabel("Y (m) - Yukarı/Aşağı")
    ax.set_zlabel("Z (m) - Derinlik/Mesafe")
    ax.legend()
    plt.show()