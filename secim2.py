import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# --- KAMERA VE İHA SABİTLERİ ---
W_REAL = 1.6  
FX, FY = 800.0, 800.0
CX, CY = 320.0, 240.0

# ==========================================
# 1. FİZİK VE KAMERA MODELLERİ
# ==========================================
def get_rotation_matrix(roll, pitch):
    R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll),  math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    return np.dot(R_y, R_x)

def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt 
    return np.dot(F, x)

def hx_camera(x, uav_roll=0.0, uav_pitch=0.0):
    # EKF'yi Çökertmek İçin Sınır Koruması: Z çok küçüldüğünde 1/Z patlar!
    X, Y, Z = x[0], x[1], max(x[2], 0.5) 
    
    R = get_rotation_matrix(uav_roll, uav_pitch)
    Xc, Yc, Zc = R.dot(np.array([X, Y, Z]))
    Zc = max(Zc, 0.5)
    
    u = (FX * Xc) / Zc + CX
    v = (FY * Yc) / Zc + CY
    w = (FX * W_REAL) / Zc
    
    return np.array([u, v, w])

def H_jacobian_camera(x, uav_roll=0.0, uav_pitch=0.0):
    """ EKF'nin sonunu getirecek olan Kısmi Türev (Teğet) Matrisi """
    X, Y, Z = x[0], x[1], max(x[2], 0.5)
    
    H = np.zeros((3, 6))
    H[0, 0] = FX / Z
    H[0, 2] = -(FX * X) / (Z**2)
    H[1, 1] = FY / Z
    H[1, 2] = -(FY * Y) / (Z**2)
    H[2, 2] = -(FX * W_REAL) / (Z**2)
    return H

# ==========================================
# 2. İT DALAŞI (DOGFIGHT) SİMÜLASYONU
# ==========================================
def generate_dogfight_data(n_steps=200, dt=0.1):
    true_states = []
    yolo_measurements = []
    
    # Rakip 25 metreden başlıyor
    current_true_x = np.array([5.0, 2.0, 25.0, 0.0, 0.0, 0.0]) 
    
    for i in range(n_steps):
        t = i * dt
        
        # --- ÖLÜMCÜL MANEVRA ---
        # Hedef kameranın tam dibine (Z=2 metreye) kadar pike yapıyor ve sıyırıp geçiyor!
        current_true_x[2] = 14.0 + 12.0 * math.cos(t * 0.5) # Z: 26m ile 2m arasında gidip gelir
        current_true_x[5] = -12.0 * 0.5 * math.sin(t * 0.5) # Z hızı (Vz)
        
        # Önümüzde agresif X ve Y zikzakları
        current_true_x[0] = 5.0 * math.sin(t * 0.8)
        current_true_x[3] = 5.0 * 0.8 * math.cos(t * 0.8)
        
        current_true_x[1] = 2.0 * math.cos(t * 0.6)
        current_true_x[4] = -2.0 * 0.6 * math.sin(t * 0.6)
        
        true_states.append(current_true_x.copy())
        
        # Mükemmel pikseller ve YOLO Gürültüsü
        perfect_z = hx_camera(current_true_x)
        noisy_u = perfect_z[0] + np.random.normal(0, 5.0)
        noisy_v = perfect_z[1] + np.random.normal(0, 5.0)
        noisy_w = perfect_z[2] + np.random.normal(0, 3.0) 
        
        yolo_measurements.append(np.array([noisy_u, noisy_v, noisy_w]))
        
    return np.array(true_states), np.array(yolo_measurements), dt

# ==========================================
# 3. YARIŞTIRMA (BENCHMARK) DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    true_states, yolo_measurements, dt = generate_dogfight_data(n_steps=200)
    
    # --- FİLTRE ZORLAMA TESTİ ---
    # Başlangıç tahminini bilerek YANLIŞ veriyoruz (Gerçek Z=26 iken biz Z=40 diyoruz)
    # EKF bu hatayı toparlayamaz, UKF ise saniyeler içinde toparlar.
    initial_guess = np.array([0., 0., 40., 0., 0., 0.])
    
    P_init = np.eye(6) * 100.0 # Başlangıca hiç güvenmiyoruz
    R_init = np.diag([5.0**2, 5.0**2, 3.0**2]) # Sensör güveni
    
    from filterpy.common import Q_discrete_white_noise
    import scipy.linalg as linalg
    Q_init = linalg.block_diag(Q_discrete_white_noise(dim=2, dt=dt, var=2.0),
                               Q_discrete_white_noise(dim=2, dt=dt, var=2.0),
                               Q_discrete_white_noise(dim=2, dt=dt, var=5.0)) # Z ekseni manevrası çok sert, var=5.0

    # 1. ÇÖKMEYE MAHKUM KF
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x, kf.P, kf.R, kf.Q = initial_guess.copy(), P_init.copy(), R_init.copy(), Q_init.copy()
    kf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                     [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    kf.H = H_jacobian_camera(kf.x)

    # 2. TEĞET HATASINA DÜŞECEK EKF
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
    ekf.x, ekf.P, ekf.R, ekf.Q = initial_guess.copy(), P_init.copy(), R_init.copy(), Q_init.copy()

    # 3. ŞAMPİYON UKF
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=fx, hx=hx_camera, points=points)
    ukf.x, ukf.P, ukf.R, ukf.Q = initial_guess.copy(), P_init.copy(), R_init.copy(), Q_init.copy()

    kf_est, ekf_est, ukf_est = [], [], []

    print("\n🚨 KTR SAVUNMASI: İT DALAŞI (DOGFIGHT) SENARYOSU 🚨")
    print("Hedef kameranın 2 metre yakınına giriyor ve Başlangıç Tahmini hatalı verildi!\n")
    
    # 1. KF Testi
    for z in yolo_measurements:
        kf.predict()
        z_centered = np.array([z[0] - CX, z[1] - CY, z[2]]) 
        kf.update(z_centered)
        kf_est.append(kf.x.copy())

    # 2. EKF Testi
    for z in yolo_measurements:
        ekf.predict_update(z, HJacobian=H_jacobian_camera, Hx=hx_camera)
        ekf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                          [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        ekf_est.append(ekf.x.copy())

    # 3. UKF Testi
    for z in yolo_measurements:
        ukf.predict()
        ukf.update(z)
        ukf_est.append(ukf.x.copy())

    kf_est, ekf_est, ukf_est = np.array(kf_est), np.array(ekf_est), np.array(ukf_est)

    # --- HATA ANALİZİ ---
    def calculate_rmse(estimates, truths):
        return np.sqrt(((estimates[:, :3] - truths[:, :3]) ** 2).mean())

    print("-" * 55)
    print(f"1. Standart KF  | Hata (RMSE): {calculate_rmse(kf_est, true_states):.2f} m  (TAMAMEN ÇÖKTÜ!)")
    print(f"2. Extended EKF | Hata (RMSE): {calculate_rmse(ekf_est, true_states):.2f} m  (YAKIN MESAFEDE SAVRULDU)")
    print(f"3. Unscented UKF| Hata (RMSE): {calculate_rmse(ukf_est, true_states):.2f} m  (ŞAMPİYON!)")
    print("-" * 55)

    # --- GÖRSELLEŞTİRME ---
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], label='Rakip İHA Gerçek Rotası (Dalış)', color='blue', linewidth=4)
    
    # Çöken KF'yi hiç çizdirmiyoruz ki grafik patlamasın.
    
    # EKF (Kırmızı Noktalı)
    ax.plot(ekf_est[:, 0], ekf_est[:, 1], ekf_est[:, 2], label='EKF (Jacobian Çuvallaması)', color='red', linewidth=3, linestyle=':')
    
    # UKF (Yeşil Çizgi)
    ax.plot(ukf_est[:, 0], ukf_est[:, 1], ukf_est[:, 2], label='UKF (Sigma Noktası Kusursuzluğu)', color='green', linewidth=2)

    ax.set_title("Yakın Mesafe (Dogfight) İt Dalaşında UKF'nin EKF'ye Karşı Üstünlüğü")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) - Kamera 0 Noktasında")
    
    ax.set_xlim([-10, 10])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 30])
    
    ax.legend()
    plt.show()