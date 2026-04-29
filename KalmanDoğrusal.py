import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# ==========================================
# 1. VERİ SETİ ÜRETİCİSİ (Senin Kodun)
# ==========================================
def generate_uav_dataset(n_steps=500, dt=0.1):
    true_state = np.array([0.0, 0.0, 10.0, 2.0, 1.0, 0.1])
    process_noise_std = 0.5  # Manevra etkisini artırmak için biraz yükselttik
    measurement_noise_std = 2.0 

    waypoints = [np.array([20, 20, 15]), np.array([40, 0, 20]), np.array([20, -20, 10]), np.array([0, 0, 30])]
    current_wp = 0

    true_states = []
    measurements = []

    for i in range(n_steps):
        target = waypoints[current_wp]
        pos = true_state[0:3]
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist < 2.0:
            current_wp = (current_wp + 1) % len(waypoints)
        
        accel_to_target = (direction / (dist + 1e-6)) * 0.5
        noise = np.random.normal(0, process_noise_std, 3)
        
        true_state[0:3] += true_state[3:6] * dt
        true_state[3:6] += accel_to_target * dt + noise
        
        z = true_state[0:3] + np.random.normal(0, measurement_noise_std, 3)
        
        true_states.append(true_state.copy())
        measurements.append(z)

    return np.array(true_states), np.array(measurements), dt

# ==========================================
# 2. FİLTRE FONKSİYONLARI VE MATRİSLER
# ==========================================
# UKF ve EKF için durum geçiş (State Transition) fonksiyonu
def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return np.dot(F, x)

# UKF ve EKF için ölçüm (Measurement) fonksiyonu
def hx(x):
    return x[:3]

# EKF için Jacobian Matrisi (Türev)
def H_jacobian(x):
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

# ==========================================
# 3. BENCHMARK VE ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    n_steps = 500
    true_states, measurements, dt = generate_uav_dataset(n_steps=n_steps)
    
    # Ortak Parametreler (Adil bir yarış için hepsi aynı ayarlarda başlıyor)
    P_init = np.eye(6) * 10.0
    R_init = np.eye(3) * (2.0 ** 2)
    from filterpy.common import Q_discrete_white_noise
    import scipy.linalg as linalg
    # Basit bir Q matrisi
    q_var = 0.5
    Q_init = linalg.block_diag(np.eye(3)*0.1, np.eye(3)*q_var)

    # --- 1. STANDART KALMAN FİLTRESİ (KF) ---
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.array([0., 0., 10., 0., 0., 0.])
    kf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                     [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
    kf.P = P_init.copy()
    kf.R = R_init.copy()
    kf.Q = Q_init.copy()

    # --- 2. EXTENDED KALMAN FİLTRESİ (EKF) ---
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
    ekf.x = np.array([0., 0., 10., 0., 0., 0.])
    ekf.P = P_init.copy()
    ekf.R = R_init.copy()
    ekf.Q = Q_init.copy()

    # --- 3. UNSCENTED KALMAN FİLTRESİ (UKF) ---
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-3)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0., 0., 10., 0., 0., 0.])
    ukf.P = P_init.copy()
    ukf.R = R_init.copy()
    ukf.Q = Q_init.copy()

    # Sonuçları tutacağımız listeler
    kf_est, ekf_est, ukf_est = [], [], []
    
    # --- TEST DÖNGÜSÜ (Hız ve CPU Ölçümü) ---
    
    # KF Testi
    start_time = time.perf_counter()
    for z in measurements:
        kf.predict()
        kf.update(z)
        kf_est.append(kf.x.copy())
    kf_time = time.perf_counter() - start_time

    # EKF Testi
    start_time = time.perf_counter()
    for z in measurements:
        ekf.predict_update(z, HJacobian=H_jacobian, Hx=hx)
        # Manüel predict için F matrisini vermek gerekiyor:
        ekf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                          [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        ekf_est.append(ekf.x.copy())
    ekf_time = time.perf_counter() - start_time

    # UKF Testi
    start_time = time.perf_counter()
    for z in measurements:
        ukf.predict()
        ukf.update(z)
        ukf_est.append(ukf.x.copy())
    ukf_time = time.perf_counter() - start_time

    # Dizilere çevirme
    kf_est = np.array(kf_est)
    ekf_est = np.array(ekf_est)
    ukf_est = np.array(ukf_est)

    # --- HATA ANALİZİ (RMSE - Root Mean Square Error) ---
    def calculate_rmse(estimates, truths):
        return np.sqrt(((estimates[:, :3] - truths[:, :3]) ** 2).mean())

    kf_rmse = calculate_rmse(kf_est, true_states)
    ekf_rmse = calculate_rmse(ekf_est, true_states)
    ukf_rmse = calculate_rmse(ukf_est, true_states)

    # --- RAPORLAMA ---
    
    print(" 🚀 FİLTRE BENCHMARK SONUÇLARI 🚀")
    print(f"1. Standart KF  | Süre: {kf_time*1000:.2f} ms | Hata (RMSE): {kf_rmse:.3f} m")
    print(f"2. Extended EKF | Süre: {ekf_time*1000:.2f} ms | Hata (RMSE): {ekf_rmse:.3f} m")
    print(f"3. Unscented UKF| Süre: {ukf_time*1000:.2f} ms | Hata (RMSE): {ukf_rmse:.3f} m")
    

    # --- GÖRSELLEŞTİRME ---
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(kf_est[:, 0], kf_est[:, 1], kf_est[:, 2], label='KF Tahmini', color='orange', linewidth=6, alpha=0.4)
    ax.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2], label='Gürültülü GPS/YOLO', color='gray', s=10, alpha=0.3)
    
  # EKF'yi ekleyelim, orta kalınlıkta ve kırmızı noktalı çizgi yapalım
    ax.plot(ekf_est[:, 0], ekf_est[:, 1], ekf_est[:, 2], label='EKF Tahmini', color='red', linewidth=3, linestyle=':')
    
    # UKF'yi incecik ve tam opak yeşil yapalım (En üstte ince bir çizgi olacak)
    ax.plot(ukf_est[:, 0], ukf_est[:, 1], ukf_est[:, 2], label='UKF Tahmini', color='green', linewidth=2)

    ax.set_title("Savaşan İHA - Kalman Filtreleri Karşılaştırması")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()