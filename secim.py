import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# --- KAMERA VE İHA SABİTLERİ ---
W_REAL = 1.6  # Rakip İHA kanat açıklığı (Metre)
FX, FY = 800.0, 800.0
CX, CY = 320.0, 240.0

# ==========================================
# 1. NON-LINEAR FİZİK VE ÖLÇÜM (KAMERA) MODELLERİ
# ==========================================
def fx(x, dt):
    """ Kinematik Fizik Modeli (Tüm filtreler için ortak) """
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt 
    return np.dot(F, x)

def hx_camera(x):
    """ EKF ve UKF için DOĞRUSAL OLMAYAN (Non-Linear) Kamera Modeli """
    X, Y, Z = x[0], x[1], max(x[2], 1.0) # Z'nin sıfır olup sistemi çökertmesini engelle
    
    u = (FX * X) / Z + CX
    v = (FY * Y) / Z + CY
    w = (FX * W_REAL) / Z
    
    return np.array([u, v, w])

def H_jacobian_camera(x):
    """ EKF için Kamera Modelinin Matematiksel Türevi (Jacobian Matrisi) """
    X, Y, Z = x[0], x[1], max(x[2], 1.0)
    
    H = np.zeros((3, 6))
    # u'nun X, Y, Z'ye göre türevleri
    H[0, 0] = FX / Z
    H[0, 2] = -(FX * X) / (Z**2)
    # v'nin X, Y, Z'ye göre türevleri
    H[1, 1] = FY / Z
    H[1, 2] = -(FY * Y) / (Z**2)
    # w'nin X, Y, Z'ye göre türevleri
    H[2, 2] = -(FX * W_REAL) / (Z**2)
    
    return H

# ==========================================
# 2. SENTETİK UÇUŞ (PİKSEL ÜRETİCİSİ)
# ==========================================
def generate_pixel_measurements(n_steps=300, dt=0.1):
    true_states = []
    yolo_measurements = []
    
    # İHA 30m ileride başlıyor, bize doğru pike (dalış) yapacak
    current_true_x = np.array([5.0, -2.0, 30.0, 0.0, 0.0, 0.0]) 
    
    for i in range(n_steps):
        # Manevra: Rakip bize doğru yaklaşıyor ve zikzak çiziyor
        current_true_x[3] = 3.0 * math.sin(i * 0.1)  # X ekseninde zikzak
        current_true_x[4] = 1.0 * math.cos(i * 0.1)  # Y ekseninde dalgalanma
        current_true_x[5] = -3.0 # Z ekseninde bize doğru hızla yaklaşıyor (Burası KF'yi bitirecek)
        
        current_true_x = fx(current_true_x, dt)
        true_states.append(current_true_x.copy())
        
        # Kusursuz pikseller
        perfect_z = hx_camera(current_true_x)
        
        # YOLO Gürültüsü (Piksel hatası)
        noisy_u = perfect_z[0] + np.random.normal(0, 5.0)
        noisy_v = perfect_z[1] + np.random.normal(0, 5.0)
        noisy_w = perfect_z[2] + np.random.normal(0, 2.0)
        
        yolo_measurements.append(np.array([noisy_u, noisy_v, noisy_w]))
        
    return np.array(true_states), np.array(yolo_measurements), dt

# ==========================================
# 3. YARIŞTIRMA (BENCHMARK) DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    true_states, yolo_measurements, dt = generate_pixel_measurements()
    
    # Ortak Başlangıç Matrisleri
    P_init = np.eye(6) * 50.0
    R_init = np.diag([5.0**2, 5.0**2, 2.0**2]) # YOLO Piksel Güvenilmezliği
    
    from filterpy.common import Q_discrete_white_noise
    import scipy.linalg as linalg
    Q_init = linalg.block_diag(Q_discrete_white_noise(dim=2, dt=dt, var=1.0),
                               Q_discrete_white_noise(dim=2, dt=dt, var=1.0),
                               Q_discrete_white_noise(dim=2, dt=dt, var=1.0))

    # --- 1. STANDART KF (ÇÖKMEYE MAHKUM) ---
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.array([0., 0., 30., 0., 0., 0.])
    kf.P, kf.R, kf.Q = P_init.copy(), R_init.copy(), Q_init.copy()
    kf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                     [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    # DİKKAT: KF'ye, hedef sonsuza kadar 30 metredeymiş gibi sabit bir H matrisi veriyoruz (Çünkü non-linear hesaplayamaz)
    kf.H = H_jacobian_camera(kf.x)

    # --- 2. EXTENDED EKF (TÜREVSEL) ---
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
    ekf.x = np.array([0., 0., 30., 0., 0., 0.])
    ekf.P, ekf.R, ekf.Q = P_init.copy(), R_init.copy(), Q_init.copy()

    # --- 3. UNSCENTED UKF (SİGMA NOKTALARI) ---
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-3)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=fx, hx=hx_camera, points=points)
    ukf.x = np.array([0., 0., 30., 0., 0., 0.])
    ukf.P, ukf.R, ukf.Q = P_init.copy(), R_init.copy(), Q_init.copy()

    kf_est, ekf_est, ukf_est = [], [], []

    print("\n🚀 KTR SAVUNMASI: SENARYO B (PİKSEL UZAYI) 🚀")
    
    # 1. KF Testi (KF merkez ofsetlerini(cx, cy) matris çarpımıyla anlayamadığı için ölçümü merkezileştirerek veriyoruz)
    start_time = time.perf_counter()
    for z in yolo_measurements:
        kf.predict()
        z_centered = np.array([z[0] - CX, z[1] - CY, z[2]]) 
        kf.update(z_centered)
        kf_est.append(kf.x.copy())
    kf_time = time.perf_counter() - start_time

    # 2. EKF Testi
    start_time = time.perf_counter()
    for z in yolo_measurements:
        ekf.predict_update(z, HJacobian=H_jacobian_camera, Hx=hx_camera)
        ekf.F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                          [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        ekf_est.append(ekf.x.copy())
    ekf_time = time.perf_counter() - start_time

    # 3. UKF Testi
    start_time = time.perf_counter()
    for z in yolo_measurements:
        ukf.predict()
        ukf.update(z)
        ukf_est.append(ukf.x.copy())
    ukf_time = time.perf_counter() - start_time

    kf_est, ekf_est, ukf_est = np.array(kf_est), np.array(ekf_est), np.array(ukf_est)

    # --- HATA ANALİZİ ---
    def calculate_rmse(estimates, truths):
        return np.sqrt(((estimates[:, :3] - truths[:, :3]) ** 2).mean())

    print("-" * 55)
    print(f"1. Standart KF  | Süre: {kf_time*1000:.2f} ms | Hata (RMSE): {calculate_rmse(kf_est, true_states):.2f} m  (TAMAMEN ÇÖKTÜ!)")
    print(f"2. Extended EKF | Süre: {ekf_time*1000:.2f} ms | Hata (RMSE): {calculate_rmse(ekf_est, true_states):.2f} m")
    print(f"3. Unscented UKF| Süre: {ukf_time*1000:.2f} ms | Hata (RMSE): {calculate_rmse(ukf_est, true_states):.2f} m  (ŞAMPİYON)")
    print("-" * 55)

    # --- GÖRSELLEŞTİRME ---
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], label='Rakip İHA Gerçek Rotası', color='blue', linewidth=3)
    
    # Çöken KF (Turuncu)
    ax.plot(kf_est[:, 0], kf_est[:, 1], kf_est[:, 2], label='Standart KF (Çöktü)', color='orange', linewidth=2, linestyle='--')
    
    # EKF (Kırmızı)
    ax.plot(ekf_est[:, 0], ekf_est[:, 1], ekf_est[:, 2], label='Extended EKF (Jacobian)', color='red', linewidth=2, linestyle=':')
    
    # UKF (Yeşil)
    ax.plot(ukf_est[:, 0], ukf_est[:, 1], ukf_est[:, 2], label='Unscented UKF (Sigma Points)', color='green', linewidth=2)

    ax.set_title("Senaryo B (Piksel Verisi): Filtrelerin Reaksiyonu")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) - Bize Yaklaşıyor")
    
    # KF o kadar çok sapacak ki grafiği bozmasın diye eksen limitlerini gerçek rotaya göre ayarlıyoruz
    ax.set_xlim([-15, 15])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 35])
    
    ax.legend()
    plt.show()