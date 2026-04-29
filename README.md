# Kalman Filtre Seçimi

kalmanDoğrusal dosyasında tüm filtrelerin birbirine yakın sonuç verdiği hatta hata paylarının aynı olduğu görülmüştür çünkü bu bir doğrusal hareket için geçerlidir.
Formüller temel fizik denklemlerilerine paraleldir.
Doğrusal olduğu için işlemci yükü düşük olduğundan kalman filtresinin tercih edilmesi mantıklı olandır.

UKFTakip dosyası UKF filtresine özeldir.
Her düzlemdeki hata payını ölçmektedir daha sonra değerler güncellenip başarı oranı artırılır.

secim.py dosyasında standart kalmanın çökmesinin sebebi modelin doğrusal olduğunu varsaymasıdır(doğrusal değil).
Ölçüm matrisi güncellenemediği için hesaplama hatası yapar ve kopup gider.
EKF 'nin başarısız olma nedeni ise türev hesaplamalarındaki değerlerden kaynaklanmaktadır.Bu yüzden yakın mesafede hatalı tahminler yapabilir.

secim2.py:Dog dalaşı sırasındaki takip sistemini gösterir
