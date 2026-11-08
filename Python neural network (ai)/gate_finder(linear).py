import numpy as np

# 1. Dataset Gerbang Logika OR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target aktual: Gerbang OR
y_true = np.array([[1], [1], [1], [0]])

# 2. Inisialisasi Bobot & Bias
np.random.seed(42)
w = np.random.randn(2, 1)  # Bobot acak awal
b = 0.0                    # Bias awal
learning_rate = 0.5

def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

def sigmoid_derivative(z):
    return z * (1.0 - z)

print("=== Kondisi Awal Sebelum Belajar ===")
print(f"Bobot awal w: [{w[0][0]:.3f}, {w[1][0]:.3f}], Bias awal b: {b:.3f}")
pred_awal = sigmoid(np.dot(X, w) + b)
for i in range(len(X)):
    print(f"  Input: {X[i]} -> Prediksi Awal: {pred_awal[i][0]:.4f} (Target: {y_true[i][0]})")

print("\n=== Proses Belajar (Training Loop) ===")
# Cetak header tabel pemantauan
print(f"{'Epoch':<8} | {'Total Loss (MSE)':<18} | {'Bobot w1':<10} | {'Bobot w2':<10} | {'Bias b':<10}")
print("-" * 65)

for epoch in range(1, 1501):
    # A. FORWARD PASS
    y_linier = np.dot(X, w) + b
    z_prediksi = sigmoid(y_linier)

    # B. HITUNG ERROR / LOSS (Mean Squared Error)
    error = z_prediksi - y_true
    loss = np.mean(error ** 2)

    # C. BACKPROPAGATION
    d_output = error * sigmoid_derivative(z_prediksi)
    grad_w = np.dot(X.T, d_output)
    grad_b = np.sum(d_output)

    # D. UPDATE BOBOT (Gradient Descent)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Tampilkan proses belajar pada interval tertentu
    if epoch in [1, 10, 50,67, 100, 300, 500, 1000, 1200, 1500]:
        print(f"{epoch:<8} | {loss:<18.6f} | {w[0][0]:<10.4f} | {w[1][0]:<10.4f} | {b:<10.4f}")

print("\n=== Hasil Akhir Setelah 1000 Epoch ===")
pred_akhir = sigmoid(np.dot(X, w) + b)
for i in range(len(X)):
    val = pred_akhir[i][0]
    print(f"Input: {X[i]} -> Prediksi: {val:.4f} (Dibulatkan: {int(round(val))}) | Target: {y_true[i][0]}")
