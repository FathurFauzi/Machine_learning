import numpy as np

# Dataset XOR: tidak bisa dipisahkan satu garis lurus
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [0],
    [1]
])

np.random.seed(1)
lr = 0.5 # learning rate
epochs = 10000

# Inisialisasi bobot (kecil & acak) dan bias nol
W1 = np.random.randn(2, 2)   # input (2) -> hidden (2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)   # hidden (2) -> output (1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

for epoch in range(1, epochs + 1):
    # ---- FORWARD ----
    H = sigmoid(X @ W1 + b1)          # 4x2
    Z = sigmoid(H @ W2 + b2)          # 4x1 ;   @ itu perkalian matrix

    # ---- BACKWARD ----
    d2 = (Z - y) * Z * (1 - Z)        # 4x1, error output
    grad_W2 = H.T @ d2                # 2x1
    grad_b2 = np.sum(d2, axis=0, keepdims=True)

    d1 = (d2 @ W2.T) * H * (1 - H)    # 4x2, error dikirim balik ke hidden
    grad_W1 = X.T @ d1                # 2x2
    grad_b1 = np.sum(d1, axis=0, keepdims=True)

    # ---- UPDATE ----
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1

    if epoch in [1, 1000, 5000, 10000]:
        loss = np.mean((Z - y) ** 2)
        print(f"Epoch {epoch:<6} | loss = {loss:.6f}")

print("\nHasil akhir:")
for i in range(len(X)):
    print(f"Input {X[i]} -> prediksi {Z[i][0]:.4f} | target {y[i][0]}")