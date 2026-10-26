# question 1 

import math
import numpy as np
import matplotlib.pyplot as plt

# reading the file
X = []
Y = []
f = open("Dataset1.csv")
first = True
for line in f:
    if first == True:     # first line is heading x1,x2,classLabel
        first = False
        continue
    line = line.strip()
    if line == "":
        continue
    p = line.split(",")
    X.append([float(p[0]), float(p[1])])
    Y.append(float(p[2]))
f.close()
n = len(X)
print("total points", n)


# ---------------- part a : PCA ----------------
mean1 = 0
mean2 = 0
for i in range(n):
    mean1 = mean1 + X[i][0]
    mean2 = mean2 + X[i][1]
mean1 = mean1 / n
mean2 = mean2 / n

# covariance matrix (after subtracting mean)
a = 0
b = 0
c = 0
for i in range(n):
    u = X[i][0] - mean1
    v = X[i][1] - mean2
    a = a + u * u
    b = b + u * v
    c = c + v * v
C = [[a / n, b / n], [b / n, c / n]]
print("covariance matrix", C)

vals, vecs = np.linalg.eigh(np.array(C))
# eigh gives small one first so index 1 is the biggest
lam1 = vals[1]
lam2 = vals[0]
pc1 = [float(vecs[0][1]), float(vecs[1][1])]
pc2 = [float(vecs[0][0]), float(vecs[1][0])]
print("PC1 =", pc1, "eigenvalue =", lam1, "variance explained =", 100 * lam1 / (lam1 + lam2), "%")
print("PC2 =", pc2, "eigenvalue =", lam2, "variance explained =", 100 * lam2 / (lam1 + lam2), "%")

plt.figure()
for i in range(n):
    if Y[i] == 1:
        plt.scatter(X[i][0], X[i][1], c="red", s=8)
    else:
        plt.scatter(X[i][0], X[i][1], c="blue", s=8)
s1 = 2 * math.sqrt(lam1)
s2 = 2 * math.sqrt(lam2)
plt.plot([mean1, mean1 + s1 * pc1[0]], [mean2, mean2 + s1 * pc1[1]], "k", linewidth=3, label="PC1")
plt.plot([mean1, mean1 + s2 * pc2[0]], [mean2, mean2 + s2 * pc2[1]], "g", linewidth=3, label="PC2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset 1 (red = +1, blue = -1) and the 2 principal components")
plt.legend()
plt.axis("equal")
plt.savefig("q1a_pca.png")
plt.close()


# ---------------- part b and c : kernel PCA ----------------
def kernel(p, q, kind, par):
    if kind == "poly":
        return (1 + p[0] * q[0] + p[1] * q[1]) ** par
    if kind == "rbf":
        d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
        return math.exp(-d / (2 * par * par))
    if kind == "cos":
        # wave kernel: cos( 3.3*(x1-x1') - 0.6*(x2-x2') )
        return math.cos(3.3 * (p[0] - q[0]) - 0.6 * (p[1] - q[1]))

tries = [["poly", 2], ["poly", 3], ["poly", 5], ["rbf", 0.1], ["rbf", 0.16], ["rbf", 0.5], ["rbf", 1], ["rbf", 2], ["cos", 0]]

for t in tries:
    kind = t[0]
    par = t[1]

    # kernel matrix
    K = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(kernel(X[i], X[j], kind, par))
        K.append(row)

    # centering the kernel matrix
    rowavg = []
    for i in range(n):
        s = 0
        for j in range(n):
            s = s + K[i][j]
        rowavg.append(s / n)
    total = 0
    for i in range(n):
        total = total + rowavg[i]
    total = total / n
    for i in range(n):
        for j in range(n):
            K[i][j] = K[i][j] - rowavg[i] - rowavg[j] + total   # K is symmetric so col avg = row avg

    vals, vecs = np.linalg.eigh(np.array(K))
    l1 = vals[n - 1]
    l2 = vals[n - 2]
    # projection of point i on component = sqrt(eigenvalue) * eigenvector[i]
    z1 = []
    z2 = []
    for i in range(n):
        z1.append(math.sqrt(l1) * vecs[i][n - 1])
        z2.append(math.sqrt(l2) * vecs[i][n - 2])

    # part c : is it linearly separable ?  sign(w^T z) = y for all points
    # just trying w in every direction (0 to 360 degree, step 0.5) and count mistakes
    best_mist = n
    best_w = [1, 0]
    for k in range(720):
        ang = k * 0.5 * math.pi / 180
        w1 = math.cos(ang)
        w2 = math.sin(ang)
        mist = 0
        for i in range(n):
            if w1 * z1[i] + w2 * z2[i] >= 0:
                pred = 1
            else:
                pred = -1
            if pred != Y[i]:
                mist = mist + 1
        if mist < best_mist:
            best_mist = mist
            best_w = [w1, w2]

    if kind == "poly":
        name = "Polynomial kernel (1 + x.y)^" + str(par)
        fname = "q1b_poly_deg" + str(par) + ".png"
    if kind == "rbf":
        name = "RBF kernel sigma = " + str(par)
        fname = "q1b_rbf_sigma" + str(par) + ".png"
    if kind == "cos":
        name = "Cosine (wave) kernel"
        fname = "q1c_cosine_kernel.png"
    print(name, ": top 2 eigenvalues", l1, l2, ": least mistakes by a line through origin =", best_mist)

    plt.figure()
    for i in range(n):
        if Y[i] == 1:
            plt.scatter(z1[i], z2[i], c="red", s=8)
        else:
            plt.scatter(z1[i], z2[i], c="blue", s=8)
    # draw the best line w^T z = 0
    xl = plt.xlim()
    yl = plt.ylim()
    if abs(best_w[1]) > 0.000001:
        plt.plot([xl[0], xl[1]], [-best_w[0] * xl[0] / best_w[1], -best_w[0] * xl[1] / best_w[1]], "k--")
    else:
        plt.plot([0, 0], [yl[0], yl[1]], "k--")
    plt.xlim(xl)
    plt.ylim(yl)
    plt.xlabel("projection on principal component 1")
    plt.ylabel("projection on principal component 2")
    plt.title(name + "\nbest line through origin: " + str(best_mist) + " mistakes")
    plt.savefig(fname)
    plt.close()

print("done")
