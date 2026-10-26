# question 2  k-means, voronoi, spectral clustering

import math
import random
import numpy as np
import matplotlib.pyplot as plt

col = ["red", "blue", "green", "orange", "purple"]

# reading the file
X = []
f = open("Dataset2.csv")
for line in f:
    line = line.strip()
    if line == "":
        continue
    p = line.split(",")
    X.append([float(p[0]), float(p[1])])
f.close()
n = len(X)
print("total points", n)


def dist2(p, q):
    s = 0
    for t in range(len(p)):
        s = s + (p[t] - q[t]) * (p[t] - q[t])
    return s


# k-means (Lloyd's algorithm) - D is the data, mu is starting means
def kmeans(D, k, mu):
    m = len(D)
    dim = len(D[0])
    z = []
    for i in range(m):
        z.append(-1)
    errlist = []
    while True:
        # assign each point to nearest mean
        changes = 0
        for i in range(m):
            best = 0
            for j in range(1, k):
                if dist2(D[i], mu[j]) < dist2(D[i], mu[best]):
                    best = j
            if z[i] != best:
                z[i] = best
                changes = changes + 1
        # for error
        err = 0
        for i in range(m):
            err = err + dist2(D[i], mu[z[i]])
        errlist.append(err)
        if changes == 0:
            break
        # new means
        newmu = []
        for j in range(k):
            s = []
            for t in range(dim):
                s.append(0)
            cnt = 0
            for i in range(m):
                if z[i] == j:
                    for t in range(dim):
                        s[t] = s[t] + D[i][t]
                    cnt = cnt + 1
            if cnt == 0:
                newmu.append(mu[j])   # empty cluster, keep old mean
            else:
                for t in range(dim):
                    s[t] = s[t] / cnt
                newmu.append(s)
        mu = newmu
    return z, mu, errlist


def random_start(D, k, seed):
    random.seed(seed)
    idx = random.sample(range(len(D)), k)
    mu = []
    for i in idx:
        mu.append(list(D[i]))
    return mu


def plot_clusters(z, k, title, fname, mu=None):
    plt.figure()
    for j in range(k):
        xs = []
        ys = []
        for i in range(n):
            if z[i] == j:
                xs.append(X[i][0])
                ys.append(X[i][1])
        plt.scatter(xs, ys, c=col[j], s=8, label="cluster " + str(j + 1))
    if mu != None:
        for j in range(k):
            plt.scatter(mu[j][0], mu[j][1], c="black", marker="x", s=120)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.axis("equal")
    plt.savefig(fname)
    plt.close()


# ---------------- part a ----------------
for run in range(1, 6):
    mu = random_start(X, 4, run)
    z, mu, errlist = kmeans(X, 4, mu)
    print("init", run, "iterations", len(errlist), "final error", errlist[-1])

    plt.figure()
    plt.plot(range(1, len(errlist) + 1), errlist, "o-")
    plt.xlabel("iteration")
    plt.ylabel("error (sum of squared distance to own mean)")
    plt.title("K-means k=4, random initialisation " + str(run))
    plt.savefig("q2a_error_init" + str(run) + ".png")
    plt.close()

    plot_clusters(z, 4, "K-means k=4 clusters, initialisation " + str(run), "q2a_clusters_init" + str(run) + ".png", mu)


# ---------------- part b : voronoi regions ----------------
xmin = X[0][0]
xmax = X[0][0]
ymin = X[0][1]
ymax = X[0][1]
for i in range(n):
    if X[i][0] < xmin: xmin = X[i][0]
    if X[i][0] > xmax: xmax = X[i][0]
    if X[i][1] < ymin: ymin = X[i][1]
    if X[i][1] > ymax: ymax = X[i][1]

for k in [2, 3, 4, 5]:
    mu = random_start(X, k, 100)    # same fixed seed for every k
    z, mu, errlist = kmeans(X, k, mu)
    print("K =", k, "centers", mu)

    # colour a grid of points by which center is nearest
    gx = []
    gy = []
    gc = []
    steps = 150
    for a in range(steps + 1):
        for b in range(steps + 1):
            px = xmin + (xmax - xmin) * a / steps
            py = ymin + (ymax - ymin) * b / steps
            best = 0
            for j in range(1, k):
                if dist2([px, py], mu[j]) < dist2([px, py], mu[best]):
                    best = j
            gx.append(px)
            gy.append(py)
            gc.append(col[best])

    plt.figure()
    plt.scatter(gx, gy, c=gc, s=6, marker="s", alpha=0.2)
    xs = []
    ys = []
    for i in range(n):
        xs.append(X[i][0])
        ys.append(X[i][1])
    plt.scatter(xs, ys, c="black", s=3)
    for j in range(k):
        plt.scatter(mu[j][0], mu[j][1], c=col[j], marker="X", s=200, edgecolors="black")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Voronoi regions of the K-means centers, K = " + str(k))
    plt.savefig("q2b_voronoi_K" + str(k) + ".png")
    plt.close()


# ---------------- part c and d : spectral clustering ----------------
# kernel 1 : normal rbf  exp(-||x-y||^2 / 2s^2)
# kernel 2 : rbf on distance from origin  exp(-(|x| - |y|)^2 / 2s^2)
def kernel(p, q, kind, s):
    if kind == "rbf":
        d = dist2(p, q)
    else:
        r1 = math.sqrt(p[0] * p[0] + p[1] * p[1])
        r2 = math.sqrt(q[0] * q[0] + q[1] * q[1])
        d = (r1 - r2) * (r1 - r2)
    return math.exp(-d / (2 * s * s))

tries = [["rbf", 0.5], ["rbf", 2], ["radial", 0.5]]
k = 4
for t in tries:
    kind = t[0]
    s = t[1]
    K = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(kernel(X[i], X[j], kind, s))
        K.append(row)
    vals, vecs = np.linalg.eigh(np.array(K))

    # top 4 eigenvectors (last 4 columns), biggest first
    V = []
    for j in range(k):
        v = []
        for i in range(n):
            v.append(vecs[i][n - 1 - j])
        # sign of an eigenvector can come either way, so make its biggest entry positive
        big = 0
        for i in range(n):
            if abs(v[i]) > abs(v[big]):
                big = i
        if v[big] < 0:
            for i in range(n):
                v[i] = -v[i]
        V.append(v)

    # part c : each point -> row (v1[i], v2[i], v3[i], v4[i]), make length 1, then k-means on rows
    H = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append(V[j][i])
        L = math.sqrt(dist2(row, [0, 0, 0, 0]))
        if L > 0:
            for j in range(k):
                row[j] = row[j] / L
        H.append(row)
    # try 5 starts and keep the one with smallest error
    besterr = -1
    for seed in range(5):
        z, mu, e = kmeans(H, k, random_start(H, k, seed))
        if besterr < 0 or e[-1] < besterr:
            besterr = e[-1]
            zbest = z

    if kind == "rbf":
        name = "RBF kernel sigma = " + str(s)
    else:
        name = "Radial kernel (on distance from origin) sigma = " + str(s)
    plot_clusters(zbest, k, "Spectral clustering\n" + name, "q2c_spectral_" + kind + "_" + str(s) + ".png")

    # part d : point i goes to cluster l where v_l[i] is the biggest
    z2 = []
    for i in range(n):
        best = 0
        for j in range(1, k):
            if V[j][i] > V[best][i]:
                best = j
        z2.append(best)
    size = [0, 0, 0, 0]
    for i in range(n):
        size[z2[i]] = size[z2[i]] + 1
    print(name, "top eigenvalues", vals[n - 1], vals[n - 2], vals[n - 3], vals[n - 4], "argmax cluster sizes", size)
    plot_clusters(z2, k, "Argmax of eigenvectors\n" + name, "q2d_argmax_" + kind + "_" + str(s) + ".png")

print("done")
