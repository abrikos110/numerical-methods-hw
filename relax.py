def upper_relaxation(A, f, ome=1, eps=1e-8):
    x = f * 0.0
    err = 1e+1000
    while err > eps:
        print(abs(f - A @ x).max())
        d = x * 0.0
        e = 0
        for i in range(len(d)):
            s = 0
            for j in range(i):
                s += A[i][j] * d[j]
            s *= ome
            le = f[i]
            for j in range(len(x)):
                le -= A[i][j] * x[j]
            s += le
            e = max(e, abs(le))
            s /= A[i][i]
            d[i] = s
        err = e
        x += d * ome
    return x
