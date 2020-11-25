def Gauss(A, f, debug=False):
    nonzero = []
    n = len(A)
    # no backward pass used, see third cycle
    # after this cycle we get matrix with only n nonzero elements
    for j in range(n):
        for i in range(n):
            if A[i][j] != 0 and i not in nonzero:
                nz = i
                nonzero.append(i)
                break
        # using nz will cause error if det(A) == 0
        # third cycle: iteration through all i!=nz, not i>nz
        # to avoid using backward pass
        for i in range(n):
            if i != nz:
                # A[i][j] + c_i A[nz][j] = 0
                c_i = -A[i][j] / A[nz][j]
                f[i] += c_i * f[nz]
                for jj in range(n):
                    A[i][jj] += c_i * A[nz][jj]
        del nz
    x = f.copy()
    for j in range(n):
        f[nonzero[j]] /= A[nonzero[j]][j]
        x[j] = f[nonzero[j]]
    if debug:
        return A, x, nonzero
    else:
        return x


def Gauss_maxabs(A, f, debug=False):
    maxabs = []
    rows = set()
    n = len(A)
    for k in range(n):
        ma = (0, 0)
        for i in range(n):
            if i not in rows:
                for j in range(n):
                    if abs(A[i][j]) >= abs(A[ma[0]][ma[1]]):
                        ma = (i, j)
        maxabs.append(ma)
        nz = ma[0]
        rows.add(nz)
        for i in range(n):
            if i != nz:
                # A[i][ma[1]] + c_i A[nz][ma[1]] = 0
                c_i = -A[i][ma[1]] / A[nz][ma[1]]
                f[i] += c_i * f[nz]
                for j in range(n):
                    A[i][j] += c_i * A[nz][j]
    x = f.copy()
    for i, j in maxabs:
        f[i] /= A[i][j]
        x[j] = f[i]
    if debug:
        return A, x, maxabs
    else:
        return x
