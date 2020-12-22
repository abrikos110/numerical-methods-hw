import numpy
from copy import deepcopy


def Gauss(A, f, extra=False):
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
    if extra:
        return A, x, nonzero
    else:
        return x


def Gauss_maxabs(A, f, extra=False):
    maxabs = []
    rows = set()
    n = len(A)
    for k in range(n):
        ma = None
        for i in range(n):
            if i not in rows:
                for j in range(n):
                    if ma is None or abs(A[i][j]) >= abs(A[ma[0]][ma[1]]):
                        ma = (i, j)
        # if det(A) == 0
        if abs(A[ma[0]][ma[1]]) == 0:
            1/0
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
    if extra:
        return A, x, maxabs
    else:
        return x


def parity_of_permutation(perm):
    if isinstance(perm[0], tuple):
        perm.sort(key=lambda x: x[0])
        *perm, = map(lambda x: x[1], perm)
    ans = 0
    # inefficient O(n*n) algorithm
    # but it is used in determinant algorithm which is O(n**3)
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            ans ^= perm[i] > perm[j]
    return ans


def det(A):
    A = numpy.array(A)
    try:
        a, x, ma = Gauss_maxabs(A, [0]*len(A), extra=True)
    except ZeroDivisionError:
        return 0
    ans = 1
    for i, j in ma:
        ans *= a[i][j]
    s = -1 if parity_of_permutation(ma) else 1
    return ans * s


def inverse_matrix(A):
    A = numpy.array(A)
    E = deepcopy(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                E[i][j] *= 0
                E[i][j] += 1
            else:
                E[i][j] *= 0
    x, inv, ma = Gauss_maxabs(A, E, extra=True)
    return inv
