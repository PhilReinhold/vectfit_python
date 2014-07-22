"""
Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers


 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
"""
__author__ = 'Phil Reinhold'
from pylab import *
from numpy.linalg import pinv, eigvals

def cc(z):
    return z.conjugate()

def model(s, poles, residues, d, h):
    return sum(r/(s - p) for p, r in zip(poles, residues)) + \
           sum(r.conjugate()/(s - p.conjugate()) for p, r in zip(poles, residues)) + d + s*h

def vectfit_step(f, s, poles):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = zeros((Ns, 2*N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    x = dot(pinv(A), b)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = diag(poles)
    b = ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = real(p), imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - outer(b, c)
    H = real(H)
    new_poles = sort(eigvals(H))
    unstable = real(new_poles) > 0
    new_poles[unstable] -= 2*real(new_poles)[unstable]
    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f, s, poles):
    Ns = len(s)
    N = len(poles)

    cindex = zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = zeros((Ns, N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N] = 1
    A[:, N+1] = s
    # Solve Ax == b using pseudo-inverse
    b = f
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    x = dot(pinv(A), b)

    # Recover complex values
    x = np.complex64(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N]
    h = x[N+1]
    return residues, d, h

def make_plot(s, freal, poles, residues, d, h):
    fpoles = sum(c/(s - a) for c, a in zip(residues, poles)) + d + s*h
    xs = imag(s)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(xs, abs(freal))
    plt.plot(xs, abs(fpoles))
    plt.show()



if __name__ == '__main__':
    test_s = 1j*np.linspace(0, 1e5, 200)
    test_poles = [
        -4500,
        -41000,
        -100+5000j, -100-5000j,
        -120+15000j, -120-15000j,
        -3000+35000j, -3000-35000j,
        -200+45000j, -200-45000j,
        -1500+45000j, -1500-45000j,
        -500+70000j, -500-70000j,
        -1000+73000j, -1000-73000j,
        -2000+90000j, -2000-90000j,
    ]
    test_residues = [
        -3000,
        -83000,
        -5+7000j, -5-7000j,
        -20+18000j, -20-18000j,
        6000+45000j, 6000-45000j,
        40+60000j, 40-60000j,
        90+10000j, 90-10000j,
        50000+80000j, 50000-80000j,
        1000+45000j, 1000-45000j,
        -5000+92000j, -5000-92000j
    ]
    test_d = .2
    test_h = 2e-5

    test_f = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
    test_f += test_d + test_h*test_s


    poles = concatenate([(1j*x + x/100., -1j*x + x/100) for x in linspace(1, 1e5, 10)])
    for i in range(4):
        poles = vectfit_step(test_f, test_s, poles)
        residues, d, h = calculate_residues(test_f, test_s, poles)
        print poles
        print residues
        print d
        print h
        make_plot(test_s, test_f, poles, residues, d, h)

