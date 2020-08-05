import numpy as np
import matplotlib.pyplot as plt

w = np.arange(-10,10,1/10)
y = (abs(w)<0.5).astype(float)
t = len(w)
n = np.arange(len(y))
c = np.zeros(t,complex)
for k in range(t):
    output = y[n]*np.exp(-1j*w[k]*n)
    c[k] = np.sum(output)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(abs(c))

## Periodicity Property of DTFT

x1 = np.linspace(-2,2,t)
y1 = np.zeros(t,complex)
y1[abs(x1)<1] = x1[abs(x1)<1]
c1 = np.zeros(t,complex)
n1 = np.arange(len(y))
for k in range(t):
    output = y1[n1]*np.exp(-1j*w[k]*n1)
    c1[k] = np.sum(output)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(abs(c1))

##Linearity Property of DTFT

a1 = 2
a2 = 3
def sin__(t):
    s = np.sin(np.pi*t*5)
    return s
w1 = np.arange(-10,10,1/10)
t1 = len(w1)
c1 = np.zeros(t1,complex)
x1 = sin__(w1)
n1 = np.arange(len(x1))
for k1 in range(t1):
    output1 = x1[n1]*np.exp(-1j*w1[k1]*n1)
    c1[k1] = np.sum(output1)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(w1,abs(c1))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(x1)")
plt.title("DTFT of Sin function")


def rect__(t):
    s = (abs(t)<0.5).astype(float)
    return s
t2 = len(w1)
c2 = np.zeros(t2,complex)
x2 = rect__(w1)
n2 = np.arange(len(x2))
for k2 in range(t2):
    output2 = x2[n2]*np.exp(-1j*w1[k2]*n2)
    c2[k2] = np.sum(output2)
plt.subplot(322)
plt.plot(w1,abs(c2))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(X2)")
plt.title("DTFT of Rect function")

X12 = a1*x1 + a2*x2
n12 = np.arange(len(X12))
c12 = np.zeros(t1,complex)
for k12 in range(t1):
    output12 = X12[n12]*np.exp(-1j*w1[k12]*n12)
    c12[k12] = np.sum(output12)
plt.subplot(323)
plt.plot(w1,abs(c12))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(a1*x1 + a2*x2)")
plt.title("DTFT of both Sin and Rect function")


C = a1*c1 + a2*c2
plt.subplot(324)
plt.plot(w1,abs(C))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(a1*c1 + a2*c2)")
plt.title("DTFT of both Sin and Rect function")
plt.tight_layout()

## Time Shifting Property of DTFT

w = np.arange(-10,10,1/10)
def rect(t):
    s = (abs(t)<0.5).astype(float)
    return s
def dtft_(y,w):
    t = len(w)
    c1 = np.zeros(t,complex)
    n = np.arange(len(y))
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c1[k] = np.sum(output)
    return c1
fs = np.arange(-10/2,10/2,10/len(w))
x = rect
X = dtft_(x(w),w)
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(w,x(w),label="Original Signal")
plt.plot(w,x(w-3),label="Shifted Signal")
plt.legend()
plt.title("Original Shifted inputs")
X1 = dtft_(x(w-3),w)
plt.subplot(122)
plt.plot(w,X.real,label='Real x(w)')
plt.plot(w,X1.real,label='Real x(w-3)')
plt.plot(w,X1.imag,label='imag X(w-3)')
plt.legend()

## Frequency Shifting Property of DTFT

w = np.arange(-10,10,1/10)
def dft_f(y,w):
    t = len(w)
    c1 = np.zeros(t,complex)
    n = np.arange(len(y))
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c1[k] = np.sum(output)
    return c1
def rect(t):
    s = (abs(t)<0.5).astype(float)
    return s
x = rect
X = dft_f(x(w),w)
plt.figure(figsize=(12,12))
plt.plot(w,X.real,label="Original Signal")
X1 = lambda w:np.exp(2j*np.pi*5*w)*x(w)
Y = dft_f(X1(w),w)
plt.plot(w,Y.real,label="Shifted Signal")
plt.legend()

## Circular Shift Property of DTFT

import random
random_numb = random.sample(range(0,30,2),10)
random_numb.sort()
b = np.array(random_numb)
print("Original Series")
print(random_numb)
w1 = np.arange(-0.5,0.5,1/10)
def dtft_c(y,w):
    t = len(w)
    c1 = np.zeros(t,complex)
    n = np.arange(len(y))
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c1[k] = np.sum(output)
    return c1
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.stem(w1,random_numb,label="Original Signal")
plt.legend()
X = dtft_c(b,w1)
plt.subplot(322)
plt.stem(w1,abs(X),label="Original Signal DTFT")
plt.legend()


random_numb_shifted = random_numb[4::]+random_numb[:4:]
print("Shifted Series")
print(random_numb_shifted)
plt.subplot(323)
plt.stem(w1,random_numb_shifted,label="Shifted Signal")
plt.legend()
b1 = np.array(random_numb_shifted)
X1 = dtft_c(b1,w1)
plt.subplot(324)
plt.stem(w1,abs(X1),label="Shifted Signal DTFT")
plt.legend()

## Convolutional Property

w1 = np.arange(-10,10,1/10)
def sin(t):
    s = np.sin(np.pi*t*2)
    return s
def cos(t):
    s = np.cos(np.pi*t*5)
    return s
def dtft_con(y):
    t = len(w1)
    n = np.arange(len(y))
    c = np.zeros(t,complex)
    for k in range(t):
        output = y[n]*np.exp(-1j*w1[k]*n)
        c[k] = np.sum(output)
    return c
x1 = sin
X1 = dtft_con(x1(w1))
x2 = cos
X2 = dtft_con(x2(w1))
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(w1,abs(X1),label="X1")
plt.plot(w1,abs(X2),label="X2")
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(X1 and X2)")
plt.title("Original Signal")
plt.legend()

plt.subplot(322)
Y = np.convolve(x1(w1),x2(w1))
b = dtft_con(Y)
plt.plot(w1,abs(b))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(b)")
plt.title("Convolved Signal")


X = X1*X2
xn = np.fft.ifft(X)
plt.subplot(323)
plt.plot(abs(X))
plt.xlabel("Frequency")
plt.ylabel("Magnitude -> DTFT(xn)")
plt.title("Convolved Signal using np.fft.ifft() ")

np.allclose(X,b)
plt.tight_layout()

## Parseval's Relation

w = np.arange(-10,10,1/10)
def rect(t):
    s = (abs(t)<0.5).astype(float)
    return s
def dtft_Par(y,w):
    t = len(w)
    n = np.arange(len(y))
    c = np.zeros(t,complex)
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c[k] = np.sum(output)
    return c
x = rect
X = dtft_Par(x(w),w)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(w,abs(X),label='Original Signal')
P1 = np.sum(x(w)**2)
t = len(w)
P2 = np.sum(np.abs(X**2))/t
print(P1)
print(P2)
print(P1-P2)
print("Hence Proved")

## Differentiation Property of DTFT

w = np.arange(-10,10,1/10)
def sin_(t):
    X = np.sin(np.pi*t*2)
    return X
def dtft_diff(y,w):
    t = len(w)
    n = np.arange(len(y))
    c = np.zeros(t,complex)
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c[k] = np.sum(output)
    return c
x = sin_
X = dtft_diff(x(w),w)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(w,abs(X))

y = sin_
Y = dtft_diff(w*y(w),w)
plt.subplot(322)
plt.plot(w,abs(Y))

## Multiplication Property

w = np.arange(-10,10,1/10)
def sin_1(t):
    X = np.sin(np.pi*t*2)
    return X
def cos_1(t):
    X = np.cos(np.pi*t*5)
    return X
def dtft_mul(y,w):
    t = len(w)
    n = np.arange(len(y))
    c = np.zeros(t,complex)
    for k in range(t):
        output = y[n]*np.exp(-1j*w[k]*n)
        c[k] = np.sum(output)
    return c
x1 = sin_1(w)
X1 = dtft_mul(x1,w)
x2 = cos_1(w)
X2 = dtft_mul(x2,w)
plt.figure(figsize=(12,12))
plt.subplot(321)
plt.plot(abs(X1))
plt.subplot(322)
plt.plot(abs(X2))

y1 = x1*x2
Y1 = dtft_mul(y1,w)
plt.subplot(323)
plt.plot(abs(Y1))
