import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b, c):
    f = a*x**b + c
    return f

xdata = np.array([17, 250, 700, 1700])
ydata = np.array([0.08, 0.02, 0.01, 0.006])

popt, pcov = curve_fit(func, xdata, ydata, maxfev=2000)

a,b, c = popt
print(a)
print(b)
print(c)


x = np.linspace(10, 1200, 100)
y = func(x, a, b, c)

plt.plot(x, y)
plt.scatter(xdata, ydata)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xscale('log')
plt.yscale('log')
plt.show()
