import numpy as np
from scipy import stats
from math import e
import matplotlib.pyplot as plt

# rand_norm_list = np.random.normal(0,1,(100))
# print(rand_norm_list)
# sum = 0
# for _ in rand_norm_list:
#     sum += _
# print(sum/100)

x=1
b = stats.norm.pdf(0,0,x)
c = stats.norm.cdf(0,0,x)-stats.norm.cdf(-0.1,0,x)

print(b)
print(c)
x = np.linspace(stats.norm.ppf(0.01),stats.norm.ppf(0.99), 100)
fig, ax = plt.subplots(1, 1)
ax.plot(x, stats.norm.pdf(x,0,1), 'r-', lw=5, alpha=0.6, label='norm pdf1')
ax.plot(x, stats.norm.pdf(x,0,10), 'r-', lw=4, alpha=0.6, label='norm pdf2')
ax.plot(x, stats.norm.pdf(x,0,10000), 'r-', lw=3, alpha=0.6, label='norm pdf2')

plt.show()