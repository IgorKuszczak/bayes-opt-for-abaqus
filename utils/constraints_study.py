import numpy as np
import matplotlib.pyplot as plt

t_min = 1
t_l_max = np.sqrt(3)/3

def rel_dens_from_t_l(t_l):
    L = 1.0831
    x_0 = 0.3197
    k = 8.3325
    b = -0.083
    return L / (1 + np.exp(-k * (t_l - x_0))) + b


def t_l_from_rel_dens(rel_dens):
    L = 1.0831
    x_0 = 0.3197
    k = 8.3325
    b = -0.083
    return x_0 -(1/k)*np.log((L/(rel_dens-b))-1.0)


rd_max = rel_dens_from_t_l(t_l_max)

l_min = t_min*t_l_max
a_min = np.sqrt(2)*l_min

a_range = np.arange(a_min, 8.0*np.sqrt(6), 0.1)

l_range = (np.sqrt(2)/2)*a_range
t_max_range = l_range*t_l_max

t_l_buckle_limit = 0.244166
print(t_l_buckle_limit)
# plt.figure()
# plt.title('Cell size (a) versus maximum strut thickness (t_max)')
# plt.plot(t_max_range,a_range)
# plt.vlines(x=1.0, ymin = 0.0, ymax=20.0, color = 'r')
# plt.xlabel('t_max (mm)')
# plt.ylabel('a (mm)')
# plt.grid()
# plt.show()


a_samples = np.arange(2.5, 12.5, 0.1)

plt.figure()
plt.title('Minimum relative density against cell size')
t_l_min = np.sqrt(2)*t_min/a_samples
plt.plot(a_samples, rel_dens_from_t_l(t_l_min), label = 'rd_min')
plt.hlines(y=rel_dens_from_t_l(np.sqrt(3)/3), xmin=np.min(a_samples), xmax=np.max(a_samples), color = 'r', label = 'rd_max')
plt.hlines(y = rel_dens_from_t_l(t_l_buckle_limit), xmin = 0.0, xmax=np.max(a_samples),label='buckling limit', color = 'g')
plt.ylabel('minimum relative density')
plt.xlabel('unit cell size (mm)')
plt.xticks(np.arange(0, np.max(a_samples), 1))
plt.xlim([np.min(a_samples),12])
plt.fill_between(a_samples, np.maximum(rel_dens_from_t_l(t_l_min), rel_dens_from_t_l(t_l_buckle_limit)), rel_dens_from_t_l(t_l_max), alpha = 0.3)
plt.legend(loc='center right')
plt.grid()
plt.show()
