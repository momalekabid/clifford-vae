"""
plot surface area vs dimension for different manifold geometries
to show hyperspherical collapse vs clifford torus stability
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
})

# unit hypersphere S^(d-1) surface area: 2 * pi^(d/2) / Gamma(d/2)
def hypersphere_sa(d):
    return 2 * np.pi**(d/2) / gamma(d/2)

# gaussian typical set: N(0,I) in R^d concentrates on a shell at radius sqrt(d)
# the surface area of that shell is S_{d-1} * r^{d-1} where r = sqrt(d)
def gaussian_typical_shell_sa(d):
    return hypersphere_sa(d) * (np.sqrt(d))**(d - 1)

# l2-normalized gaussian lives on S^(d-1), same manifold as vMF/PowerSpherical
# so its surface area is identical to the hypersphere
# (plotted separately to make the point explicit)

# --- single plot, log scale ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# left: linear scale, short range, showing the hypersphere collapse clearly
mdims_short = np.arange(1, 40)
sa_sphere_short = np.array([hypersphere_sa(d+1) for d in mdims_short])
ax1.plot(mdims_short, sa_sphere_short, 'b-', linewidth=2,
         label=r'Power Spherical')
ax1.axvline(x=6, color='gray', linestyle='--', alpha=0.5, label='$d=6$ (peak)')
ax1.set_xlabel('manifold dimension $d$')
ax1.set_ylabel('surface area')
ax1.set_title('Hypersphere Surface Area Collapse')
ax1.legend(fontsize=9)
ax1.set_xlim(1, 39)

# right: log scale comparison, all distributions
mdims_comp = np.arange(1, 65)
sa_sphere_comp = np.array([hypersphere_sa(d+1) for d in mdims_comp])
sa_clifford_comp = np.array([(2*np.pi)**d for d in mdims_comp])
sa_gauss_comp = np.array([gaussian_typical_shell_sa(d+1) for d in mdims_comp])

ax2.semilogy(mdims_comp, sa_gauss_comp, '-', color='#2ca02c', linewidth=2,
             label=r'Gaussian')
ax2.semilogy(mdims_comp, sa_clifford_comp, 'r-', linewidth=2,
             label=r'Clifford torus')
ax2.semilogy(mdims_comp, sa_sphere_comp, 'b-', linewidth=2,
             label=r'Power Spherical')
ax2.set_xlabel('manifold dimension $d$')
ax2.set_ylabel('surface area (log scale)')
ax2.set_title('Surface Area Comparison Across Geometries')
ax2.legend(fontsize=9)
ax2.set_xlim(1, 64)

plt.tight_layout()
import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/surface_area_comparison.png', dpi=500, bbox_inches='tight')
plt.savefig('figures/surface_area_comparison.pdf', bbox_inches='tight')
print("saved to figures/")
plt.close()
