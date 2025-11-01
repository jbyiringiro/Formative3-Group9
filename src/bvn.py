import numpy as np, math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def compute_mu_sigma(X2d: np.ndarray):
    mu = X2d.mean(axis=0)
    Sigma = np.cov(X2d.T, ddof=1)
    rho = Sigma[0,1] / (math.sqrt(Sigma[0,0]) * math.sqrt(Sigma[1,1]))
    return mu, Sigma, rho

def bivariate_normal_pdf(x_vec, mu_vec, Sigma_mat):
    diff = x_vec - mu_vec
    det = np.linalg.det(Sigma_mat)
    inv = np.linalg.inv(Sigma_mat)
    const = 1.0 / (2.0 * math.pi * math.sqrt(det))
    expo = -0.5 * float(diff.T @ inv @ diff)
    return const * math.exp(expo)

def make_grid(X2d: np.ndarray, pad=0.5, n=140):
    x1_min, x1_max = X2d[:,0].min()-pad, X2d[:,0].max()+pad
    x2_min, x2_max = X2d[:,1].min()-pad, X2d[:,1].max()+pad
    g1 = np.linspace(x1_min, x1_max, n)
    g2 = np.linspace(x2_min, x2_max, n)
    G1, G2 = np.meshgrid(g1, g2)
    P = np.column_stack([G1.ravel(), G2.ravel()])
    return G1, G2, P

def plot_contour_and_save(G1, G2, Z, X2d, colx, coly, mu, rho, outpath):
    plt.figure()
    plt.contour(G1, G2, Z, levels=14)
    plt.scatter(X2d[:,0], X2d[:,1], s=6)
    plt.title(f"BVN Contour — {colx} vs {coly}\nμ={mu.round(3)}, ρ={rho:.3f}")
    plt.xlabel(colx); plt.ylabel(coly)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.show()

def plot_surface3d_and_save(G1, G2, Z, colx, coly, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(G1, G2, Z, linewidth=0, antialiased=True)
    ax.set_title("BVN Surface")
    ax.set_xlabel(colx); ax.set_ylabel(coly); ax.set_zlabel("PDF")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.show()
    