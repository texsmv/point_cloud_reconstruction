import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import trimesh
from matplotlib import pyplot
# Don't delete this line, even if PyCharm says it's an unused import.
# It is required for projection='3d' in add_subplot()
from mpl_toolkits.mplot3d import Axes3D
from os.path import join


def rand_rotation_matrix(deflection=1.0, seed=None):
    """Creates a random rotation matrix.

    Args:
        deflection: the magnitude of the rotation. For 0, no rotation; for 1,
                    completely random rotation. Small deflection => small
                    perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """
    if seed is not None:
        np.random.seed(seed)

    theta, phi, z = np.random.uniform(size=(3,))

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r,
         np.cos(phi) * r,
         np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def add_rotation_to_pcloud(pcloud):
    r_rotation = rand_rotation_matrix()

    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch = batch.dot(r_rotation)
    return batch


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside
    the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
    plt.switch_backend('agg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def transform_point_clouds(X, only_z_rotation=False, deflection=1.0):
    r_rotation = rand_rotation_matrix(deflection)
    if only_z_rotation:
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
    X = X.dot(r_rotation).astype(np.float32)
    return X


def save_point_cloud(X, X_rec, epoch, n_fig=1, results_dir = "results"):
    for k in range(n_fig):
            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2], in_u_sphere=True, show=False,
                                           title=str(epoch))
            fig.savefig(
                join(results_dir, 'samples', f'{epoch:05}_{k}_real.png'))
            plt.close(fig)

    for k in range(n_fig):
        fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][1], X_rec[k][2],
                                    in_u_sphere=True, show=False,
                                    title=str(epoch))
        fig.savefig(join(results_dir, 'samples',
                            f'{epoch:05}_{k}_reconstructed.png'))
        plt.close(fig)
    
    


def show_pc(X_iso):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = list(X_iso[:, 0:1]) 
    sequence_containing_y_vals = list(X_iso[:, 1:2]) 
    sequence_containing_z_vals = list(X_iso[:, 2:3])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals) 
    


def precision(pcomp, pgt, ro):
    counter = 0
    for i in range (len(pcomp)):
        punto = pcomp[i]
        distances =  np.linalg.norm(pgt - punto, axis=1)
        min_distance = min(distances)
        if min_distance <= ro:
            counter = counter + 1
    
    return counter / (float)(len(pcomp))

def completitud(pcomp, pgt, ro):
    return precision(pgt, pcomp, ro)

def F1(pcomp, pgt, ro):
    return (completitud(pcomp, pgt, ro) + precision(pcomp, pgt, ro)) / 2.0

if __name__=='__main__':
    from utils.data import *
    mesh1 = trimesh.load("data/SimplifiedManifolds/vase1s.obj")
    mesh2 = trimesh.load("data/SimplifiedManifolds/simple_vase.obj")


    mesh2p = mesh2.sample(1024)                                                                                                                                                                                                            
    mesh1p = mesh1.sample(1024)                                                                                                                                                                                                            

    mesh1p = normalize(mesh1p)
    mesh2p = normalize(mesh2p)

    print(mesh1p.shape)

    print(precision(mesh1p, mesh2p, 0.05) )

    show_pc(np.concatenate([mesh2p , mesh1p]))

