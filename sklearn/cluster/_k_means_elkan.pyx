import numpy as np
cimport numpy as np
cimport numpy as npc
cimport cython

from cpython cimport bool

from ..metrics import euclidean_distances
from .k_means_ import _tolerance
from ._k_means import _centers_dense


cdef d(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


cdef assign_labels(double[:, :] X, double[:, :] centers, double[:, :] center_distances):
    # assigns closest center to X
    # uses triangle inequality
    new_centers, distances = [], []
    for x in X:
        # assign first cluster center
        c_x = 0
        d_c = d(x, centers[0])
        for j, c in enumerate(centers):
            #print("d_c: %f" % d_c)
            #print("d(d_c, c'): %f" % center_distances[c_x, j])
            if d_c > center_distances[c_x, j]:
                dist = d(x, c)
                if dist < d_c:
                    d_c = dist
                    c_x = j
        new_centers.append(c_x)
        distances.append(d_c)
    return np.array(new_centers, dtype=np.int32), np.array(distances)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def k_means_elkan(X_, n_clusters, init, float tol=1e-4, int max_iter=30, verbose=False):
    #initialize
    tol = _tolerance(X_, tol)
    cdef double[:, :] centers = init
    cdef double[:, :] new_centers

    cdef int n_samples = X_.shape[0]
    cdef int n_features = X_.shape[1]
    cdef int n_centers = centers.shape[0]
    cdef int point_index, center_index, label, sample, feature
    cdef float upper_bound, distance, inertia
    cdef double[:, :] center_distances = euclidean_distances(centers) / 2.
    cdef double[:, :] lower_bounds = np.zeros((n_samples, n_centers))
    cdef double[:, :] X = X_
    cdef double[:] center_shift
    labels_, upper_bounds_ = assign_labels(X, centers,
                                           center_distances)
    cdef np.int32_t[:] labels = labels_
    cdef float[:] upper_bounds = upper_bounds_
    # make bounds tight for current labels
    for sample in range(n_samples):
        lower_bounds[sample, labels[sample]] = upper_bounds[sample]
    cdef np.uint8_t[:] bounds_tight = np.ones(n_samples, dtype=np.uint8)
    for iteration in range(max_iter):
        distance_next_center = np.sort(center_distances, axis=0)[1]
        points_to_update = distance_next_center[labels] < upper_bounds
        for point_index in np.where(points_to_update)[0]:
            upper_bound = upper_bounds[point_index]
            label = labels[point_index]
            # check other update conditions
            for center_index, center in enumerate(centers):
                if (center_index != label
                        and (upper_bound > lower_bounds[point_index, center_index])
                        and (upper_bound > center_distances[center_index, label])):
                    # update distance to center
                    if not bounds_tight[point_index]:
                        upper_bound = d(X[point_index], centers[label, :])
                        lower_bounds[point_index, label] = upper_bound
                        bounds_tight[point_index] = 1
                    # check for relabels
                    if (upper_bound > lower_bounds[point_index, center_index]
                            or (upper_bound > center_distances[label, center_index])):
                        distance = d(X[point_index], center)
                        lower_bounds[point_index, center_index] = distance
                        if distance < upper_bound:
                            label = center_index
                            upper_bound = distance
            labels[point_index] = label
            upper_bounds[point_index] = upper_bound

        # compute new centers
        new_centers = _centers_dense(X, labels, n_centers, upper_bounds)
        bounds_tight = np.zeros(n_samples, dtype=np.uint8)

        # compute distance each center moved
        center_shift = np.zeros((n_centers, n_features))
        for center_index in range(n_centers):
            for feature in range(n_features):
                center_shift[center_index] += (centers[center_index, feature] - new_centers[center_index, feature]) ** 2
            center_shift[center_index] = np.sqrt(center_shift[center_index])
        # update bounds accordingly
        for sample_index in range(n_samples):
            for cluster_index in range(n_clusters):
                lower_bounds[sample_index, center_index] = np.max(lower_bounds[sample_index, center_index] - center_shift[center_index], 0)
            upper_bounds[sample_index] = upper_bounds[sample_index] + center_shift[labels[sample_index]]
        # reassign centers
        centers = new_centers
        # update between-center distances
        center_distances = euclidean_distances(centers) / 2.
        if verbose:
            inertia = 0
            for point_index in range(n_samples):
                for feature in range(n_features):
                    inertia += np.sum((X[point_index, feature] - centers[labels[point_index], feature]) ** 2)

            print('Iteration %i, inertia %s'
                  % (iteration, inertia))

        if np.sum(center_shift) < tol:
            print("center shift within tolerance")
            break
    return centers, labels
