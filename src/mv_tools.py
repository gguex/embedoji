import numpy as np
import scipy

# ------------------------------------------------------------------------------
# PROJECT FOR A MULTIVARIATE ANALYSIS TOOLS PACKAGE
# Guillaume Guex 
# April 2025
# ------------------------------------------------------------------------------

# Fonction for eigen decomposition
def sorted_eig(matrix, dim_max=None):
    """
    Compute the eigenvalues and eigenvectors of a real matrix, 
    sorted in descending order.
    
    Parameters:
    matrix (ndarray):   A square matrix for which to compute the eigenvalues 
                        and eigenvectors.
    dim_max (int, optional):    If specified and less than the number of rows 
                                in the matrix minus one, the function will 
                                compute the largest `dim_max` eigenvalues and 
                                corresponding eigenvectors using sparse matrix 
                                methods. Otherwise, it will compute all
                                eigenvalues and eigenvectors.
    Returns:
    tuple: A tuple containing:
        eigval (ndarray): The sorted eigenvalues in descending order.
        eigvec (ndarray): The eigenvectors corresponding to the 
        sorted eigenvalues.
    """
    if (dim_max is not None) and dim_max < matrix.shape[0] - 1:
        eigval, eigvec = scipy.sparse.linalg.eigs(matrix, dim_max)
    else:
        eigval, eigvec = scipy.linalg.eig(matrix)
    sorted_indices = eigval.argsort()[::-1]
    eigval = eigval[sorted_indices]
    eigvec = eigvec[:, sorted_indices]

    return np.real(eigval), np.real(eigvec)

# ------------------------------------------------------------------------------

# Function to construct a scalar product matrix from a distance matrix
def scalar_product_matrix(distances, weights=None):
    """
    Construct a scalar product matrix from a distance matrix.
    
    Parameters:
    distances (ndarray): A square matrix representing the distances between 
                         points.
    weights (ndarray): A vector of weights for each point. If None,
                       uniform weights are used.
    
    Returns:
    ndarray: The scalar product matrix.
    """
    n = distances.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)   
        
    centering_matrix = np.eye(n) - np.outer(np.ones(n), weights)
    scalar_product = -0.5 * centering_matrix @ distances @ centering_matrix.T
    
    return scalar_product

# ------------------------------------------------------------------------------

# Weighted scalar product matrix from a unweighted scalar product matrix
def weighted_scalar_product_matrix(scalar_product, weights=None):
    """
    Construct a weighted scalar product matrix from an unweighted scalar 
    product matrix.
    
    Parameters:
    scalar_product (ndarray): A square matrix representing the unweighted 
                              scalar product.
    weights (ndarray): A vector of weights for each point. If None,
                       uniform weights are used.
    
    Returns:
    ndarray: The weighted scalar product matrix.
    """
    n = scalar_product.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)   
        
    return np.sqrt(np.outer(weights, weights)) * scalar_product

# ------------------------------------------------------------------------------

# Function to compute the kernel matrix from a membership matrix
def kernel_membership_matrix(membership, weights=None):
    """
    Compute the kernel matrix from a membership matrix.
    
    Parameters:
    membership (ndarray): A matrix representing the group memberships of 
                          points. Each row corresponds to a point and each 
                          column corresponds to a group.
    weights (ndarray): A vector of weights for each point. If None,
                       uniform weights are used.
    
    Returns:
    ndarray: The kernel matrix.
    """
    n = membership.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
        
    # Weights for each group
    rho_g = np.sum(np.diag(weights) @ membership, axis=0)
    
    # The diagonal matrix of sqrt of weights
    sqrt_w = np.sqrt(weights)
    # The diagonal matrix of inverse of group weights
    d_i_rho_g = np.diag(1/rho_g)
    
    # Compute the kernel matrix
    kernel_matrix = np.outer(sqrt_w, sqrt_w) * \
        (membership @ d_i_rho_g @ membership.T - 1)
    
    return kernel_matrix
    

# ------------------------------------------------------------------------------

# A function to compute inter-group distance from a distance matrix
# and a group membership vector
def group_distance(distances, memberships, weights = None):
    """
    Compute the inter-group distance from a distance matrix and a group
    membership vector.
    Parameters:
    distances (ndarray): A square matrix representing the distances between 
                         points.
    memberships (ndarray): A matrix representing the group memberships of 
                           points. Each row corresponds to a point and each 
                           column corresponds to a group.
    weights (ndarray): A vector of weights for each point. If None,
                        uniform weights are used.  
    Returns:
    ndarray: The inter-group distance matrix.
    """
    n, n_g = memberships.shape
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
        
    # Weights for each group
    rho_g = np.sum(np.diag(weights) @ memberships, axis=0)
    
    # Compute the distances 
    a_ig = np.outer(weights, 1/rho_g) * memberships
    ada = a_ig.T @ distances @ a_ig
    diag_ada = np.diag(ada)
    distances_g = -0.5*(
        np.outer(diag_ada, np.ones(n_g)) \
        + np.outer(np.ones(n_g), diag_ada) \
        - 2*ada)
    
    # Return the distances
    return distances_g

# ------------------------------------------------------------------------------

# Function which computes the isometry for a kernel to another kernel
def isometry_matrix(kernel, kernel_target):
    """
    Compute the isometry matrix for a kernel to another kernel.
    
    Parameters:
    kernel (ndarray): A square matrix representing the original kernel.
    kernel_target (ndarray): A square matrix representing the target kernel.
    
    Returns:
    ndarray: The isometry matrix.
    """
    n = kernel.shape[0]
    k_eigval, k_eigvec = sorted_eig(kernel)
    k_eigval[k_eigval < 0] = 0
    sqrt_kernel = k_eigvec @ np.diag(np.sqrt(k_eigval)) @ k_eigvec.T
    composed_kernel = sqrt_kernel @ kernel_target @ sqrt_kernel
    _, ck_eigvec = sorted_eig(composed_kernel)
        
    # Compute the isometry
    isometry_matrix = k_eigvec.T @ ck_eigvec
    
    return isometry_matrix

# ------------------------------------------------------------------------------

# Function to perform a weighted MDS on a weighted scalar product matrix
def weighted_mds(kernel, weights=None, dim_max=None):
    """
    Perform a weighted MDS on a weighted scalar product matrix.
    
    Parameters:
    kernel (ndarray): A square matrix representing the weighted 
                      scalar product.
    weights (ndarray): A vector of weights for each point. If None,
                       uniform weights are used.
    dim_max (int, optional): The maximum dimension for the MDS. If None,
                              all dimensions are used.
    
    Returns:
    tuple: A tuple containing:
        - coordinates (ndarray): The coordinates of the points in the MDS space.
        - eigen_values (ndarray): The eigenvalues of the weighted scalar 
                                  product matrix.
    """
    n = kernel.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
    if dim_max is None:
        dim_max = n
    eigval, eigvec = sorted_eig(kernel, dim_max)
    eigval[eigval < 0] = 0
    coordinates = np.sqrt(np.outer(1/weights, eigval[:dim_max]))\
        *eigvec[:,:dim_max]
    
    return coordinates, eigval

# ------------------------------------------------------------------------------

# Function to perform a weighted MDS on a weighted scalar product matrix
def weighted_pca(data, weights=None, correlation=True, dim_max=None):
    """
    Perform a weighted PCA on data matrix 
    Parameters:
    data (ndarray): A matrix representing the data.
    weights (ndarray): A vector of weights for each point. If None,
                          uniform weights are used.
    correlation (bool): If True, the PCA is performed on the correlation
                        matrix. If False, it is performed on the covariance
                        matrix.
    dim_max (int, optional): The maximum dimension for the PCA. If None,
                            all dimensions are used.
    Returns:
    dictionnary: A dictonnary containing the following elements:
        - dim_max (int): The maximum dimension of the PCA.
        - eigval (ndarray): The eigenvalues.
        - coordinates (ndarray): The coordinates of the individuals.
        - saturations (ndarray): The saturation of the variables.
    """
    
    n = data.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
        
    # Center the data
    X_c = data - np.sum(data.T * weights, axis=1)
    
    # Variance of the data
    s_j = np.sqrt(np.sum((X_c**2).T * weights, axis=1))
    
    # Chose which 
    if correlation:
        X_s = X_c / s_j
        cor_mat = 1/n * X_s.T @ X_s
        eigval, eigvec = sorted_eig(cor_mat, dim_max)
        eigval[eigval < 0] = 0
        coordinates = X_s @ eigvec
        saturations = eigvec * np.sqrt(eigval)
    else: 
        cov_mat = 1/n * X_c.T @ X_c
        eigval, eigvec = sorted_eig(cov_mat, dim_max)
        eigval[eigval < 0] = 0
        coordinates = X_c @ eigvec
        saturations = (eigvec.T / s_j).T * np.sqrt(eigval)
        
    results = {
        'dim_max': dim_max,
        'eigval': eigval,
        'coordinates': coordinates,
        'saturations': saturations
    }
    
    return results

# ------------------------------------------------------------------------------

# Fonction for correspondence analysis
def correspondence_analysis(contingency):
    """
    Perform Correspondence Analysis (CA) on a given contingency table.
    Parameters:
    contingency (array-like): A 2D array representing the contingency table.
    Returns:
    dictionnary: A dictonnary containing the following elements:
        - dim_max (int): The maximum dimension of the CA.
        - eigval (ndarray): The eigenvalues.
        - row_coord (ndarray): The coordinates of the rows.
        - col_coord (ndarray): The coordinates of the columns.
        - row_contrib (ndarray): The contributions of the rows.
        - col_contrib (ndarray): The contributions of the columns.
        - row_cos2 (ndarray): The cos2 of the rows.
        - col_cos2 (ndarray): The cos2 of the columns.
    """

    contingency = np.array(contingency)
    n_row, n_col = contingency.shape
    dim_max = min(n_row, n_col) - 1

    total = np.sum(contingency)
    f_row = contingency.sum(axis=1)
    f_row = f_row / sum(f_row)
    f_col = contingency.sum(axis=0)
    f_col = f_col / sum(f_col)
    independency = np.outer(f_row, f_col) * total
    normalized_quotient = contingency / independency - 1

    b_mat = (normalized_quotient * f_col) @ normalized_quotient.T
    k_mat = np.outer(np.sqrt(f_row), np.sqrt(f_row)) * b_mat
    eigval, eigvec = sorted_eig(k_mat, dim_max)
    eigval = np.abs(eigval[:dim_max])
    eigvec = eigvec[:, :dim_max]

    row_coord = np.real(np.outer(1 / np.sqrt(f_row), np.sqrt(eigval)) \
        * eigvec)
    col_coord = (normalized_quotient.T * f_row) @ row_coord / np.sqrt(eigval)
    row_contrib = eigvec ** 2
    col_contrib = np.outer(f_col, 1 / eigval) * col_coord ** 2
    row_cos2 = row_coord ** 2
    row_cos2 = (row_cos2.T / row_cos2.sum(axis=1)).T
    col_cos2 = col_coord ** 2
    col_cos2 = (col_cos2.T / col_cos2.sum(axis=1)).T

    results = {
        'dim_max': dim_max,
        'eigval': eigval,
        'row_coord': row_coord,
        'col_coord': col_coord,
        'row_contrib': row_contrib,
        'col_contrib': col_contrib,
        'row_cos2': row_cos2,
        'col_cos2': col_cos2
    }
    
    return results

# ------------------------------------------------------------------------------

# Fonction for the RV coefficient
def rv_coefficient(K_X, K_Y):
    """
    Compute the RV coefficient between two matrices.
    
    Parameters:
    K_X (ndarray): A matrix representing the kernel of the first configuration.
    K_Y (ndarray): A matrix representing the kernel of the second configuration.
    
    Returns:
    dictionnary: A dictonnary containing the following elements:
        - rv (int): The RV coefficient.
        - lisa (ndarray): The local RV coefficients.
        - e_rv (int): The expected RV coefficient.
        - var_rv (int): The variance of the RV coefficient.
        - rv_s (int): The standardized RV coefficient.
        - p_value (float): The p-value for the RV coefficient.
    """
    n = K_X.shape[0]
    
    # Compute quantities useful for the RV coefficient
    tr_x = np.trace(K_X)
    tr_y = np.trace(K_Y)
    diag_xy = np.diag(K_X @ K_Y)
    tr_xx = np.trace(K_X @ K_X)
    tr_yy = np.trace(K_Y @ K_Y)
    v_x = tr_x ** 2 / tr_xx
    v_y = tr_y ** 2 / tr_yy
    
    # Compute the Lisa and RV coefficient
    loc_rv = diag_xy / np.sqrt(tr_xx * tr_yy)
    rv = np.sum(loc_rv)
    
    # Compute e_rv 
    e_rv = np.sqrt(v_x * v_y) / (n - 1)
    
    # Compute var_rv 
    var_rv = 2 * (n - 1 - v_x) * (n - 1 - v_y) \
        / ((n - 1) ** 2 * (n - 2) * (n + 1))
        
    # Compute the standardized RV coefficient
    rv_s = (rv - e_rv) / np.sqrt(var_rv)
    
    # Compute the p-value
    p_value = 1 - scipy.stats.norm.cdf(rv_s)
    
    return {
        "rv": rv,
        "loc_rv": loc_rv,
        "e_rv": e_rv,
        "var_rv": var_rv,
        "rv_s": rv_s,
        "p_value": p_value
    }