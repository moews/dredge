"""Fast thresholded subspace-constrained mean shift for geospatial data.

Introduction:
-------------
DREDGE, short for 'density ridge estimation describing geospatial evidence',
arguably an unnecessarily forced acronym, is a tool to find density ridges.
Based on the subspace-constrained mean shift algorithm [1], it approximates
principal curves for a given set of latitude-longitude coordinates. Various
improvements over the initial algorithm, and alterations to facilitate the
application to geospatial data, are implemented: Thresholding, as described
in cosmological research [2, 3] avoids dominant density ridges in sparsely
populated areas of the dataset. In addition, the haversine formula is used
as a distance metric to calculate the great circle distance, which makes the
tool applicable not only to city-scale data, but also to datasets spanning 
multiple countries by taking the Earth's curvature into consideration.

Since DREDGE was initially developed to be applied to crime incident data,
the default bandwidth calculation follows a best-practice approach that is
well-accepted within quantitative criminology, using the mean distance to a 
given number of nearest neighbors [4]. Since practitioners in that area of
study are often interested in the highest-density regions of dataset, the
tool also features the possibility to specify a top-percentage level for a
kernel density estimate that the ridge points should fall within.

Quickstart:
-----------
DREDGE is designed to be easy to use and needs only one input, name the
array of latitude-longitude values for coordinates. This data has to be
provided in the form of a NumPy array with two columns, with the latitudes
in the first and the longitudes in the second column. Additionally, four 
optional parameters can be manually set by the user:

(1) The parameter 'neighbors' specifies the number of nearest neighbors
    that should be used to calculate the optimal bandwidth if the latter
    is not provided by the user. The default number of neighbors is 10.
    
(2) The parameter 'bandwidth' provides the bandwidth that is used for the 
    kernel density estimator and Gaussian kernel evaluations. By default,
    an optimal bandwidth using the average distance to a number of neighbors
    across all points in the provided dataset is calculated, with the number
    of neighbors given by the parameter 'neighbors' explained above.

(3) The parameter 'convergence' specifies the convergence threshold to
    determine when to stop iterations and return the density ridge points.
    If the resulting density ridges don't follow clearly visible lines,
    this parameter can be set to a lower value. The default is 0.01.

(4) The parameter 'percentage' should be set if only density ridge points
    from high-density regions, as per a kernel density estimate of the
    provided set of coordinates, are to be returned. If, fore example, the
    parameter is set to '5', the density ridge points are evaluated via
    the kernel density estimator, and only those above the 95th percentile,
    as opposed to all of them as the default, are returned to the user.

A simple example for using DREDGE looks like this:

    ------------------------------------------------------------
    |  from dredge import filaments                            |
    |                                                          |
    |  filaments(coordinates = your_latitudes_and_longitudes,  |
    |            percentage = your_top_density_percentage)     |
    |                                                          |
    ------------------------------------------------------------

Here, the optional parameter 'percentage', which is explained above, is used.

Author:
--------
Ben Moews
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh

References:
-----------
[1] Ozertem, U. and Erdogmus, D. (2011): "Locally defined principal curves 
    and surfaces", JMLR, Vol. 12, pp. 1249-1286
[2] Chen, Y. C. et al. (2015), "Cosmic web reconstruction through density 
    ridges: Method and algorithm", MNRAS, Vol. 454, pp. 1140-1156
[3] Chen, Y. C. et al. (2016), "Cosmic web reconstruction through density 
    ridges: Catalogue", MNRAS, Vol. 461, pp. 3896-3909
[4] Williamson, D. et al. (1999), "A better method to smooth crime incident 
    data", ESRI ArcUser Magazine, January-March 1999, pp. 1-5
    
Packages and versions:
----------------------
The versions listed below were used in the development of X, but the exact 
version numbers aren't specifically required. The installation process via 
PyPI will take care of installing or updating every library to at least the
level that fulfills the requirement of providing the necessary functionality.

Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
Scikit-learn 0.19.1
"""
# Load the necessary libraries
import sys
import numpy as np
import scipy as sp
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import NearestNeighbors as KNN

def filaments(coordinates, 
              neighbors = 10, 
              bandwidth = None, 
              convergence = 0.005,
              percentage = None):
    """Estimate density rigdges for a user-provided dataset of coordinates.
    
    This function uses an augmented version of the subspace-constrained mean
    shift algorithm to return density ridges for a set of langitude-longitude
    coordinates. Apart from the haversine distance to compute a more accurate
    version of a common optimal kernel bandwidth calculation in criminology,
    the code also features thresholding to avoid ridges in sparsely populated
    areas. While only the coordinate set is a required input, the user can
    override the number of nearest neighbors used to calculate the bandwidth
    and the bandwidth itself, as well as the convergence threshold used to
    assess when to terminate and the percentage indicating which top-level of
    filament points in high-density regions should be returned. If the latter
    is not chose, all filament points are returned in the output instead.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    neighbors : int, defaults to 10
        The number of neighbors used for the optimal bandwidth calculation.
    
    bandwidth : float, defaults to None
        The bandwidth used for kernel density estimates of data points.
    
    convergence : float, defaults to 0.005
        The convergence threshold for the inter-iteration update difference.
    
    percentage : float, defaults to None
        The percentage of highest-density filament points that are returned.
        
    Returns:
    --------
    ridges : array-like
        The coordinates for the estimated density ridges of the data.
        
    Attributes:
    -----------
    None
    """
    # Check if the inputs are valid
    parameter_check(coordinates = coordinates,
                    neighbors = neighbors,
                    bandwidth = bandwidth,
                    convergence = convergence,
                    percentage = percentage)
    print("Input parameters valid!\n")
    print("Preparing for iterations ...\n")
    # Check whether no bandwidth is provided
    if bandwidth is None:
        # Compute the average distance to the given number of neighbors
        nearest_neighbors = KNN(n_neighbors = neighbors,
                            algorithm = 'ball_tree', 
                            metric = 'haversine').fit(coordinates)
        distances, _ = nearest_neighbors.kneighbors(X = coordinates)
        bandwidth = np.mean(distances[:, 1:distances.shape[1]])
        print("Automatically computed bandwidth: %f\n" % bandwidth)
    # Compute a Gaussian KDE with the haversine formula
    density_estimate = KDE(bandwidth = bandwidth, 
                           metric = 'haversine',
                           kernel = 'gaussian',
                           algorithm = 'ball_tree').fit(coordinates)
    # Create an evenly-spaced mesh in for the provided coordinates
    mesh = mesh_generation(coordinates)
    # Compute the threshold to omit mesh points in low-density areas
    threshold, densities = threshold_function(mesh, density_estimate)
    # Cut low-density mesh points from the set
    ridges = mesh[densities > threshold, :]
    # Intitialize the update change as larger than the convergence
    update_change = np.multiply(2, convergence)
    # Initialize the previous update change as zero
    previous_update = 0
    # Loop over the number of prescripted iterations
    iteration_number = 0
    #while not update_change < convergence:
    while not update_change < convergence:
        # Update the current iteration number
        iteration_number = iteration_number + 1
        # Print the current iteration number
        print("Iteration %d ..." % iteration_number)
        # Create a list to store all update values
        updates = []
        # Loop over the number of points in the mesh
        for i in range(ridges.shape[0]):
            # Compute the update movements for each point
            point_updates = update_function(ridges[i], coordinates, bandwidth)
            # Add the update movement to the respective point
            ridges[i] = ridges[i] + point_updates
            # Store the change between updates to check convergence
            updates.append(np.abs(np.mean(np.sum(point_updates))))
        # Get the update change to check convergence
        update_average = np.mean(np.sum(updates))
        update_change = np.abs(previous_update - update_average)
        previous_update = update_average
    # Check whether a top-percentage of points should be returned
    if percentage is not None:
        # Evaluate all mesh points on the kernel density estimate
        evaluations = density_estimate.score_samples(ridges)
        # Calculate the threshold value for a given percentage
        valid_percentile = np.percentile(evaluations, [100 - percentage])
        # Retain only the mesh points that are above the threshold
        ridges = ridges[np.where(evaluations > valid_percentile)]
    # Return the iteratively updated mesh as the density ridges
    print("\nDone!")
    return ridges

def haversine(point_1, 
              point_2):
    """Calculate the haversine distance between two coordinates.
    
    This function calculates he haversine formula for two latitude-longitude
    tuples, a formula used for the great-circle distance on a sphere. While
    the effect of using this more accurate distance, as opposed to the more
    common Euclidean distance, is negligible for smaller scales, this choice
    allows the code to also be used on larger scales by taking the curvature
    of the Earth into account.
    
    Parameters:
    -----------
    point_1 : array-like
        The coordinates for a point as a tuple of type [float, float].
        
    point_2 : array-like
        The coordinates for a point as a tuple of type [float, float].
        
    Returns:
    --------
    haversine_distance : float
        The haversine distance between the two provided points.
        
    Attributes:
    -----------
    None
    """
    # Specify the radius of the Earth in kilometers
    earth_radius = 6372.8
    # Extract latitudes and longitudes from the provided points
    latitude_1 = point_1[0]
    latitude_2 = point_2[0]
    longitude_1 = point_1[1]
    longitude_2 = point_2[1]
    # Convert the latitudes and longitudes to radians
    latitude_1, longitude_1 = np.radians((latitude_1, longitude_1))
    latitude_2, longitude_2 = np.radians((latitude_2, longitude_2))
    # Calculate the differences between latitudes in radians
    latitude_difference = latitude_2 - latitude_1
    # Calculate the differences between longitudes in radians
    longitude_difference = longitude_2 - longitude_1
    # Calculate the haversine distance between the coordinates
    step_1 = np.square(np.sin(np.multiply(latitude_difference, 0.5)))
    step_2 = np.square(np.sin(np.multiply(longitude_difference, 0.5)))
    step_3 = np.multiply(np.cos(latitude_1), np.cos(latitude_2))
    step_4 = np.arcsin(np.sqrt(step_1 + np.multiply(step_2, step_3)))
    haversine_distance = np.multiply(np.multiply(2, earth_radius), step_4)
    # Return the computed haversine distance for the coordinates
    return haversine_distance

def mesh_generation(coordinates):
    """Generate a set of uniformly-random distributed points as a mesh.
    
    The subspace-constrained mean shift algorithm operates on either a grid
    or a uniform-random set of coordinates to iteratively shift them towards
    the estimated density ridges. Due to the functionality of the code, the
    second approach is chosen, with a uniformly-random set of coordinates
    in the intervals covered by the provided dataset as a mesh. In order to
    not operate on a too-small or too-large number of mesh points, the size
    of the mesh is constrained to a lower limit of 50,000 and an upper limit
    of 100,000, with the size of the provided dataset being used if it falls
    within these limits. This is done to avoid overly long running times.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
        
    Returns:
    --------
    mesh : array-like
        The set of uniform-random coordinates in the dataset's intervals.
        
    Attributes:
    -----------
    None
    """
    # Get the minimum and maximum for the latitudes
    min_latitude = np.min(coordinates[:, 0])
    max_latitude = np.max(coordinates[:, 0])
    # Get the minimum and maximum for the longitudes
    min_longitude = np.min(coordinates[:, 1])
    max_longitude = np.max(coordinates[:, 1])
    # Get the number of provided coordinates
    size = int(np.min([1e5, np.max([5e4, len(coordinates)])]))
    # Create an array of uniform-random points as a mesh
    mesh_1 = np.random.uniform(min_latitude, max_latitude, size)
    mesh_2 = np.random.uniform(min_longitude, max_longitude, size)
    mesh = np.vstack((mesh_1.flatten(), mesh_2.flatten())).T
    # Return the evenly-spaced mesh for the coordinates
    return mesh

def threshold_function(mesh, 
                       density_estimate):
    """Calculate the cut-off threshold for mesh point deletions.
    
    This function calculates the threshold that is used to deleted mesh
    points from the initial uniformly-random set of mesh points. The
    rationale behind this approach is to avoid filaments in sparsely
    populated regions of the provided dataset, leading to a final result
    that only covers filaments in regions of a suitably high density.
    
    Parameters:
    -----------
    mesh : array-like
        The set of uniform-random coordinates in the dataset's intervals.
        
    density_estimate : scikit-learn object
        The kernel density estimator fitted on the provided dataset.
        
    Returns:
    --------
    threshold : float
        The cut-off threshold for the omission of points in the mesh.
    
    density_array : array-like
        The density estimates for all points in the given mesh.
        
    Attributes:
    -----------
    None
    """
    # Calculate the average of density estimates for the data
    density_array = np.exp(density_estimate.score_samples(mesh))
    density_sum = np.sum(density_array)
    density_average = np.divide(density_sum, len(mesh))
    # Compute the threshold via the RMS in the density fluctuation
    density_difference = np.subtract(density_array, density_average)
    square_sum = np.sum(np.square(density_difference))
    threshold = np.sqrt(np.divide(square_sum, len(density_difference)))
    # Return the threshold for the provided mesh and density etimate
    return threshold, density_array

def update_function(point, 
                    coordinates, 
                    bandwidth):
    """Calculate the mean shift update for a provided mesh point.
    
    This function calculates the mean shift update for a given point of 
    the mesh at the current iteration. This is done through a spectral
    decomposition of the local inverse covariance matrix, shifting the
    respective point closer towards the nearest estimated ridge. The
    updates are provided as a tuple in the latitude-longitude space to
    be added to the point's coordinate values.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
        
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
        
    Returns:
    --------
    point_updates : float
        The tuple of latitude and longitude updates for the mesh point.
        
    Attributes:
    -----------
    None
    """
    # first, calculate the interpoint distance
    squared_distance = np.sum(np.square(coordinates - point), axis=1)
    # eqvaluate the kernel at each distance
    weights = gaussian_kernel(squared_distance, bandwidth)
    # now reweight each point
    shift = np.divide(coordinates.T.dot(weights), np.sum(weights))
    # first, we evaluate the mean shift update
    update = shift - point
    # Calculate the local inverse covariance for the decomposition
    inverse_covariance = local_inv_cov(point, coordinates, bandwidth)
    # Compute the eigendecomposition of the local inverse covariance
    eigen_values, eigen_vectors = np.linalg.eig(inverse_covariance)
    # Align the eigenvectors with the sorted eigenvalues
    sorted_eigen_values = np.argsort(eigen_values)
    eigen_vectors = eigen_vectors[:, sorted_eigen_values]
    # Cut the eigenvectors according to the sorted eigenvalues
    cut_eigen_vectors = eigen_vectors[:, 1:]
    # Project the update to the eigenvector-spanned orthogonal subspace
    point_updates = cut_eigen_vectors.dot(cut_eigen_vectors.T).dot(update)    
    # Return the projections as the point updates
    return point_updates

def gaussian_kernel(values, 
                    bandwidth):
    """Calculate the Gaussian kernel evaluation of distance values.
    
    This function evaluates a Gaussian kernel for the squared distances
    between a mesh point and the dataset, and for a given bandwidth.
    
    Parameters:
    -----------
    values : array-like
        The distances between a mesh point and provided coordinates.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    kernel_value : float
        The Gaussian kernel evaluations for the given distances.
        
    Attributes:
    -----------
    None
    """
    # Compute the kernel value for the given values
    temp_1 = np.multiply(np.pi, np.square(bandwidth))
    temp_2 = np.divide(1, np.sqrt(temp_1))
    temp_3 = np.divide(values, np.square(bandwidth))
    kernel_value = np.exp(np.multiply(np.negative(0.5), temp_3))
    # Return the computed kernel value
    return kernel_value

def local_inv_cov(point, 
                  coordinates, 
                  bandwidth):
    """Compute the local inverse covariance from the gradient and Hessian.
    
    This function computes the local inverse covariance matrix for a given
    mesh point and the provided dataset, using a given bandwidth. In order
    to reach this result, the covariance matrix for the distances between
    a mesh point and the dataset is calculated. After that, the Hessian
    matrix is used to calculate the gradient at the given point's location.
    Finally, the latter is used to arrive at the local inverse covariance.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
    
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    inverse_covariance : array-like
        The local inverse covariance for the given point and coordinates.
        
    Attributes:
    -----------
    None
    """
    # Calculate the squared distance between points
    squared_distance = np.sum(np.square(coordinates - point), axis=1)
    # Compute the average of the weights as the estimate
    weights = gaussian_kernel(squared_distance, bandwidth)
    weight_average = np.mean(weights)
    # Get the number of points and the dimensionality
    number_points, number_columns = coordinates.shape 
    # Calculate one over the given bandwidth
    fraction_1 = np.divide(1, np.square(bandwidth))
    # Calculate one over the given number of points
    fraction_2 = np.divide(1, number_points)
    # Compute the mean for the provided points
    mu = np.multiply(fraction_1, (coordinates - point))
    # Compute the covariance matrix for the provided points
    covariance = gaussian_kernel(squared_distance, bandwidth)
    # Compute the Hessian matrix for the provided points
    temp_1 = np.multiply(fraction_1, np.eye(number_columns))
    temp_2 = (np.multiply(covariance, mu.T)).dot(mu)
    temp_3 = np.multiply(fraction_2, temp_2)
    temp_4 = np.multiply(temp_1, np.sum(covariance))
    hessian = temp_3 - np.multiply(fraction_2, temp_4)
    # Get the number of data points and the dimensionality
    number_rows, number_columns = coordinates.shape 
    # Compute the gradient at the given point for the data
    temp_5 = np.mean(np.multiply(covariance, mu.T), axis=1)
    gradient = np.negative(temp_5)
    # Compute the loval inverse covariance for the inputs
    temp_6 = np.divide(np.negative(1), weight_average)
    temp_7 = np.divide(1, np.square(weight_average))
    temp_8 = np.multiply(temp_7, gradient.dot(gradient.T))
    inverse_covariance = np.multiply(temp_6, hessian) + temp_8
    # Return the local inverse covariance
    return inverse_covariance

def parameter_check(coordinates, 
                    neighbors, 
                    bandwidth, 
                    convergence, 
                    percentage):
    """Check the main function inputs for unsuitable formats or values.
    
    This function checks all of the user-provided main function inputs for 
    their suitability to be used by the code. This is done right at the
    top of the main function to catch input errors early and before any
    time is spent on time-consuming computations. Each faulty input is
    identified, and a customized error message is printed for the user
    to inform about the correct inputs before the code is terminated.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    neighbors : int
        The number of neighbors used for the optimal bandwidth calculation.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
    
    convergence : float
        The convergence threshold for the inter-iteration update difference.
    
    percentage : float
        The percentage of highest-density filament points that are returned.
        
    Returns:
    --------
    None
        
    Attributes:
    -----------
    None
    """
    # Create a boolean vector to keep track of incorrect inputs
    incorrect_inputs = np.zeros(5, dtype = bool)
    # Check whether two-dimensional coordinates are provided
    if not type(coordinates) == np.ndarray:
        incorrect_inputs[0] = True
    elif not coordinates.shape[1] == 2:
        incorrect_inputs[0] = True
    # Check whether neighbors is a positive integer or float
    if not ((type(neighbors) == int and neighbors > 0)
        and not ((type(neighbors) == float) 
                 and (neighbors > 0)
                 and (neighbors.is_integer() == True))):
        incorrect_inputs[1] = True
    # Check whether bandwidth is a positive integer or float
    if not bandwidth == None:
        if not ((type(bandwidth) == int and bandwidth > 0)
            or (type(bandwidth) == float) and bandwidth > 0):
            incorrect_inputs[2] = True
    # Check whether convergence is a positive integer or float
    if not convergence == None:
        if not ((type(convergence) == int and convergence > 0)
            or (type(convergence) == float) and convergence > 0):
            incorrect_inputs[3] = True
    # Check whether percentage is a valid percentage value
    if not percentage == None:
        if not ((type(percentage) == int and percentage >= 0 
                 and percentage <= 100)
                or ((type(percentage) == float) and percentage >= 0 
                    and percentage <= 100)):
            incorrect_inputs[4] = True
    # Define error messages for each parameter failing the tests
    errors = ['ERROR: coordinates: Must be a 2-column numpy.ndarray',
              'ERROR: neighbors: Must be a whole-number int or float > 0',
              'ERROR: bandwidth: Must be an int or float > 0, or None',
              'ERROR: convergence: Must be an int or float > 0, or None',
              'ERROR: percentage: Must be an int or float in [0, 100], or None']
    # Print eventual error messages and terminate the code
    if any(value == True for value in incorrect_inputs):
        for i in range(0, len(errors)):
            if incorrect_inputs[i] == True:
                print(errors[i])
        sys.exit()
