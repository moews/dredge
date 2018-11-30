# DREDGE

### Fast thresholded subspace-constrained mean shift for geospatial data

<img src="/logo.png" alt="logo" width="200px"/>

DREDGE, short for _Density Ridge Estimation Describing Geospatial Evidence_, arguably an unnecessarily forced acronym, offers a new tool to find density ridges in latitude-longitude coordinates based on the subspace-constrained mean shift (SCMS) algorithm introduced by [Ozertem and Erdogmus (2011)](http://www.jmlr.org/papers/v12/ozertem11a.html). The tool approximates principal curves for a given set of coordinates, featuring various improvements over the initial algorithm and alterations to facilitate the application to geospatial data: Thresholding, as described in cosmological research by [Chen et al.,(2015)](https://academic.oup.com/mnras/article/454/1/1140/1138949) and [Chen et al.,(2015)](https://academic.oup.com/mnras/article-abstract/461/4/3896/2608626), avoids dominant density ridges in sparsely populated areas of the dataset. In addition, the [haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) is used as a distance metric to calculate the great circle distance, which makes the tool applicable not only to city-scale data, but also to datasets spanning  multiple countries by taking the Earth's curvature into consideration.

Since DREDGE was initially developed to be applied to crime incident data, the default bandwidth calculation follows a best-practice approach that is well-accepted within quantitative criminology, using the mean distance to a given number of nearest neighbors ([Williamson et al., 1999](http://www.esri.com/news/arcuser/0199/crimedata.html)). Since practitioners in that area of study are often interested in the highest-density regions of dataset, the tool also features the possibility to specify a top-percentage level for a [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) that the ridge points should fall within.

### Installation

DREDGE can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install dredge
```

Alternatively, the file `dredge.py` can be downloaded from the folder `dredge` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide

DREDGE only requires a two-column NumPy array as its primary input (`coordinates`), with one data point per row, and latitude and longitude values in the columns. Four additional optional parameters can, however, be set: The number of nearest neighbors (`neighbors`) used to automatically calculate an optimal bandwidth can be manually changed, the bandwidth (`bandwidth`) itself can be forced to a certain value, and the threshold used to check for convergence between iterations can be set (`threshold`). The fourth parameter (`percentage`) unlocks an additional functionality of DREDGE, as the interest of practitioners is often constrained to high-density areas. For a user-provided percentage value _p_, the kernel density estimation in the tool's inner workings is used to only retain ridge points above the (100 - _p_)th percentile of the provided dataset's density landscape. This allows, for example, route matching to be focused on these areas.

<br></br>

| Variables                    | Explanations                                        | Default               |
|:-----------------------------|:----------------------------------------------------|:----------------------|
| coordinates                  | The spatial data as latitude-longitude coordinates  |                       |
| neighbors (optional)         | The number of nearest neighbors to get a bandwidth  | 10                    |
| bandwidth (optional)         | The bandwidth used for kernel density estimates     | None                  |
| convergence (optional)       | The threshold used for inter-iteration convergence  | 0.01                  |
| percentage (optional)        | The aimed-for percentage of highest-density ridges  | None                  |

<br></br>

After the installation via [PyPI](https://pypi.org), or using the `dredge.py` file locally, the usage looks like this:

```python
from dredge import filaments

filaments(coordinates = your_coordinates,
                        bandwidth = 0.1) 
```

As an example, for homicide instances from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) from 2013 to 2017, the above call to the `filaments` function results in the ridges shown in red in the left-hand figure below, with homicide instances over the given time interval depicted in cyan. Additionally setting the input parameter `percentage` is set to a value of 5 to only retain values in regions above the 95th percentile of a kernel density estimate over the provided coordinates results in the right-hand figure.

<img src="/example.png" alt="logo" width="600px"/>
