# Community detection over a population of graphs

`Python` implementation to perform community dectection over a heterogenous population of non-aligned networks using joint spectral clustering.

## How to use?

1) Install requirements:

`pip install -r requirements.txt`

2)  Prepare data: (see data for examples)

 - Each graph needs to be in edge list format or regular adjancency matrix

 - Cluster retrieval perfomance can be assessed if true memberships are available

3) The algorithm returns: 

- Memberships 
- Connectivity matrix 
- Global centers

