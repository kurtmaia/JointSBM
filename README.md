# Community detection over a population of graphs

`Python` implementation to perform community dectection over a heterogenous population of non-aligned networks using joint spectral clustering.

## How to use?

1) Install requirements:

`pip install pandas sklearn joblib multiprocessing argparse`

2)  Prepare data: (see data/ for examples)

 - Each graph needs to be in edge list format and `.edge` extension

 ```Python
 "from","to"
1,3
1,8
1,9
1,15
1,22
...

 ```
 - True memberships need to have `.memb` extension (if available)

3) Call `joint_spec.py` as

`python joint_spec.py --input data/N_500_varying_mu200size1_alpha_1.67/ --K 6 --parallel`

4) The algorithm returns: 

- Memberships 
- Connectivity matrix 
- Global centers

