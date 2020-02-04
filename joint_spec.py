import argparse
import numpy as np 
import pandas as pd
from joblib import Parallel, delayed
import os

from jointSBM import joint_spec
# from isoSBM import iso_spec

import sys

def parse_args():
	parser = argparse.ArgumentParser(description="Joint SBM")

	parser.add_argument('--input', nargs='?', default='data/example/',
	                    help='Input graphs folder')

	parser.add_argument('--output', nargs='?', default='',
	                    help='Output results folder')

	parser.add_argument('--K', type=int, default=6,
	                    help='Number of clusters. Default is 6.')

	parser.add_argument('--maxiter', type=int, default=200,
	                    help='Maximum number of iterations. Default is 200.')

	parser.add_argument('--no-parallel', dest='parallel', action='store_false')
	parser.add_argument('--parallel', dest='parallel', action='store_true')
	parser.set_defaults(parallel=False)
	
	parser.add_argument('--n_cores', type=int, default=-1,
	                    help='Number of cores. Default is cpu_count.')

	return parser.parse_args()


def get_edgelist(f):
	edgelist = pd.read_csv(f)
	return edgelist.values

def edgelist2adj(a, nn = None):
	if nn==None:
		n = np.max(np.unique(a))
	else:
		n = nn
	adj = np.zeros([n,n])
	for i in range(a.shape[0]):
		adj[a[i,0]-1,a[i,1]-1] = 1.
	return adj

def read_graph(file,**kargs):
	adj = edgelist2adj(get_edgelist(file),**kargs)
	return adj

def get_graphs():
	files = os.listdir(args.input)
	adj_files = [f for f in files if (f.find(".edge")>0)]
	memb_files = [f for f in files if (f.find(".memb")>0)]
	graph_Names = [a.split(".")[0] for a in adj_files]
	if os.path.isfile(args.input+"n_nodes.csv"):
		n_nodes = pd.read_csv(args.input+"n_nodes.csv")
		new_columns = n_nodes.columns.values
		new_columns[0] = 'graph_id'
		n_nodes.columns = new_columns
		n_nodes = n_nodes.set_index('graph_id').to_dict("index")
		if args.parallel:
			Gs = Parallel(n_jobs=args.n_cores)(delayed(read_graph)(args.input+f+".edge", nn = n_nodes[f]['n_nodes']) for f in graph_Names)
		else: 
			Gs = [read_graph(args.input+f+".edge", nn = n_nodes[f]['n_nodes']) for f in graph_Names]
	else:
		if args.parallel:
			Gs = Parallel(n_jobs=args.n_cores)(delayed(read_graph)(args.input+f+".edge", nn = None) for f in graph_Names)
		else:
			Gs = [read_graph(args.input+f+".edge", nn = None) for f in graph_Names]
	Gs = dict(zip(graph_Names,Gs))
	if (len(memb_files)>0):
		trueXn = [pd.read_csv(args.input+a+".memb").values[:,1] for a in graph_Names]
		trueXn = dict(zip(graph_Names,trueXn))
	else:
		trueXn = None
	return Gs, trueXn

def main(args):
	if args.output == '':
		output_path = args.input+'results/'
	else:
		output_path = args.output
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	
	Graphs, trueMemberships = get_graphs()
	
	ans = joint_spec(Graphs, args.K, groundTruth = trueMemberships, maxIter = args.maxiter, parallel = args.parallel, n_cores = args.n_cores, seed = 21)
	X = ans[0]
	W = ans[1]
	theta = ans[2]
	if trueMemberships != None:
		nmis = ans[3]
		overall_nmi = ans[4]
		nmis_dt = pd.DataFrame.from_dict({(i):nmis[i] for i in X.keys()},orient='index',columns = ['nmi'])
		nmis_dt['overall_nmi'] = overall_nmi
		nmis_dt.index.name = 'graph_id'
		nmis_dt.reset_index(inplace=True)
		nmis_dt.to_csv(output_path+"nmis.csv",index = False)
		
	
	X_dt = pd.DataFrame.from_dict({(i,j):X[i][j] for i in X.keys() for j in X[i].keys()}, orient='index',columns = ['dish'])

	X_dt.index.name = 'id'
	X_dt.reset_index(inplace=True)

	W_dt = pd.DataFrame(W)
	theta_dt = pd.DataFrame(theta)

	X_dt.to_csv(output_path+"memberships.csv",index = False)
	W_dt.to_csv(output_path+"W.csv",index = False, header=False)
	theta_dt.to_csv(output_path+"theta.csv",index = False, header=False)

if __name__ == "__main__":
	args = parse_args()
	main(args)
