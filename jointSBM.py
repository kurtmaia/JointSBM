import numpy as np 
from numpy.linalg import inv, cholesky, eig
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans

from joblib import Parallel, delayed
import multiprocessing

def get_Qn(adj,K):
	eig_decomp = eig(adj)
	args = np.argsort(-abs(eig_decomp[0]),)[:K]
	D = (eig_decomp[0][args])
	U = (eig_decomp[1][:,args])
	return abs(np.matmul(U,np.diag(D)))

def onehot_vec(arg,K):
	ans = np.zeros([K])
	ans[arg] = 1.
	return ans

def frobenius_norm(M):
	return np.sqrt(np.trace(np.matmul(M.T,M)))

def inv_s(M):
	diag_m = 1/np.diag(M)
	diag_m[diag_m==float('inf')] = 0
	return(np.diag(diag_m))

def get_counts(X):
	delta2_n = np.array([np.diag(np.sum(X[i],axis=0)) for i in range(len(X))])
	delta2 = np.sum(delta2_n,axis=0)

	gamma_n = np.array([np.sum(inv(delta2)*delta2_n[i]) for i in range(len(X))])
	return delta2_n, delta2, gamma_n


def get_collapseds_spec(x,An):
	K = x.shape[0]
	An_collapsed = np.matmul(np.matmul(x.T,An*np.tril(An)),x)
	An_collapsed = An_collapsed+An_collapsed.T
	np.fill_diagonal(An_collapsed,np.diag(An_collapsed)/2)

	mm = np.reshape(np.sum(x,0),[-1,1])

	N = np.matmul(mm,mm.T)
	np.fill_diagonal(N, mm*(mm-1)/2)
	return An_collapsed, N

def estimate_theta(X,An):
	An_c, N_c = np.sum([get_collapseds_spec(X[i],An[i]) for i in range(len(X))],axis = 0)
	ans = np.zeros(N_c.shape)
	ans[N_c > 0.] = An_c[N_c > 0.]*1./N_c[N_c > 0.]
	return ans

def update_W(X,Q,deltas_):
	K = X[0].shape[1]
	delta2_n, delta2, gamma_n = deltas_

	XnQn = []
	XXn = []
	for i in range(len(Q)):
		rt = np.sum(delta2)/np.sum(delta2_n[i])
		XnQn.append(np.matmul(X[i].T,Q[i]*gamma_n[i]))
		XXn.append(delta2_n[i]*gamma_n[i])
	XQ = np.sum(XnQn,axis = 0)	
	denom = np.sum(XXn,axis = 0)

	W = np.matmul(inv(denom),XQ)
	return W


def update_xni(q,W,x,V,Vn,q_q):
	K = W.shape[0]
	rt = np.sum(V)/np.sum(Vn)
	dif = W - q
	gamman = np.ndarray([K])
	term2 = np.ndarray([K])

	for k in range(K):
		fac = np.diag((Vn - x + np.eye(K)[k,:])/(V - x + np.eye(K)[k,:]))
		term2[k] = np.sum((np.matmul(q,np.sqrt(fac) + np.eye(K)/np.sqrt(rt)))**2)
		gamman[k] = np.sum(fac)

	dist = np.diag(np.matmul(dif,dif.T))*gamman + term2

	x = onehot_vec(np.argmin(dist),K)
	return x

def initX(Qn, K, n_nodes, seed, init):
	if init == 'kmeans++':
		km = KMeans(n_clusters=K,random_state=seed).fit(Qn).labels_
		Xn = np.vstack([onehot_vec(r,K) for r in km])
	else:
		Xn = np.random.multinomial(1,[1./K]*K,size = n_nodes)
	return Xn

def processLoop(Qn,W,Xn,V,Q_Qn,deltas2_n):
	Vn = np.diag(deltas2_n)
	for i in range(Xn.shape[0]):
 		Xn[i,] = update_xni(Qn[i,],W,Xn[i,],V,Vn,Q_Qn)
 	return Xn


def joint_spec(graphs, K, tol = 1e-15, groundTruth = None, maxIter = 200, init = 'kmeans++', parallel = False, n_cores = -1,seed = 242):
	if (type(graphs)!=dict):
		graphNames = ["graph"+str(g) for g in range(1,len(graphs)+1)]
		graphs = dict(zip(graphNames,graphs))
	else:	
		graphNames = graphs.keys()
	if (groundTruth!=None):
		if (type(groundTruth)==dict):
			groundTruth = groundTruth.values()	

	graphs = graphs.values()
	n_graphs = len(graphs)

	n_nodes_array = [graphs[i].shape[0] for i in range(n_graphs)]
	total_nodes = np.sum(n_nodes_array)

	# if parallel:
	# 	Q = Parallel(n_jobs=n_cores)(delayed(lambda i,K: get_Qn(graphs[i],K)*np.sqrt(total_nodes/n_nodes_array[i]))(i,K) for i in range(n_graphs))
	# 	X = Parallel(n_jobs=n_cores)(delayed(initX)(Q[i],K,n_nodes_array[i], seed, init) for i in range(n_graphs))
	# else:
	Q = []
	X = []

	for i in range(n_graphs):
		Qn = get_Qn(graphs[i],K)*np.sqrt(total_nodes/n_nodes_array[i])
		Q.append(Qn)
		X.append(initX(Qn, K,n_nodes_array[i],seed, init))

	deltas_ = get_counts(X)
	Q_Q = np.array([np.matmul(Q[i].T,Q[i])**2 for i in range(len(Q))])


	Loss = [0]
	stopValue = 1
	iter = -1
	while (stopValue > tol and iter < maxIter):
		iter = iter + 1

		W = update_W(X,Q,deltas_)

		V = np.diag(deltas_[1])
		memberships = []
		counts_memberships = np.zeros([1,K])
		if parallel:
			if n_cores == -1:
				num_cores = multiprocessing.cpu_count()
			else: 
				num_cores = n_cores
			X = Parallel(n_jobs=num_cores)(delayed(processLoop)(Q[n],W,X[n],V,Q_Q[n],deltas_[0][n]) for n in range(n_graphs))
			for x in X:
				memberships.append(np.argmax(x,1))
				counts_memberships += np.sum(x,0)
		else:
			for n in range(n_graphs):
				X[n] = processLoop(Q[n],W,X[n],V,Q_Q[n],deltas_[0][n])
				memberships.append(np.argmax(X[n],1))
				counts_memberships += np.sum(X[n],0)
		if (np.sum(counts_memberships==0.)>0):
			print "Restarting..."
			X = [np.random.multinomial(1,[1./K]*K,size = n_nodes_array[i]) for i in range(n_graphs)]
		
		deltas_ = get_counts(X)

		loss = np.ndarray([n_graphs])
		for n in range(n_graphs):
			XW = np.matmul(X[n],W)
			rt = np.sum(deltas_[1])/np.sum(deltas_[0][n])
			term = np.matmul(inv(deltas_[1]),deltas_[0][n]) + np.eye(K)*1./rt
			loss[n] = deltas_[2][n]*frobenius_norm(XW-Q[n])**2 + frobenius_norm(np.matmul(Q[n],term))**2

		Loss.append(np.sum(loss))
		print "Iter:",iter,"| Loss:", Loss[iter]

		stopValue =  abs(Loss[iter] - Loss[iter-1])	

		if (groundTruth!=None):
			individual_nmi = np.ndarray([n_graphs])
			for n in range(n_graphs):
				individual_nmi[n] = nmi(np.reshape(groundTruth[n],[-1]),memberships[n])

			print "Individual NMI:", np.mean(individual_nmi)
			

			trueMemberships_stacked = np.reshape(np.hstack(groundTruth),[-1])
			memberships_stacked = np.hstack(memberships)
			overall_nmi = nmi(trueMemberships_stacked,memberships_stacked)

			print "Overall NMI:", overall_nmi

	theta = estimate_theta(X,graphs)
	order = np.argsort(-1*np.diag(theta),)
	theta =  theta[order,][:,order]

	W = W[order,]

	memberships = []
	for n in range(n_graphs):
		X[n] = X[n][:,order]
		memberships.append(dict(zip(range(1,len(X[n])+1),np.argmax(X[n],1))))

	memberships = dict(zip(graphNames,memberships))
	if (groundTruth!=None):
		return memberships, W, theta, dict(zip(graphNames,individual_nmi)), overall_nmi
	else:
		return memberships, W, theta

