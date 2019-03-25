import numpy as np 
from numpy.linalg import inv, cholesky, eig
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans

from joblib import Parallel, delayed
import multiprocessing



def inv_s(M):
	diag_m = 1/np.diag(M)
	diag_m[diag_m==float('inf')] = 0
	return(np.diag(diag_m))

def get_laplacian(A):
	# D = np.diag(A.sum(axis = 1))
	# L = D - A
	D_2 = inv_s(np.diag(A.sum(axis = 1)))**(.5)
	L = np.eye(A.shape[0]) - np.matmul(np.matmul(D_2,A),D_2)
	return L

def get_Qn(adj,K, laplacian = False):
	if laplacian:
		eig_decomp = eig(get_laplacian(adj))
		args = np.argsort(eig_decomp[0])[:K]
		U = (eig_decomp[1][:,args])
		return U
	else:
		eig_decomp = eig(adj)
		args = np.argsort(-abs(eig_decomp[0]),)[:K]
		U = (eig_decomp[1][:,args])
		return abs(U)

def onehot_vec(arg,K):
	ans = np.zeros([K])
	ans[arg] = 1.
	return ans

def frobenius_norm(M):
	return np.sqrt(np.trace(np.matmul(M.T,M)))

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


def update_xni(q,W):
	K = W.shape[0]
	dif = W - q

	dist = np.diag(np.matmul(dif,dif.T))

	x = onehot_vec(np.argmin(dist),K)
	return x

def processLoop(Qn,W,Xn):
	for i in range(Xn.shape[0]):
 		Xn[i,] = update_xni(Qn[i,],W)
 	return Xn

def realign_type1(X,An):
	memberships = []
	for n in range(len(An)):
		An_c, N_c  = get_collapseds_spec(X[n],An[n])
		theta_n = np.zeros(N_c.shape)
		theta_n[N_c > 0.] = An_c[N_c > 0.]*1./N_c[N_c > 0.]
		order = np.argsort(-1*np.diag(theta_n),)
		X[n] = X[n][:,order]
		memberships.append(np.argmax(X[n],1))	
	return X, memberships

def realign_type2(X,Q,W):
	memberships = []
	for n in range(len(X)):
		X[n] = processLoop(Q[n],W,X[n])
		memberships.append(np.argmax(X[n],1))
	return X, memberships

def getperm(X1,X2):
	K = X1.shape[0]
	perm_matrix = np.zeros(X1.shape)
	for i in range(X1.shape[0]):
		dif = (X1[i,:] -  X2)
		dist =  np.diag(np.matmul(dif,dif.T))
		perm_matrix[i,:] = onehot_vec(np.argmin(dist),K)

	return perm_matrix

# def realign_type3(X,An):
# 	memberships = []
# 	for n in range(len(X)):
# 		An_c, N_c  = get_collapseds_spec(X[n],An[n])
# 		theta_n = np.zeros(N_c.shape)
# 		theta_n[N_c > 0.] = An_c[N_c > 0.]*1./N_c[N_c > 0.]
# 		if n>0:
# 			perm = getperm(theta_n1,theta_n)
# 			X[n] = np.matmul(X[n],perm)
# 			theta_n = np.matmul(np.matmul(perm.T,theta_n),perm)
# 		theta_n1 = theta_n
# 		print theta_n
# 		memberships.append(np.argmax(X[n],1))
# 	return X, memberships	

def realign_type3(X,Wn):
	memberships = [np.argmax(X[0],1)]
	for n in range(1,len(X)):
		perm = getperm(Wn[n-1],Wn[n])
		X[n] = np.matmul(X[n],perm)
		Wn[n] = np.matmul(Wn[n].T,perm).T
		memberships.append(np.argmax(X[n],1))
	return X, memberships		


def mcr(x,y):
	cm = contingency_matrix(x,y)
	return (cm.max(axis = 0).sum())*1./cm.sum()

def compute_nmi(groundTruth,memberships, printNMI = True, ARI = False, MCR = False):
	n_graphs = len(memberships)
	individual_nmi = np.ndarray([n_graphs])
	individual_ari = np.ndarray([n_graphs])
	individual_mcr = np.ndarray([n_graphs])
	for n in range(n_graphs):
		individual_nmi[n] = nmi(np.reshape(groundTruth[n],[-1]),memberships[n])
		if ARI:
			individual_ari[n] = adjusted_rand_score(np.reshape(groundTruth[n],[-1]),memberships[n])
		if MCR:
			individual_mcr[n] = mcr(np.reshape(groundTruth[n],[-1]),memberships[n])


	trueMemberships_stacked = np.reshape(np.hstack(groundTruth),[-1])
	memberships_stacked = np.hstack(memberships)
	overall_nmi = nmi(trueMemberships_stacked,memberships_stacked)
	if ARI:
		overall_ari = adjusted_rand_score(trueMemberships_stacked,memberships_stacked)
	else:
		overall_ari = -1
	if MCR:
		overall_mcr = mcr(trueMemberships_stacked,memberships_stacked)
	else:
		overall_mcr = -1

	return individual_nmi, overall_nmi, individual_ari, overall_ari, individual_mcr, overall_mcr


def iso_spec(graphs, K, tol = 1e-15, useLaplacian = False, groundTruth = None, maxIter = 200, init = 'kmeans++', initX = None, parallel = False, n_cores = -1,seed = 242, printNMI=True, ARI = False, MCR = False):
	if (type(graphs)!=dict):
		graphNames = ["Graph"+str(g) for g in range(1,len(graphs)+1)]
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
	Wn = np.zeros([n_graphs,K,K])
	# memberships = []

	for i in range(n_graphs):
		Qn = get_Qn(graphs[i],K, laplacian = useLaplacian)
		Q.append(Qn)
		if init=='kmeans++':
			km = KMeans(n_clusters=K, random_state=seed).fit(Qn)		
			Wn[i,:,:] = km.cluster_centers_
			Xn = np.vstack([onehot_vec(rr,K) for rr in km.labels_])
		else:
			Xn = np.vstack([onehot_vec(rr-1,K) for rr in initX[graphNames[i]]['Xn']])
			Wn[i,:,:] = np.matmul(np.matmul(inv_s(np.matmul(Xn.T,Xn)),Xn.T),Qn)
		X.append(Xn)
		# memberships.append(km.labels_)

	# Re-alignment steps
	
	# (1) rank within prob (2) re order accord
	X1,memberships1 = realign_type1(X,graphs)		

	# (1) cluster the centers (2) re-assign nodes to clusters based on the center of the centers
	W = KMeans(n_clusters=K,random_state=seed).fit(np.reshape(Wn,[-1,K])).cluster_centers_
	X2,memberships2 = realign_type2(X,Q,W)		

	# (1) search over all permutations
	# X3,memberships3 = realign_type3(X,graphs)		
	X3,memberships3 = realign_type3(X,Wn)

	X = X1
	memberships = memberships1

	if (groundTruth!=None):

		# print "type 1 performance"
		ind1, ov1,ari1,ari_ov1,mcr1, mcr_ov1 = compute_nmi(groundTruth,memberships1,printNMI = printNMI, ARI = ARI, MCR = MCR)

		# print "type 2 performance"
		ind2, ov2,ari2,ari_ov2,mcr2, mcr_ov2 = compute_nmi(groundTruth,memberships2,printNMI = printNMI, ARI = ARI, MCR = MCR)

		# print "type 3 performance"
		ind3, ov3,ari3,ari_ov3,mcr3, mcr_ov3 = compute_nmi(groundTruth,memberships3,printNMI = printNMI, ARI = ARI, MCR = MCR)

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
		return memberships, W, theta, ind1, ov1, ind2, ov2, ind3, ov3,ari1,ari_ov1,mcr1, mcr_ov1,ari2,ari_ov2,mcr2, mcr_ov2,ari3,ari_ov3,mcr3, mcr_ov3
	else:
		return memberships, W, theta

