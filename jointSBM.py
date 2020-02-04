import numpy as np 
import scipy as sp

from numpy.linalg import inv, cholesky
from scipy.linalg import eig
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
import random

import time
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import logging

from utils.utils import *



def update_W(X,Q,deltas_,K):
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

def update_xni(q,W,rt,fac,gamman,K):
    dif = W - q
    term2 = ((q*np.sqrt(fac) + q*1./np.sqrt(rt))**2).sum(axis = 1)
    dist = np.diag(np.matmul(dif,dif.T))*gamman + term2
    x = onehot_vec(np.argmin(dist),K)
    return x

def get_fac(Vn,V,K):
    x = np.eye(K)   
    fac = np.nan_to_num([[(Vn - x[i,:] + x[k,:])/(V - x[i,:] + x[k,:]) for k in range(K)] for i in range(K)])
    # fac[fac<0] = 0
    gamman = fac.sum(axis=2)
    return fac, gamman

def processLoop(Qn,W,Xn,V,deltas2_n,K):
    Vn = np.diag(deltas2_n)
    rt = np.sum(V)/np.sum(Vn)
    fac,gamman = get_fac(Vn,V,K)
    for i in range(Xn.shape[0]):
        xx = Xn[i,].argmax()
        Xn[i,] = update_xni(Qn[i,],W,rt,fac[xx],gamman[xx],K)
    return Xn

def mcr(x,y):
    cm = contingency_matrix(x,y)
    return (cm.max(axis = 0).sum())*1./cm.sum()

def get_Qn(adj,K):
    if sp.sparse.issparse(adj):
        eig_decomp = sp.sparse.linalg.eigs(adj,K)
    else:
        eig_decomp = eig(adj)
    args = np.argsort(-abs(eig_decomp[0]),)[:K]
    D = (eig_decomp[0][args])
    U = (eig_decomp[1][:,args])
    return abs(np.matmul(U,np.diag(D)))

def get_counts(X):
    delta2_n = np.array([np.diag(np.sum(X[i],axis=0)) for i in range(len(X))])
    delta2 = np.sum(delta2_n,axis=0)

    gamma_n = np.array([np.sum(inv(delta2)*delta2_n[i]) for i in range(len(X))])
    return delta2_n, delta2, gamma_n    

class jointSBM(object):
    def __init__(self, graphs, K, \
        edgelist = False,
        tol = 1e-5, groundTruth = None,
        init = 'kmeans++',  seed = 242, **kwargs):
        graphs = graphs.copy()
        if (type(graphs)!=dict):
            self.graphNames = ["graph"+str(g) for g in range(1,len(graphs)+1)]
            graphs = dict(zip(self.graphNames,graphs))
        else:   
            self.graphNames = [k for k in graphs.keys()]
        if (groundTruth!=None):
            if (type(groundTruth)==dict):
                self.groundTruth = [groundTruth[g] for g in self.graphNames]
                # self.groundTruth = [g for k,g in groundTruth.items()]


        if np.any([g.shape[0]!=g.shape[1] for k,g in graphs.items()]):
            print("Converting edgelists to sparse adjacency matrices...")
            edgelist = True

        if edgelist:
            self.idx2node = {}
            for gg in self.graphNames:
                graphs[gg], self.idx2node[gg] = edgelist2sparse(graphs[gg],**kwargs)
            # print("Converted edgelists to sparse adjacency matrices.")

        self.graphs = [graphs[g] for g in self.graphNames]
        n_graphs = len(graphs)


        self.n_graphs = n_graphs
        self.n_nodes_array = [self.graphs[i].shape[0] for i in range(n_graphs)]
        self.total_nodes = np.sum(self.n_nodes_array)

        self.K = K
        self.tol = tol 
        self.init = init 
        self.seed = seed
        self.data_prepared = False
        
    def prepare_data(self):
        Q = []
        X = []
        for i in tqdm(range(self.n_graphs)):
            Qn = get_Qn(self.graphs[i],self.K)*np.sqrt(self.total_nodes*1./self.n_nodes_array[i])
            Q.append(Qn)
            X.append(self.initX(Qn))

        self.Q = Q
        self.X = X
        self.deltas_ = get_counts(X)
        self.data_prepared = True

    def initX(self, Qn):
        K = self.K
        if self.init == 'kmeans++':
            km = KMeans(n_clusters=K,random_state=self.seed).fit(Qn).labels_
            Xn = np.vstack([onehot_vec(r,K) for r in km])
        else:
            random.seed(self.seed)
            Xn = np.random.multinomial(1,[1./K]*K,size = n_nodes)
        return Xn

    def fit(self, printLoss = False, maxIter = 200, parallel = False, n_cores = -1):
        self.maxIter = maxIter 
        self.parallel = parallel 
        self.n_cores = n_cores 
        if not self.data_prepared:
            self.prepare_data()
        X = self.X
        Q = self.Q
        deltas_ = self.deltas_
        K = self.K
        n_graphs = self.n_graphs
        n_nodes_array = self.n_nodes_array        
        Loss = [0]
        stopValue = 1
        iter = -1
        measures = {}
        while (stopValue > self.tol and iter < self.maxIter):
            t0 = time.time()
            iter = iter + 1
            W = update_W(X,Q,deltas_,K)

            V = np.diag(deltas_[1])
            memberships = []
            counts_memberships = np.zeros([1,K])
            if self.parallel:
                if n_cores == -1:
                    num_cores = multiprocessing.cpu_count()
                else: 
                    num_cores = self.n_cores
                    
                X = Parallel(n_jobs=num_cores)(delayed(processLoop)(Q[n],W,X[n],V,deltas_[0][n],K) for n in range(n_graphs))

                for x in X:
                    # memberships.append(np.argmax(x,1))
                    counts_memberships += np.sum(x,0)
            else:
                for n in range(n_graphs):
                    X[n] = processLoop(Q[n],W,X[n],V,deltas_[0][n],K)
                    counts_memberships += np.sum(X[n],0)
            
            if (np.sum(counts_memberships==0.)>0):
                iter = 1
                logging.warning("Restarting...")
                X = [np.random.multinomial(1,[1./K]*K,size = n_nodes_array[i]) for i in range(n_graphs)]

            memberships = [np.argmax(X[n],1) for n in range(len(X))]
            deltas_ = get_counts(X)
            
            loss = np.ndarray([n_graphs])
            for n in range(n_graphs):
                XW = np.matmul(X[n],W)
                rt = np.sum(deltas_[1])/np.sum(deltas_[0][n])
                term = np.sqrt(np.matmul(inv(deltas_[1]),deltas_[0][n])) + np.sqrt(np.eye(K)*1./rt)
                loss[n] = deltas_[2][n]*frobenius_norm(XW-Q[n])**2 + frobenius_norm(np.matmul(Q[n],term))**2
            t1 = time.time()
            Loss.append(np.sum(loss))
            if printLoss:
                print("Iter: {} | Loss: {}".format(iter, Loss[iter]))

            stopValue =  abs(Loss[iter] - Loss[iter-1])

            if (self.groundTruth!=None):
                measures[iter] = self.evalutate(memberships)
                measures[iter]['Time'] = t1-t0
            else:
                measures[iter] = {'Time':t1-t0}
        
        theta = estimateTheta(X,self.graphs)
        order = np.argsort(-1*np.diag(theta),)
        theta =  theta[order,][:,order]
        self.theta = theta

        self.W = W[order,]

        memberships = []
        for n in range(n_graphs):
            X[n] = X[n][:,order]
            memberships.append(dict(zip(range(1,len(X[n])+1),np.argmax(X[n],1))))

        memberships = dict(zip(self.graphNames,memberships))
        self.memberships = memberships
        self.X = X
        self.measures = measures
        self.iter = iter

        return memberships, theta, W, measures

    def evalutate(self, memberships):
        groundTruth = self.groundTruth
        n_graphs = self.n_graphs
        individual_nmi = np.zeros([n_graphs])
        individual_ari = np.zeros([n_graphs])
        individual_mcr = np.zeros([n_graphs])
        for n in range(n_graphs):
            # print(n)
            individual_nmi[n] = nmi(memberships[n],groundTruth[n])
            individual_ari[n] = ari(memberships[n],groundTruth[n])
            individual_mcr[n] = mcr(memberships[n],groundTruth[n])

        trueMemberships_stacked = np.reshape(np.hstack(groundTruth),[-1])
        memberships_stacked = np.hstack(memberships)
        overall_nmi = nmi(memberships_stacked,trueMemberships_stacked)
        overall_ari = ari(memberships_stacked,trueMemberships_stacked)
        overall_mcr = mcr(memberships_stacked,trueMemberships_stacked)
        
        return {"NMI" : {'nmi' : np.mean(individual_nmi),'overall_nmi' : overall_nmi},"ARI" : {'ari' : np.mean(individual_ari),'overall_ari' : overall_ari},"MCR" : {'mcr' : np.mean(individual_mcr),'overall_mcr' : overall_mcr}}

