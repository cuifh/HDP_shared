"""function for NPL posterior bootstrap sampler for GMM

Parameters
----------
B_postsamples : int 
    Number of posterior samples to generate

alph_conc : float
    Concentration parameter for DP prior

T_trunc: int > 0 
    Number of prior pseudo-samples from DP base measure for truncated sampling

y: array
    Observed datapoints

N_data: int
    Number of data points

D_data: int
    Dimension of observables

K_clusters: int
    Number of clusters in GMM model

R_restarts: int 
    Number of random restarts per posterior bootstrap sample

tol: float
    Stopping criterion for weighted EM

max_iter: int
    Maximum number of iterations for weighted EM

init: function
    Returns initial parameters for random restart maximizations 

sampleprior: function
    To generate prior pseudo-samples for DP prior

postsamples: array
    Centering posterior samples for MDP-NPL
    
n_cores: int
    Number of cores Joblib can parallelize over; set to -1 to use all cores
"""


import numpy as np
import pandas as pd
import time
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from npl import maximise_gmm as mgmm


def bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init,sampleprior,postsamples= None,n_cores = -1):
    #Declare parameters
    eps_dirichlet=10**(-100)
    pi_bb_1 = np.zeros((B_postsamples,K_clusters))              #mixing weights 
    mu_bb_1 = np.zeros((B_postsamples,K_clusters,D_data))       #means
    sigma_bb_1 = np.zeros((B_postsamples,K_clusters,D_data))    #covariances 
    
    pi_bb_2 = np.zeros((B_postsamples,K_clusters))              #mixing weights 
    mu_bb_2 = np.zeros((B_postsamples,K_clusters,D_data))       #means
    sigma_bb_2 = np.zeros((B_postsamples,K_clusters,D_data))    #covariances 
  

    #Generate prior pseudo-samples and concatenate y_tots
    alpha_top_layer=1
    if alph_conc!=0: #gamma
        alphas = np.concatenate((np.ones(N_data), (alph_conc/T_trunc)*np.ones(T_trunc)))
        beta_weights = np.random.dirichlet(alphas,B_postsamples) 
        y_prior = sampleprior(D_data,T_trunc,K_clusters,B_postsamples, postsamples)
        alpha_top_layer_beta=alpha_top_layer*beta_weights
        Weights_1=np.repeat([np.concatenate((np.ones(int(N_data/2)), np.zeros(int(N_data/2+T_trunc))))],B_postsamples,axis=0)+alpha_top_layer_beta
        Weights_2=np.repeat([np.concatenate((np.zeros(int(N_data/2)),np.ones(int(N_data/2)), np.zeros(T_trunc)))],B_postsamples,axis=0)+alpha_top_layer_beta
    else:
        beta_weights = np.random.dirichlet(np.ones(N_data), B_postsamples)
        y_prior = np.zeros(B_postsamples)
        alpha_top_layer_beta=alpha_top_layer*beta_weights
        Weights_1=np.repeat([np.concatenate((np.ones(int(N_data/2)),\
                                             np.zeros(int(N_data/2))))],B_postsamples,axis=0)+alpha_top_layer_beta
        Weights_2=np.repeat([np.concatenate((np.zeros(int(N_data/2)),np.ones(int(N_data/2))))]\
                            ,B_postsamples,axis=0)+alpha_top_layer_beta
    
    weights_1=Weights_1
    weights_2=Weights_2
    
    for b in range(B_postsamples):
        weights_1[b,:]=np.random.dirichlet(Weights_1[b,:]+eps_dirichlet,1)
        weights_2[b,:]=np.random.dirichlet(Weights_2[b,:]+eps_dirichlet,1)
    
    
    
    
        

    #Initialize parameters randomly for RR-NPL
    pi_init,mu_init,sigma_init = init(R_restarts, K_clusters,B_postsamples, D_data)


    #Parallelize bootstrap
    if R_restarts == 0: #FI-NPL (with MLE initialization to select single mode) ###add [i] after y_prior 
        pi_init_mle,mu_init_mle,sigma_init_mle = mgmm.init_params(y,N_data,K_clusters,D_data,tol,max_iter)
        temp_1 = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise_mle)(y,y_prior[i],alph_conc,weights_1[i],pi_init_mle,mu_init_mle,sigma_init_mle,K_clusters,tol,max_iter,N_data) for i in tqdm(range(B_postsamples))) 
    else:
        temp_1 = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise)(y,y_prior[i],weights_1[i],\
            pi_init[i*R_restarts:(i+1)*R_restarts],mu_init[i*R_restarts:(i+1)*R_restarts],sigma_init[i*R_restarts:(i+1)*R_restarts],\
            alph_conc, T_trunc,K_clusters,tol,max_iter,R_restarts,N_data,D_data, postsamples = postsamples) for i in tqdm(range(B_postsamples)))
        
       
            
            
        
        
        
    if R_restarts == 0: #FI-NPL (with MLE initialization to select single mode) ###add [i] after y_prior 
        pi_init_mle,mu_init_mle,sigma_init_mle = mgmm.init_params(y,N_data,K_clusters,D_data,tol,max_iter)
        temp_2 = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise_mle)(y,y_prior[i],alph_conc,weights_2[i],pi_init_mle,mu_init_mle,sigma_init_mle,K_clusters,tol,max_iter,N_data) for i in tqdm(range(B_postsamples))) 
    else:
        temp_2 = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise)(y,y_prior[i],weights_2[i],\
            pi_init[i*R_restarts:(i+1)*R_restarts],mu_init[i*R_restarts:(i+1)*R_restarts],sigma_init[i*R_restarts:(i+1)*R_restarts],\
            alph_conc, T_trunc,K_clusters,tol,max_iter,R_restarts,N_data,D_data, postsamples = postsamples) for i in tqdm(range(B_postsamples)))
        
        
        
        
    
    #Convert to numpy array
    for i in range(B_postsamples):
        pi_bb_1[i] = temp_1[i][0]
        mu_bb_1[i] = temp_1[i][1]
        sigma_bb_1[i]= temp_1[i][2]
        pi_bb_2[i] = temp_2[i][0]
        mu_bb_2[i] = temp_2[i][1]
        sigma_bb_2[i]= temp_2[i][2]

    return pi_bb_1,mu_bb_1,sigma_bb_1,pi_bb_2,mu_bb_2,sigma_bb_2
