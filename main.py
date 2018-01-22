import os

import numpy as np
import pandas as pd
from scipy import sparse

import model

DATA_DIR = 'C:/Users/esouser/Desktop/POITrajectory/github/Newyork/'
poi_pt = DATA_DIR+'tempData/POI.csv'
uimatrix_pt = DATA_DIR + 'tempData/userPOI_matrix.csv'
utmatrix_pt = DATA_DIR + 'tempData/userTime_matrix.csv'
itmatrix_pt = DATA_DIR + 'tempData/POItime_matrix.csv'

#new york
n_users = 1083
n_pois = 38333

#tokyo
# n_users = 2293
# n_pois = 61858
categotyIdNameDic={}
poiIdNameDic={}

if __name__ == '__main__':

    def termfrequency(D):
        tf = D.toarray()
        [row,col]=tf.shape
        rowSum = np.sum(tf, axis=1)
        for i in range(row):
            for j in range(col):
                tf[i][j] = tf[i][j]/rowSum[i]
                #print tf[i][j]
        return sparse.csr_matrix(tf)

    #load matrix user-poi
    ui = pd.read_csv(uimatrix_pt)
    rows, cols = np.array(ui['user_id']), np.array(ui['poi_id'])
    matrixUI = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32')
    matrixUI = termfrequency(matrixUI)

    ut = pd.read_csv(utmatrix_pt)
    rows, cols = np.array(ut['user_id']), np.array(ut['time_id'])
    matrixUT = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32')
    matrixUT = termfrequency(matrixUT)

    it = pd.read_csv(itmatrix_pt)
    rows, cols = np.array(it['poi_id']), np.array(it['time_id'])
    matrixIT = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32')
    matrixIT = termfrequency(matrixIT)

    #load matrix poi-poi
    data = np.load(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_data.npy'))
    indices = np.load(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_indices.npy'))
    indptr = np.load(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_indptr.npy'))
    matrixII = sparse.csr_matrix((data, indices, indptr), shape=(n_pois, n_pois))
    #see the sparseness
    print(matrixUI.shape, matrixUT.shape, matrixII.shape, matrixIT.shape)
    #print(float(matrixUI.nnz) / np.prod(matrixUI.shape))
    #print(float(matrixII.nnz) / np.prod(matrixII.shape))

    def get_row(Y, i):
        lo, hi = Y.indptr[i], Y.indptr[i + 1]
        return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]

    def SPPMI(matrixII):
        count_row = np.asarray(matrixII.sum(axis=1)).ravel()
        count_column = np.asarray(matrixII.sum(axis=0)).ravel()
        n_pairs = matrixII.data.sum()
        n_row = matrixII.shape[0]
        #constructing the SPPMI matrix
        MII = matrixII.copy()
        for i in range(n_row):
            lo, hi, d, idx = get_row(MII, i)
            MII.data[lo:hi] = np.log(d * n_pairs / (count_row[i] * count_column[idx]))

        MII.data[MII.data < 0] = 0
        MII.eliminate_zeros()
        k_ns = 1
        MII_ns = MII.copy()
        if k_ns > 1:
            offset = np.log(k_ns)
        else:
            offset = 0.

        MII_ns.data -= offset
        MII_ns.data[MII_ns.data < 0] = 0
        MII_ns.eliminate_zeros()
        return MII_ns


    MII_ns = SPPMI(matrixII)
    MIT_ns = SPPMI(matrixIT)


    #start training
    # n_embeddings = [10,20,30,40,50,100,150,200,250,300]
    n_embeddings = [50]
    max_iter = 10
    K = [10]
    # K = [5,10,15,20,25,30,35,40,45,50]
    lam = 1e-7
    lam_1 = 1e-2
    lam_2 = 1e-2
    lam_3 = 1e-2
    lam_i=0.5
    lam_t=0.5
    for index1 in range(n_embeddings.__len__()):
        for index in range(K.__len__()):
            save_dir = os.path.join(DATA_DIR, 'results_parallel')
            hmrm = model.model(n_embeddings=n_embeddings[index1], K=K[index], max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=8,
                                              random_state=98765, save_params=True, save_dir=save_dir, verbose=True,
                                              lam=lam, lam_1=lam_1, lam_2=lam_2, lam_3=lam_3, lam_i=lam_i, lam_t = lam_t)
            hmrm.fit(matrixUI, matrixUT, MII_ns, MIT_ns, poi_pt)



            userfile = DATA_DIR + 'model/userActivity' + str(K[index]) + '-' + str(
                n_embeddings[index1])+ '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(userfile, hmrm.theta)

            aifile = DATA_DIR + 'model/activityPoi' + str(K[index]) + '-' + str(
                n_embeddings[index1]) + '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(aifile, hmrm.ai)

            atfile = DATA_DIR + 'model/activityTime' + str(K[index]) + '-' + str(
                n_embeddings[index1]) + '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(atfile, hmrm.at)

            poiembeddingfile = DATA_DIR + 'model/poiEmbedding' + str(K[index]) + '-' + str(
                n_embeddings[index1]) + '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(poiembeddingfile, hmrm.ei)

            etfile = DATA_DIR + 'model/timeEmbedding' + str(K[index]) + '-' + str(
                n_embeddings[index1]) + '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(etfile, hmrm.et)

            topicembeddingfile = DATA_DIR + 'model/activityEmbedding' + str(K[index]) + '-' + str(
                n_embeddings[index1]) + '-'+ str(lam_1) + '-'+ str(lam_2)+ '-'+ str(lam_3)+ '-'+ str(lam_i)+ '-'+ str(lam_t)+'.txt'
            np.savetxt(topicembeddingfile, hmrm.ea)