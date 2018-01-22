# translate word into id in documents
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import itertools
import os

poi2id = {}
DATA_DIR = 'C:/Users/esouser/Desktop/POITrajectory/github/Newyork/'

def indexFile(trajectory_pt, userpoiid_pt, uimatrix_pt, utmatrix_pt, itmatrix_pt, batch_size, window_size):
    print('index file: ', trajectory_pt)
    wf = open(userpoiid_pt, 'w')
    ui_matrix = open(uimatrix_pt, 'w')
    ui_matrix.writelines('user_id,poi_id')

    ut_matrix = open(utmatrix_pt, 'w')
    ut_matrix.writelines('user_id,time_id')

    it_matrix = open(itmatrix_pt, 'w')
    it_matrix.writelines('poi_id,time_id')

    saveid = 0
    count=0
    rows = []
    cols = []
    for l in open(trajectory_pt):
        trajectory = l.strip().split(',')
        userid = str(int(trajectory[0])-1)

        for tuple in trajectory[1:]:
            poi=tuple.split('#')[0]
            if poi not in poi2id:
                poi2id[poi] = len(poi2id)
            time = tuple.split('#')[1]
            ut_matrix.writelines('\n')
            ut_matrix.writelines(userid+','+time)

        wf.writelines(userid)


        for tuple in trajectory[1:]:
            poiid = str(poi2id[tuple.split('#')[0]])
            time = tuple.split('#')[1]
            wf.writelines(' ' + poiid)
            ui_matrix.writelines('\n')
            ui_matrix.writelines(userid + ',' + poiid)
            it_matrix.writelines('\n')
            it_matrix.writelines(poiid + ',' + time)

        wf.writelines('\n')

        poiids = [poi2id[tuple.split('#')[0]] for tuple in trajectory[1:]]
        count=count+1
        for ind_focus, poiid_focus in enumerate(poiids):
            ind_lo = max(0, ind_focus-window_size)
            ind_hi = min(len(poiids), ind_focus+window_size+1)

            for ind_c in range(ind_lo, ind_hi):
                if ind_c == ind_focus:
                    continue
                '''diagonals are zeros or not'''
                if poiid_focus == poiids[ind_c]:
                    continue
                rows.append(poiid_focus)
                cols.append(poiids[ind_c])
        if count%batch_size == 0 and count != 0:
            np.save(os.path.join(DATA_DIR, 'intermediate/coo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
            saveid = saveid + batch_size
            print('%dth user, %dth user' % (saveid, count))
            rows = []
            cols = []
    np.save(os.path.join(DATA_DIR, 'intermediate/coo_%d_%d.npy' % (saveid, count)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
    
    wf.close()
    ui_matrix.close()
    print('write file: ', userpoiid_pt)
    print('write file: ', uimatrix_pt)
    return count


def write_poi2id(res_pt):
    print('write:', res_pt)
    wf = open(res_pt, 'w')
    for poi, poiid in sorted(poi2id.items(), key=lambda d:d[1]):
        wf.writelines('%d\t%s' % (poiid, poi))
        wf.writelines('\n')
    wf.close()
    
def load_data(csv_file, shape):
    print('loading data')
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['user_id']), np.array(tp['poi_id'])
    data = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), dtype='float32').tocsr()
    return data

def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in range(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
        if u%1000 == 0:
            print('%dth user' % u)
    np.save(os.path.join(DATA_DIR, 'intermediate/coo_%d_%d.npy' % (lo, hi)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))

def _matrixw_batch(lo, hi, matW):
    coords = np.load(os.path.join(DATA_DIR, 'intermediate/coo_%d_%d.npy' % (lo, hi)))
    rows = coords[:, 0]
    cols = coords[:, 1]
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_pois, n_pois), dtype='float32').tocsr()
    matW = matW + tmp
    print("User %d to %d finished" % (lo, hi))
    sys.stdout.flush()
    return matW
        
if __name__ == '__main__':
    trajectory_pt = DATA_DIR+'trainingData/trajectory_NYC.csv'
    userpoiid_pt = DATA_DIR+'tempData/user_trajectory.csv'
    uimatrix_pt = DATA_DIR+'tempData/userPOI_matrix.csv'
    utmatrix_pt = DATA_DIR + 'tempData/userTime_matrix.csv'
    poi_pt = DATA_DIR+'tempData/POI.csv'
    itmatrix_pt = DATA_DIR+'tempData/POItime_matrix.csv'
    batch_size = 1000
    window_size = 5 #actually half window size
    n_users = indexFile(trajectory_pt, userpoiid_pt, uimatrix_pt, utmatrix_pt, itmatrix_pt, batch_size, window_size)
    n_pois = len(poi2id)
    print('n(user)=', n_users, 'n(poi)=', n_pois)
    write_poi2id(poi_pt)



    matrixUI = load_data(uimatrix_pt, (n_users, n_pois))
    start_idx = list(range(0, n_users, batch_size))
    end_idx = start_idx[1:] + [n_users]
    #for lo, hi in zip(start_idx, end_idx):
        #_coord_batch(lo, hi, matrixD)
        
    matrixII = sparse.csr_matrix((n_pois, n_pois), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        matrixII = _matrixw_batch(lo, hi, matrixII)
        print(float(matrixII.nnz) / np.prod(matrixII.shape))
    
    np.save(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_data.npy'), matrixII.data)
    np.save(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_indices.npy'), matrixII.indices)
    np.save(os.path.join(DATA_DIR, 'matrix/coordinate_co_binary_indptr.npy'), matrixII.indptr)

    
        
    print(matrixUI.shape, matrixII.shape)
    