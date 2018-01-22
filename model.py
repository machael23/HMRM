import os
import sys
import time

import numpy as np
from numpy import linalg as LA, argpartition
from scipy import sparse

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class model(BaseEstimator, TransformerMixin):
    def __init__(self, n_embeddings=50, K=20, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=8, random_state=None,
                 save_params=False, save_dir='.', verbose=False, **kwargs):
        '''
        Parameters
        ---------
        n_embeddings : int
            Dimensionality of embeddings
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        dtype: str or type
            Data-type for the parameters, default 'float32' (np.float32)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.n_embeddings = n_embeddings
        self.n_activities = K
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.set_state(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters
        Parameters
        ---------
        lambda: float
            Regularization parameter.
        '''
        self.lam = float(kwargs.get('lam', 1e-7))
        self.lam_1 = float(kwargs.get('lam_1', 1e-2))
        self.lam_2 = float(kwargs.get('lam_2', 1e-2))
        self.lam_3 = float(kwargs.get('lam_3', 1e-2))
        self.lam_i = float(kwargs.get('lam_i', 0.5))
        self.lam_t = float(kwargs.get('lam_t', 0.5))

    def _init_params(self, n_users, n_pois, n_times):
        ''' Initialize all the latent factors and biases '''
        self.theta = self.init_std * np.random.randn(n_users, self.n_activities).astype(self.dtype) + 1
        self.ai = self.init_std * np.random.randn(n_pois, self.n_activities).astype(self.dtype) + 1
        self.at = self.init_std * np.random.randn(n_times, self.n_activities).astype(self.dtype) + 1
        self.ei = self.init_std * np.random.randn(n_pois, self.n_embeddings).astype(self.dtype)
        self.ec = self.init_std * np.random.randn(n_pois, self.n_embeddings).astype(self.dtype)
        self.ea = self.init_std * np.random.randn(self.n_activities, self.n_embeddings).astype(self.dtype)
        self.et = self.init_std * np.random.randn(n_times, self.n_embeddings).astype(self.dtype)
        assert np.all(self.theta > 0)
        assert np.all(self.ai > 0)
        assert np.all(self.at > 0)

    def fit(self, UI, UT, II, IT, poi_pt):
        '''Fit the model to the data in U.

        Parameters
        ----------
        UI : scipy.sparse.csr_matrix, shape (n_users, n_pois)
            Training click matrix.

        UT : matrix, shape(n_users, n_times)

        II : scipy.sparse.csr_matrix, shape (n_pois, n_pois)
            Training co-occurrence matrix.
        IT : matrix shape (n_pois, n_times)

        voca_pt : vocabulary file pointer string
        '''
        self._read_poi(poi_pt)
        n_users, n_pois = UI.shape
        n_times = UT.shape[1]
        assert II.shape == (n_pois, n_pois)
        assert IT.shape == (n_pois, n_times)

        self._init_params(n_users, n_pois, n_times)
        self._update(UI, UT, II, IT)
        return self

    def _update(self, UI, UT, II, IT):
        UIT = UI.T.tocsr()  # pre-compute this
        IIT = II.T.tocsr()
        UTT = UT.T.tocsr()
        ITT = IT.T.tocsr()
        for i in range(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            # time1=time.time()
            self._update_factors(UI, UIT, UT, UTT, II, IIT, IT, ITT)
            # time2=time.time()
            # print(time2-time1)
            # loss_1_i = 0
            # U_row, U_column = UI.shape
            # for i in range(U_row):
            #     taij = UI[i, :] - np.dot(self.theta[i], self.ai.T)
            #     loss_1_i = loss_1_i + np.dot(taij, taij.T)
            # loss_1_i = self.lam_i * np.sqrt(loss_1_i)
            # # print('loss_1_i')
            #
            #
            # utta = UT-np.dot(self.theta,self.at.T)
            # loss_1_t = self.lam_t*np.sqrt(np.sum(np.multiply(utta, utta)))
            # # print('loss_1_t')
            #
            # loss_1 = self.lam_1*(loss_1_i+loss_1_t)
            #
            # loss_2_i = 0
            # I_row, I_column = II.shape
            # start_idx = list(range(0, I_row, 1000))
            # end_idx = start_idx[1:] + [I_row]
            #
            # for i in range(start_idx.__len__()):
            #     icij = II[start_idx[i]:end_idx[i], :] - np.dot(self.ei[start_idx[i]:end_idx[i], :], self.ec.T)
            #     loss_2_i = loss_2_i + np.sum(np.multiply(icij, icij))
            # loss_2_i = self.lam_i * np.sqrt(loss_2_i)
            # # print('loss_2_i')
            #
            # itit = IT-np.dot(self.ei,self.et.T)
            # loss_2_t = self.lam_t * np.sqrt(np.sum(np.multiply(itit, itit)))
            # # print('loss_2_t')
            #
            # loss_2 = self.lam_2 * (loss_2_i + loss_2_t)
            #
            # l3_i = self.ai - np.dot(self.ec, self.ea.T)
            # loss_3_i = self.lam_i * np.sqrt(np.sum(np.multiply(l3_i, l3_i)))
            # # print('loss_3_i')
            # l3_t = self.at - np.dot(self.et, self.ea.T)
            # loss_3_t = self.lam_t * np.sqrt(np.sum(np.multiply(l3_t, l3_t)))
            # # print('loss_3_t')
            #
            # loss_3 = self.lam_3 * (loss_3_i + loss_3_t)
            #
            # loss_ai = self.lam * np.sqrt(np.sum(np.multiply(self.ai, self.ai)))
            # # print('loss_ai')
            #
            # loss_at = self.lam * np.sqrt(np.sum(np.multiply(self.at, self.at)))
            # # print('loss_at')
            #
            # loss_theta = self.lam * np.sqrt(np.sum(np.multiply(self.theta, self.theta)))
            # # print('loss_theta')
            #
            # loss_ea = self.lam * np.sqrt(np.sum(np.multiply(self.ea, self.ea)))
            # # print('loss_ea')
            #
            # loss_ei = self.lam * np.sqrt(np.sum(np.multiply(self.ei, self.ei)))
            # # print('loss_ei')
            #
            # loss_ec = self.lam * np.sqrt(np.sum(np.multiply(self.ec, self.ec)))
            # # print('loss_ec')
            #
            # loss_et = self.lam * np.sqrt(np.sum(np.multiply(self.et, self.et)))
            # # print('loss_et')
            #
            # loss = loss_1 + loss_2 + loss_3 + loss_ai + loss_at + loss_theta +loss_ea + loss_ec + loss_ei  +loss_et
            # # print('loss')
            # print('%.4f' % loss[0, 0])

            if self.save_params:
                self._save_params(i)

    def _update_factors(self, UI, UIT, UT, UTT, II, IIT, IT, ITT):
        # if self.verbose:
        #     start_t = _writeline_and_time('\tUpdating theta...')
        self.theta = update_theta(UI,  UT, self.ai, self.at, self.lam_i, self.lam_t, self.lam_1, self.lam, self.n_jobs,
                                  self.batch_size)
        # if self.verbose:
        #     print('\r\tUpdating theta: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating ai...')

        self.ai = update_ai(UIT, self.theta, self.ea, self.ec, self.lam_i, self.lam_1, self.lam_3, self.lam,
                            self.n_jobs, self.batch_size)
        self.ai = _normalize_activity(self.ai)
        # if self.verbose:
        #     print('\r\tUpdating ai: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating at...')

        self.at = update_at(UTT, self.theta, self.ea, self.et, self.lam_t, self.lam_1, self.lam_3,
                            self.lam, self.n_jobs, self.batch_size)
        self.at = _normalize_activity(self.at)
        # if self.verbose:
        #     print('\r\tUpdating at: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating ec embeddings...')

        self.ec = update_ec(IIT, self.ea, self.ei, self.ai, self.lam_i, self.lam_2, self.lam_3, self.lam, self.n_jobs,
                            self.batch_size)
        # if self.verbose:
        #     print('\r\tUpdating ec embeddings: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating ei embeddings...')

        self.ei = update_ei(II, IT, self.ec, self.et, self.lam_i, self.lam_t, self.lam_2, self.lam, self.n_jobs,
                            self.batch_size)
        # if self.verbose:
        #     print('\r\tUpdating ei embeddings: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating ea embeddings...')

        self.ea = update_ea(self.ec, self.et, self.ai.T, self.at.T, self.lam_i, self.lam_t, self.lam_3, self.lam, self.n_jobs, self.batch_size)
        # if self.verbose:
        #     print('\r\tUpdating ea embeddings: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating et embeddings...')

        self.et = update_et(ITT, self.ei, self.ea, self.at, self.lam_t, self.lam_2, self.lam_3, self.lam, self.n_jobs, self.batch_size)
        # if self.verbose:
        #     print('\r\tUpdating et embeddings: time=%.2f' % (time.time() - start_t))

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'Embeddings_K%d_iter%d.npz' % (self.n_embeddings, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta, V=self.ai, C=self.ec, B=self.ei, A=self.ea)


    def _read_poi(self, poi_pt):
        self.l2id = {}
        self.id2l = {}
        for l in open(poi_pt):
            poiindex = l.strip().split()
            self.l2id[poiindex[1]] = int(poiindex[0])
            self.id2l[int(poiindex[0])] = poiindex[1]

# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def update_theta(UI, UT, ai, at, lam_i, lam_t, lam_1, lam, n_jobs, batch_size):
    '''Update user latent factors'''
    m, n = UI.shape  # m: number of users, n: number of pois
    f = ai.shape[1]  # f: number of latent factors
    left = lam_1 * lam_i * np.dot(ai.T, ai) + lam_1 * lam_t * np.dot(at.T, at) + lam * np.eye(f, dtype=ai.dtype)

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs)(delayed(_solve_theta)(lo, hi, UI, UT, ai, at, left, f, lam_i, lam_t, lam_1) for lo, hi in
                           zip(start_idx, end_idx))
    theta = np.vstack(res)
    return theta


def _solve_theta(lo, hi, UI, UT, ai, at, left, f, lam_i, lam_t, lam_1):
    theta_batch = np.empty((hi - lo, f), dtype=ai.dtype)
    for ib, u in enumerate(range(lo, hi)):
        x_u, idx_u = get_row(UI, u)
        t_u, idt_u = get_row(UT, u)
        res = lam_1 * lam_i * np.dot(x_u, ai[idx_u]) + lam_1 * lam_t * np.dot(t_u, at[idt_u])
        theta_batch[ib] = LA.solve(left, res)
    theta_batch = theta_batch.clip(0)
    return theta_batch


def update_ai(UIT, theta, ea, ec, lam_i, lam_1, lam_3, lam, n_jobs, batch_size):
    m, n = UIT.shape  # m: number of pois, n: number of users
    f = theta.shape[1]  # f: number of factors
    left = lam_1 * lam_i * np.dot(theta.T, theta) + (lam_3 * lam_i + lam) * np.eye(f, dtype=theta.dtype)

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs)(delayed(_solve_ai)(lo, hi, UIT, theta, ea, ec, left, f, lam_i, lam_1, lam_3) for lo, hi in
                           zip(start_idx, end_idx))
    ai = np.vstack(res)
    return ai


def _solve_ai(lo, hi, UIT, theta, ea, ec, left, f, lam_i, lam_1, lam_3):
    ai_batch = np.empty((hi - lo, f), dtype=theta.dtype)
    for ib, u in enumerate(range(lo, hi)):
        x_u, idx_u = get_row(UIT, u)
        res = lam_1 * lam_i * np.dot(x_u, theta[idx_u]) + lam_3 * lam_i * np.dot(ec[u], ea.T)
        ai_batch[ib] = LA.solve(left, res)
    ai_batch = ai_batch.clip(0)
    return ai_batch


def update_at(UTT, theta, ea, et, lam_t, lam_1, lam_3, lam, n_jobs, batch_size):
    m, n = UTT.shape  # m: number of time slots, n: number of users
    f = theta.shape[1]  # f: number of factors
    left = lam_1 * lam_t * np.dot(theta.T, theta) + (lam_3 * lam_t + lam) * np.eye(f, dtype=theta.dtype)

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs)(delayed(_solve_at)(lo, hi, UTT, theta, ea, et, left, f, lam_t, lam_1, lam_3) for lo, hi in
                           zip(start_idx, end_idx))
    ai = np.vstack(res)
    return ai


def _solve_at(lo, hi, UTT, theta, ea, et, left, f, lam_t, lam_1, lam_3):
    at_batch = np.empty((hi - lo, f), dtype=theta.dtype)
    for ib, u in enumerate(range(lo, hi)):
        x_u, idx_u = get_row(UTT, u)
        res = lam_1 * lam_t * np.dot(x_u, theta[idx_u]) + lam_3 * lam_t * np.dot(et[u], ea.T)
        at_batch[ib] = LA.solve(left, res)
    at_batch = at_batch.clip(0)
    return at_batch


def update_ec(IIT, ea, ei, ai, lam_i, lam_2, lam_3, lam, n_jobs, batch_size):
    n, m = ai.shape  # number of pois
    f = ea.shape[1]  # number of embeddings
    assert ea.shape[0] == m
    assert ei.shape == (n, f)
    left = lam_2 * lam_i * np.dot(ei.T, ei) + lam_3 * lam_i * np.dot(ea.T, ea) + lam * np.eye(f, dtype=ei.dtype)

    start_idx = list(range(0, n, batch_size))
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs)(
        delayed(_solve_ec)(lo, hi, IIT, ea, ei, ai, left, f, lam_i, lam_2, lam_3) for lo, hi in zip(start_idx, end_idx))
    ec = np.vstack(res)
    return ec


def _solve_ec(lo, hi, IIT, ea, ei, ai, left, f, lam_i, lam_2, lam_3):
    ec_batch = np.empty((hi - lo, f), dtype=ea.dtype)
    for ib, i in enumerate(range(lo, hi)):
        m_i, idx_m_i = get_row(IIT, i)
        res = lam_3 * lam_i * np.dot(ai[i, :], ea) + lam_2 * lam_i * np.dot(m_i, ei[idx_m_i])
        ec_batch[ib] = LA.solve(left, res)
    return ec_batch


def update_ei(II, IT, ec, et, lam_i, lam_t, lam_2, lam, n_jobs, batch_size):
    n, f = ec.shape  # n: number of pois, f: number of embeddings
    left = lam_2 * np.dot(ec.T, ec) + lam * np.eye(f, dtype=ec.dtype)

    start_idx = list(range(0, n, batch_size))
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs)(delayed(_solve_ei)(lo, hi, II, IT, ec, et, left, f, lam_i, lam_t, lam_2) for lo, hi in zip(start_idx, end_idx))
    ei = np.vstack(res)
    return ei


def _solve_ei(lo, hi, II, IT, ec, et, left, f, lam_i, lam_t, lam_2):
    ei_batch = np.empty((hi - lo, f), dtype=ec.dtype)
    for ib, j in enumerate(range(lo, hi)):
        x_j, idx_j = get_row(II, j)
        m_j, idm_j = get_row(IT, j)
        res = lam_2 * lam_i * np.dot(x_j, ec[idx_j])+lam_2 * lam_t * np.dot(m_j, et[idm_j])
        ei_batch[ib] = LA.solve(left, res)
    return ei_batch


def update_ea(ec, et, aiT, atT, lam_i, lam_t, lam_3, lam, n_jobs, batch_size):
    n, f = ec.shape  # n: number of pois, f: number of embeddings
    k = aiT.shape[0]  # number of activities
    left = lam_3 * lam_i * np.dot(ec.T, ec) + lam_3 * lam_t * np.dot(et.T, et) + lam * np.eye(f, dtype=ec.dtype)

    start_idx = list(range(0, k, batch_size))
    end_idx = start_idx[1:] + [k]
    res = Parallel(n_jobs)(delayed(_solve_ea)(lo, hi, ec, et, aiT, atT, left, f, lam_i, lam_t, lam_3) for lo, hi in zip(start_idx, end_idx))
    ea = np.vstack(res)
    return ea


def _solve_ea(lo, hi, ec, et, aiT, atT, left, f, lam_i, lam_t, lam_3):
    ea_batch = np.empty((hi - lo, f), dtype=ec.dtype)
    for ib, j in enumerate(range(lo, hi)):
        res = lam_3 * lam_i * np.dot(aiT[j, :], ec) + lam_3 * lam_t * np.dot(atT[j, :], et)
        ea_batch[ib] = LA.solve(left, res)
    return ea_batch


def update_et(ITT, ei, ea, at, lam_t, lam_2, lam_3, lam, n_jobs, batch_size):
    n, m = at.shape  # number of time slots, number of activities
    f = ea.shape[1]  # number of embeddings
    assert ea.shape == (m,f)
    left = lam_2 * lam_t * np.dot(ei.T, ei) + lam_3 * lam_t * np.dot(ea.T, ea) + lam * np.eye(f, dtype=ei.dtype)

    start_idx = list(range(0, n, batch_size))
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs)(
        delayed(_solve_et)(lo, hi, ITT, ei, ea, at, left, f, lam_t, lam_2, lam_3) for lo, hi in zip(start_idx, end_idx))
    ec = np.vstack(res)
    return ec


def _solve_et(lo, hi, ITT, ei, ea, at, left, f, lam_t, lam_2, lam_3):
    et_batch = np.empty((hi - lo, f), dtype=ea.dtype)
    for ib, i in enumerate(range(lo, hi)):
        m_i, idx_i = get_row(ITT, i)
        res = lam_3 * lam_t * np.dot(at[i, :], ea) + lam_2 * lam_t * np.dot(m_i, ei[idx_i])
        et_batch[ib] = LA.solve(left, res)
    return et_batch


def _normalize_activity(ai):
    norms = np.sum(ai, axis=0)
    normactivity = ai / norms
    return normactivity
