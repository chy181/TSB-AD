import math
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import traceback

from .utils.slidingWindows import find_length_rank

def _align_score_length(score, target_len):
    score = np.asarray(score, dtype=float).reshape(-1)
    if score.shape[0] == target_len:
        return score
    if score.shape[0] == 0:
        return np.zeros(target_len, dtype=float)
    if score.shape[0] < target_len:
        pad_len = target_len - score.shape[0]
        return np.pad(score, (pad_len, 0), mode="edge")
    return score[:target_len]


def _minmax_normalize(score):
    score = np.asarray(score, dtype=float).reshape(-1)
    score_min = np.min(score)
    score_max = np.max(score)
    return (score - score_min) / (score_max - score_min + 1e-8)


Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS',
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'MMPAD', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS', 'TSPulse_ZS', 'Time_RCD', 'Time_RCD_1w', "Time_RCD_8000", "Time_RCD_5000", "Time_RCD_2000", "Time_RCD_1000"]
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'PatchTST',
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'TSPulse_FT', 'xLSTMAD', 'CHARM', 'MultiAdapter_FT', 'TimeRCD_MAFT', "TSPulse_MultiAdapter_FT"]

def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message


def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)

        # 关键：打印完整 traceback
        traceback.print_exc()

        return error_message

def run_FFT(data, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50, max_sign_change_distance=10):
    from .models.FFT import FFT
    clf = FFT(ifft_parameters=ifft_parameters, local_neighbor_window=local_neighbor_window, local_outlier_threshold=local_outlier_threshold, max_region_size=max_region_size, max_sign_change_distance=max_sign_change_distance)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from .models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Left_STAMPi(data_train, data):
    from .models.Left_STAMPi import Left_STAMPi
    clf = Left_STAMPi(n_init_train=len(data_train), window_size=100)
    clf.fit(data)
    score = clf.decision_function(data)
    return score.ravel()

def run_SAND(data_train, data_test, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data_test.squeeze(), online=True, overlaping_rate=int(1.5*slidingWindow), init_length=len(data_train), alpha=0.5, batch_size=max(5*(slidingWindow), int(0.1*len(data_test))))
    score = clf.decision_scores_
    return score.ravel()

def run_KShapeAD(data, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5*slidingWindow))
    score = clf.decision_scores_
    return score.ravel()

def run_Series2Graph(data, periodicity=1):
    from .models.Series2Graph import Series2Graph
    slidingWindow = find_length_rank(data, rank=periodicity)

    data = data.squeeze()
    s2g = Series2Graph(pattern_length=slidingWindow)
    s2g.fit(data)
    query_length = 2*slidingWindow
    s2g.score(query_length=query_length,dataset=data)

    score = s2g.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    return score.ravel()

def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCA(data, slidingWindow=100, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
    from .models.NormA import NORMA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1):
    from .models.MCD import MCD
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_MCD(data_train, data_test, support_fraction=None, slidingWindow=1, n_jobs=1):
    from .models.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from .models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from .models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from .models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_COPOD(data, n_jobs=1):
    from .models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from .models.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_COF(data, n_neighbors=30):
    from .models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EIF(data, n_trees=100):
    from .models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RobustPCA(data, max_iter=1000):
    from .models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_SR(data, periodicity=1):
    from .models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)

def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    from .models.AE import AutoEncoder
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):
    from .models.CNN import CNN
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008):
    from .models.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TranAD(data_train, data_test, win_size=10, lr=1e-3):
    from .models.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_PatchTST(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.PatchTST import PatchTST
    clf = PatchTST(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002):
    from .models.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_USAD(data_train, data_test, win_size=5, lr=1e-4):
    from .models.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Donut(data_train, data_test, win_size=120, lr=1e-4, batch_size=128):
    from .models.Donut import Donut
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4):
    from .models.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_FITS(data_train, data_test, win_size=100, lr=1e-3):
    from .models.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OFA(data_train, data_test, win_size=100, batch_size = 64):
    from .models.OFA import OFA
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Lag_Llama(data, win_size=96, batch_size=64):
    from .models.Lag_Llama import Lag_Llama
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Chronos(data, win_size=50, batch_size=64):
    from .models.Chronos import Chronos
    clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base', batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_TimesFM(data, win_size=96):
    from .models.TimesFM import TimesFM
    clf = TimesFM(win_size=win_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_ZS(data, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data.shape[1])

    # Zero shot
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Time_RCD(data,
                 win_size=15000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TSB-AD/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Time_RCD_1w(data,
                 win_size=10000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD-Adapter/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TSB-AD-Adapter/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Time_RCD_8000(data,
                 win_size=8000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD-Adapter/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TTSB-AD-Adapter/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Time_RCD_5000(data,
                 win_size=5000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD-Adapter/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TSB-AD-Adapter/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Time_RCD_2000(data,
                 win_size=2000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD-Adapter/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TSB-AD-Adapter/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Time_RCD_1000(data,
                 win_size=1000,
                 batch_size=64,
                 device=None,
                 checkpoint="/root/TSB-AD-Adapter/checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
                 model_id="/root/TSB-AD-Adapter/checkpoints/time-rcd",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_FT(data_train, data_test, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data_test.shape[1])

    # Finetune
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_M2N2(
        data_train, data_test, win_size=12, stride=12,
        batch_size=64, epochs=100, latent_dim=16,
        lr=1e-3, ttlr=1e-3, normalization='Detrend',
        gamma=0.99, th=0.9, valid_size=0.2, infer_mode='online'
    ):
    from .models.M2N2 import M2N2
    clf = M2N2(
        win_size=win_size, stride=stride,
        num_channels=data_test.shape[1],
        batch_size=batch_size, epochs=epochs,
        latent_dim=latent_dim,
        lr=lr, ttlr=ttlr,
        normalization=normalization,
        gamma=gamma, th=th, valid_size=valid_size,
        infer_mode=infer_mode
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_TSPulse_ZS(data,
                   model="ibm-granite/granite-timeseries-tspulse-r1",
                   win_size=96,
                   batch_size=256,
                   smoothing_window=8,
                   prediction_mode="time",
                   **kwargs,
                   ):
    from TSB_AD.models.TSPulse import TSPulsePipeline
    num_input_channels = data.shape[1]
    clf = TSPulsePipeline(
            model_path=model,
            num_input_channels=num_input_channels,
            batch_size=batch_size,
            aggr_win_size=win_size,
            smoothing_window=smoothing_window,
            prediction_mode=prediction_mode,
        )
    score = clf.decision_function(data)
    return score.ravel()

def run_TSPulse_FT(data_train,
                   data_test,
                   model="ibm-granite/granite-timeseries-tspulse-r1",
                   win_size=96,
                   batch_size=256,
                   smoothing_window=8,
                   prediction_mode="time",
                   decoder_mode="common_channels",
                   num_epochs=20,
                   freeze_backbone=False,
                   validation_fraction=0.2,
                   lr=1e-4,
                   **kwargs):
    from TSB_AD.models.TSPulse import TSPulsePipeline
    num_input_channels = data_train.shape[1]

    clf = TSPulsePipeline(
            model_path=model,
            num_input_channels=num_input_channels,
            batch_size=batch_size,
            aggr_win_size=win_size,
            smoothing_window=smoothing_window,
            prediction_mode=prediction_mode,
            finetune_decoder_mode=decoder_mode,
            finetune_validation=validation_fraction,
            finetune_freeze_backbone=freeze_backbone,
            finetune_epochs=num_epochs,
            finetune_lr=lr,
        )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_MultiAdapter_FT(
    data_train,
    data_test,
    win_size=64,
    stride=1,
    train_stride=1,
    batch_size=64,
    epochs=5,
    lr=1e-3,
    weight_decay=1e-5,
    validation_size=0.1,
    patch_len=8,
    d_model=64,
    n_heads=4,
    encoder_layers=2,
    dropout=0.1,
    scales="1,2,4",
    use_multiscale=True,
    use_prototype=True,
    use_wgad=True,
    use_self_imp=True,
    use_star=False,
    star_discrete_threshold=5,
    n_prototypes=16,
    prototype_temp=0.2,
    prototype_entropy_weight=1e-3,
    w_rec=1.0,
    w_proto=0.1,
    w_star=0.1,
    w_wgad=0.1,
    proto_alpha=0.5,
    wgad_alpha=0.5,
    star_alpha=0.5,
    self_imp_alpha=0.3,
    star_num_shared_experts=4,
    star_num_experts=8,
    star_top_k=4,
    star_dropout=0.1,
    star_sim_weight=0.1,
    wgad_graph_window_size=2,
    wgad_gcn_layers=1,
    wgad_wave_scales=2,
    wgad_lambda_wavelet=0.5,
    wgad_lambda_cl=0.1,
    wgad_graph_alpha=1.0,
    wgad_score_mode="product",
    wgad_affine=True,
    wgad_subtract_last=False,
    wgad_cl_temperature=1.0,
    self_imp_steps=10,
    self_imp_lr=0.05,
    self_imp_l1=1e-3,
    self_imp_tv=1e-3,
    self_imp_huber_delta=1.0,
    self_imp_hidden_dim=16,
    self_imp_raw_hidden_dim=16,
    self_imp_raw_weight=0.1,
    self_imp_seed=42,
    patience=3,
    cuda=True,
    checkpoint_path=None,
    return_score=True,
    **kwargs,
):
    from .models.MultiAdapter import MultiAdapter

    clf = MultiAdapter(
        input_c=data_test.shape[1],
        win_size=win_size,
        stride=stride,
        train_stride=train_stride,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        validation_size=validation_size,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        encoder_layers=encoder_layers,
        dropout=dropout,
        scales=scales,
        use_multiscale=use_multiscale,
        use_prototype=use_prototype,
        use_wgad=use_wgad,
        use_self_imp=use_self_imp,
        use_star=use_star,
        star_discrete_threshold=star_discrete_threshold,
        n_prototypes=n_prototypes,
        prototype_temp=prototype_temp,
        prototype_entropy_weight=prototype_entropy_weight,
        w_rec=w_rec,
        w_proto=w_proto,
        w_star=w_star,
        w_wgad=w_wgad,
        proto_alpha=proto_alpha,
        wgad_alpha=wgad_alpha,
        star_alpha=star_alpha,
        self_imp_alpha=self_imp_alpha,
        star_num_shared_experts=star_num_shared_experts,
        star_num_experts=star_num_experts,
        star_top_k=star_top_k,
        star_dropout=star_dropout,
        star_sim_weight=star_sim_weight,
        wgad_graph_window_size=wgad_graph_window_size,
        wgad_gcn_layers=wgad_gcn_layers,
        wgad_wave_scales=wgad_wave_scales,
        wgad_lambda_wavelet=wgad_lambda_wavelet,
        wgad_lambda_cl=wgad_lambda_cl,
        wgad_graph_alpha=wgad_graph_alpha,
        wgad_score_mode=wgad_score_mode,
        wgad_affine=wgad_affine,
        wgad_subtract_last=wgad_subtract_last,
        wgad_cl_temperature=wgad_cl_temperature,
        self_imp_steps=self_imp_steps,
        self_imp_lr=self_imp_lr,
        self_imp_l1=self_imp_l1,
        self_imp_tv=self_imp_tv,
        self_imp_huber_delta=self_imp_huber_delta,
        self_imp_hidden_dim=self_imp_hidden_dim,
        self_imp_raw_hidden_dim=self_imp_raw_hidden_dim,
        self_imp_raw_weight=self_imp_raw_weight,
        self_imp_seed=self_imp_seed,
        patience=patience,
        cuda=cuda,
    )
    clf.fit(data_train)
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in clf.model.state_dict().items()
        }
        torch.save(
            {
                "model_state_dict": state_dict,
                "train_mean": clf.train_mean,
                "train_std": clf.train_std,
                "original_input_c": clf.original_input_c,
                "continuous_cols": clf.continuous_cols,
                "discrete_cols": clf.discrete_cols,
                "constant_cols": clf.constant_cols,
                "discrete_vocab_sizes": clf.discrete_vocab_sizes,
                "discrete_value_maps": clf.discrete_value_maps,
                "config": {
                    "input_c": data_test.shape[1],
                    "win_size": win_size,
                    "stride": stride,
                    "train_stride": train_stride,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "validation_size": validation_size,
                    "patch_len": patch_len,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "encoder_layers": encoder_layers,
                    "dropout": dropout,
                    "scales": scales,
                    "use_multiscale": use_multiscale,
                    "use_prototype": use_prototype,
                    "use_wgad": use_wgad,
                    "use_self_imp": use_self_imp,
                    "use_star": use_star,
                    "wgad_graph_window_size": wgad_graph_window_size,
                    "patience": patience,
                },
            },
            checkpoint_path,
        )
    if not return_score:
        return None
    score = clf.decision_function(data_test)
    return score.ravel()


"""
联合训练；实现adapter
"""
def run_TimeRCD_MAFT_FT(
    data_train,
    data_test,
    win_size=64,
    weight=0.5,
    lr_adapter=1e-3,
    epochs_adapter=5,
    adapter_mode="train",
    device="cuda:0",
    adapter_checkpoint_dir="checkpoints/MAFT",
    timercd_win_size=15000,
    timercd_checkpoint="checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
    filename=None,
    **kwargs,
):
    from .models.TimeRCD_MAFT import run_maft

    if filename is None:
        filename = f"official_tr_{len(data_train)}_1st_0.csv"

    return run_maft(
        filename=filename,
        data=data_test,
        win_size=win_size,
        weight=weight,
        lr_adapter=lr_adapter,
        epochs_adapter=epochs_adapter,
        mode=adapter_mode,
        device=device,
        adapter_checkpoint_dir=adapter_checkpoint_dir,
        timercd_win_size=timercd_win_size,
        timercd_checkpoint=timercd_checkpoint,
    ).ravel()


def run_TimeRCD_MAFT(*args, **kwargs):
    return run_TimeRCD_MAFT_FT(*args, **kwargs)


def run_TSPulse_MultiAdapter_FT(
    data_train, 
    data_test,
    win_size=64,             
    batch_size=32,
    num_epochs=5,
    lr_backbone=1e-6,        
    lr_adapter=1e-3,         
    checkpoint="/root/TSB-AD/checkpoints/tspulse",
    device="cuda",
    **kwargs
):
    from TSB_AD.models.TSPulse import TSPulsePipeline as TSPulse
    from .models.MultiAdapter import MultiAdapter
    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
    import torch
    import numpy as np
    
    # 强制刷新缓冲区，确保在 Docker/HPC 环境下能看到输出
    print(">>> Initializing Joint Training (TSPulse + MultiAdapter) without Freezing...", flush=True)

    # ================= 1. 初始化模型 =================
    # 使用你之前修复过的 TSPulsePipeline，加载 D=24, L=8 配置
    clf_tsp = TSPulse(
        aggr_win_size=win_size,
        num_input_channels=data_train.shape[1],
        batch_size=batch_size,
        model_path=checkpoint,         
        finetune_epochs=num_epochs,
        finetune_lr=lr_backbone,
    )
    
    # 确保内部模型已加载
    if clf_tsp._model is None:
        raise ValueError("TSPulse model failed to initialize from checkpoint.")
    
    model_tsp = clf_tsp._model.to(device)

    # 初始化 Adapter
    clf_adapter = MultiAdapter(
        win_size=win_size,
        input_c=data_train.shape[1],
        batch_size=batch_size,
        epochs=num_epochs,
        lr=lr_adapter,
        #**kwargs
    )
    clf_adapter._prepare_array(data_train)
    clf_adapter._fit_feature_schema(data_train)
    clf_adapter._build_model(data_train.shape[1])
    model_adapter = clf_adapter.model.to(device)

    # ================= 2. 差异化优化器 =================
    optimizer = torch.optim.AdamW([
        {'params': model_tsp.parameters(), 'lr': lr_backbone}, 
        {'params': model_adapter.parameters(), 'lr': lr_adapter}
    ], weight_decay=1e-4)

    # ================= 3. 数据准备 =================
    data_train_norm = clf_adapter._normalize(data_train)
    # 使用 Adapter 自带的 loader 逻辑
    loader, _ = clf_adapter._make_loader(data_train_norm, discrete=None, stride=1, shuffle=True)

    # ================= 4. 联合训练循环 =================
    print(f">>> Starting Joint Training (Epochs: {num_epochs})...", flush=True)
    model_tsp.train()
    model_adapter.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device).float()
            # TSPulse 需要的 mask (全 1 表示全重构)
            batch_mask = torch.ones(batch_x.shape[0], batch_x.shape[1], dtype=torch.bool).to(device)
            
            optimizer.zero_grad()

            # 1) Adapter 前向
            out_adapter = model_adapter.compute_window_outputs(batch_x)
            loss_adapter = (
                float(clf_adapter.w_rec) * out_adapter["loss_rec"] +
                float(clf_adapter.w_proto) * out_adapter["loss_proto"] +
                float(clf_adapter.w_wgad) * out_adapter["loss_wgad"]
            )

            # 2) TSPulse 前向
            #print(model_tsp.forward)
            #print("batch_x:", batch_x.shape)
            #print("batch_mask:", batch_mask.shape)
            outputs_tsp = model_tsp(batch_x, batch_mask, return_loss=False )
            # 根据 Granite-TSFM 结构，从 last_hidden_state 获取重构
            if hasattr(model_tsp, 'reconstruction_head'):
                preds_tsp = model_tsp.reconstruction_head(outputs_tsp.last_hidden_state).squeeze(-1)
                scale = batch_x.shape[1] // preds_tsp.shape[1]   # 512 / 16 = 32
                preds_tsp = preds_tsp.repeat_interleave(scale, dim=1)
                loss_tsp = torch.nn.functional.mse_loss(preds_tsp, batch_x.squeeze(-1))
            else:
                loss_tsp = torch.tensor(0.0, device=device)

            loss_total = loss_tsp + loss_adapter
            loss_total.backward()
            
            # 针对大模型的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_tsp.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss_total.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Joint Loss: {epoch_loss/len(loader):.4f}", flush=True)

    # ================= 5. 联合推理 (关键修复点) =================
    print(">>> Joint Training Finished. Starting Inference...", flush=True)
    model_tsp.eval()
    model_adapter.eval()

    print("data_test shape:", data_test.shape)
    
    with torch.no_grad():
        data_test_norm = clf_adapter._normalize(data_test)
        print("data_test_norm =", data_test_norm.shape)

        test_loader, _ = clf_adapter._make_loader(
            data_test_norm,
            discrete=None,
            stride=1,
            shuffle=False
        )

        print("test_loader:", test_loader)
        print("len(test_loader):", len(test_loader))

        scores_tsp_list = []
        for batch_test, _ in test_loader:
            try:
                batch_test = batch_test.to(device).float()
                # 形状: [batch, seq_len, 1]
                mask_test = torch.ones(batch_test.shape[0], batch_test.shape[1]).bool().to(device)

                # ==========================================
                # 🔥 绕过 forward 逻辑，直接调 backbone
                # ==========================================
                # TSPulse 的主干网络通常返回 last_hidden_state
                outputs = model_tsp.backbone(
                    time_series=batch_test, 
                    mask=mask_test
                )
                
                # 获取特征向量 (last_hidden_state)
                last_hidden_state = outputs.last_hidden_state

                if hasattr(model_tsp, "reconstruction_head"):
                    # 运行重构头拿到预测值
                    pred = model_tsp.reconstruction_head(last_hidden_state)
                    
                    # 确保维度对齐 [batch, seq_len, channels]
                    if pred.dim() == 4: # 有时会多出一维 patch
                        pred = pred.mean(dim=-2) 
                    
                    pred = pred.reshape(batch_test.shape)

                    # 计算 MSE 分数
                    score = torch.mean((pred - batch_test) ** 2, dim=(1, 2))
                    scores_tsp_list.append(score.cpu().numpy())
                else:
                    # 如果没有重构头，可以拿特征的范数作为临时分数，或者打印属性看看
                    print("Warning: reconstruction_head not found!")

            except Exception as e:
                print(f"Batch failed: {e}")
                # 打印出具体的维度信息，方便排查
                print(f"Batch shape: {batch_test.shape}")
                continue

    
        # =========================
        # ✔ FIX 3: 防止 concat 空
        # =========================
        if len(scores_tsp_list) == 0:
            raise RuntimeError("❌ scores_tsp_list is empty, inference failed in all batches")

        score_tsp = np.concatenate(scores_tsp_list)

        # =========================
        # ✔ FIX 4: 对齐长度（window -> point）
        # =========================
        target_len = len(data_test)

        if len(score_tsp) < target_len:
            pad_len = target_len - len(score_tsp)
            score_tsp = np.pad(score_tsp, (pad_len, 0), mode='edge')
        else:
            score_tsp = score_tsp[:target_len]

        # =========================
        # Adapter score（保持原逻辑）
        # =========================
        score_adapter = clf_adapter.decision_function(data_test).reshape(-1)

        # =========================
        # ✔ FIX 5: safe normalization
        # =========================
        score_tsp_norm = score_tsp / (np.max(score_tsp) + 1e-8)
        score_adapter_norm = score_adapter / (np.max(score_adapter) + 1e-8)

        final_score = 0.5 * score_tsp_norm + 0.5 * score_adapter_norm

    print(type(final_score))
    print(final_score)
    
    # 返回字典，彻底解决 'list' object has no attribute 'keys'
    return {"score": final_score.ravel()}

def run_xLSTMAD(data_train, data_test, window_size=100, lr=0.005, batch_size=32, embedding_dim=40):
    from .models.xLSTMAD import xLSTMAD, xLSTMADModule
    model = xLSTMADModule(embedding_dim=embedding_dim, window_size=window_size, lr=lr, features_no=data_test.shape[1])
    clf = xLSTMAD(model=model, window_size=window_size, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_MMPAD(data, periodicity=1, n_dim=None, n_neighbor=1,
              sorting_place='pre', mode='discord', post_processing=2, 
              n_job=None, backend=None):
    from .models.MMPAD import MMPAD
    clf = MMPAD(periodicity=periodicity, n_dim=n_dim, n_neighbor=n_neighbor,
                sorting_place=sorting_place, mode=mode, post_processing=post_processing, 
                n_job=n_job, backend=backend)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CHARM(
    data_train,
    data_test,
    window_size=128,
    k=3,
    pointwise_agg="mean",
    stride=1,
    train_stride=1,
    min_window=64,
):
    from .models.CHARM import CHARM_AD
    """Semisupervised runner — matches TSB-AD's run_Semisupervise_AD dispatcher."""
    clf = CHARM_AD(
        HP={},
        window_size=window_size,
        stride=stride,
        k=k,
        pointwise_agg=pointwise_agg,
        train_stride=train_stride,
        min_window=min_window,
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    )
    return score
