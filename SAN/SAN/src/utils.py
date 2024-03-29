import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import pickle
import copy


def plot_grad_iter(result_dict, title, save_path, threshold=1e-6):
    plt.rc('text')
    plt.rc('font', family='sans-serif')
    plt.figure(figsize=(9, 6), dpi=1200)

    markers = ["^-", "d-", "*-", ">-", "+-", "o-", "v-", "<-"]
    for algo_name, marker in zip(sorted(result_dict.keys()), markers):
        result = result_dict[algo_name]
        # result is a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
        len_cut = len(min(result, key=len))
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        # plot
        grad_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = np.argmax(grad_avg <= threshold) + \
                1 if np.sum(grad_avg <= threshold) > 0 else len(grad_avg)
            grad_avg = grad_avg[:len_cut]
        grad_min = np.min(result, axis=0)[:len(grad_avg)]
        grad_max = np.max(result, axis=0)[:len(grad_avg)]
        # grad_std = np.std(result, axis=0)
        # grad_min = np.add(grad_avg, -grad_std)
        # grad_max = np.add(grad_avg, grad_std)
        plt.semilogy(np.arange(len(grad_avg)), grad_avg, marker, label=algo_name, lw=2)
        plt.fill_between(np.arange(len(grad_avg)), grad_min, grad_max, alpha=0.2)

    plt.tick_params(labelsize=20)
    plt.legend(fontsize=30)
    plt.xlabel("Effective Passes", fontsize=25)
    plt.ylabel("Del", fontsize=25)
    plt.grid(True)
    plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + "_grad_iter.pdf"),
                bbox_inches='tight', pad_inches=0.01)


def plot_grad_time(grad_dict, time_dict, title, save_path, threshold=1e-6):
    plt.rc('text')
    plt.rc('font', family='sans-serif')
    plt.figure(figsize=(9, 6), dpi=1200)

    markers = ["^-", "d-", "*-", ">-", "+-", "o-", "v-", "<-"]
    for algo_name, marker in zip(sorted(grad_dict.keys()), markers):
        grad = grad_dict[algo_name]
        time = time_dict[algo_name]
        # grad and time are both a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
        len_cut = len(min(grad, key=len))
        grad_r = np.array(list(map(lambda arr: arr[:len_cut], grad)))
        time_r = np.array(list(map(lambda arr: arr[:len_cut], time)))
        # plot
        grad_avg = np.mean(grad_r, axis=0)
        time_avg = np.mean(time_r, axis=0)
        if threshold:
            len_cut = np.argmax(grad_avg <= threshold) + \
                1 if np.sum(grad_avg <= threshold) > 0 else len(grad_avg)
            grad_avg = grad_avg[:len_cut]
            time_avg = time_avg[:len_cut]
        plt.semilogy(time_avg, grad_avg, marker, label=algo_name, lw=2)

    plt.tick_params(labelsize=20)
    plt.legend(fontsize=30)
    plt.xlabel("time (s)", fontsize=25)
    plt.ylabel("Del", fontsize=25)
    plt.grid(True)
    plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + "_grad_time.pdf"),
                bbox_inches='tight', pad_inches=0.01)


def plot_sub_optimality(result_dict, title, save_path, f_opt=0.0, threshold=1e-6):
    plt.rc('text')
    plt.rc('font', family='sans-serif')
    plt.figure(figsize=(9, 6), dpi=1200)

    markers = ["^-", "d-", "*-", ">-", "+-", "o-", "v-", "<-"]
    for algo_name, marker in zip(sorted(result_dict.keys()), markers):
        result = result_dict[algo_name]
        # result is a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
        len_cut = len(min(result, key=len))
        result =  np.abs(np.array(list(map(lambda arr: arr[:len_cut], result))) - f_opt)
        if (result < 0).any():
            print("WARNING: there are negative numbers in sub-optimality plots !!!")
            print("WARNING: replace all negative numbers by absolute in in sub-optimality plots !!!")
            result = np.abs(result)
        # result = np.array([r - f_opt for r in result_])
        # plot
        gap_avg = np.mean(result, axis=0)
        # if threshold:
        #     len_cut = np.argmax(grad_avg <= threshold) + \
        #         1 if np.sum(grad_avg <= threshold) > 0 else len(grad_avg)
        #     grad_avg = grad_avg[:len_cut]
        gap_min = np.min(result, axis=0)[:len(gap_avg)]
        gap_max = np.max(result, axis=0)[:len(gap_avg)]
        plt.semilogy(np.arange(len(gap_avg)), gap_avg, marker, label=algo_name, lw=2)
        plt.fill_between(np.arange(len(gap_avg)), gap_min, gap_max, alpha=0.2)

    plt.tick_params(labelsize=20)
    plt.legend(fontsize=30)
    plt.xlabel("Effective Passes", fontsize=25)
    plt.ylabel("$ f - f^* $", fontsize=25)
    plt.grid(True)
    plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + "_subopt.pdf"),
                bbox_inches='tight', pad_inches=0.05)


def run_algorithm(algo_name, solver, algo_kwargs, n_repeat):
    logging.info("------START {}------".format(algo_name))
    grad_iter, grad_time, error_iter = [], [], []
    copy_solvers = [copy.deepcopy(solver) for _ in range(n_repeat)]
    for i in range(n_repeat):
        logging.info("{}-th repetition:".format(i + 1))
        final_w, norm, times, errors = copy_solvers[i].run(**algo_kwargs)
        grad_iter.append(norm)
        grad_time.append(times)
        error_iter.append(errors)
        # print("accuracy: {}".format(np.mean(np.sign(algo_kwargs['data'] @ final_w) == algo_kwargs['label'])))
    logging.info("------END {}------".format(algo_name))
    # print("MAE: {}".format(np.mean(np.abs(algo_kwargs['data']@final_w - algo_kwargs['label']))))
    return grad_iter, grad_time, error_iter


def save(path_grad_iter, grad_iter, path_grad_time, grad_time, path_error_iter, error_iter):
    with open(path_grad_iter, 'wb') as fp:
        pickle.dump(grad_iter, fp)
    with open(path_grad_time, 'wb') as fp:
        pickle.dump(grad_time, fp)
    with open(path_error_iter, 'wb') as fp:
        pickle.dump(error_iter, fp)


def load(path_grad_iter, path_grad_time, path_error_iter):
    grad_iter, grad_time, error_iter = None, None, None
    if os.path.isfile(path_grad_iter):
        with open(path_grad_iter, 'rb') as fp:
            grad_iter = pickle.load(fp)
    if os.path.isfile(path_grad_time):
        with open(path_grad_time, 'rb') as fp:
            grad_time = pickle.load(fp)
    if os.path.isfile(path_error_iter):
        with open(path_error_iter, 'rb') as fp:
            error_iter = pickle.load(fp)
    return grad_iter, grad_time, error_iter


def lipschitz_ridge(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / n + reg


def lipschitz_logistic(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / (4. * n) + reg


def max_Li_ridge(X, reg):
    return np.max(np.sum(X ** 2, axis=1)) + reg


def max_Li_logistic(X, reg):
    return 0.25 * np.max(np.sum(X ** 2, axis=1)) + reg


def cond(data):
    # data: numpy array of shape n x d
    # singular val of (data.T @ data)
    # return condition number of a given dataset, i.e., max(singular val) / min non zero(singular val)
    n, d = data.shape
    if n >= d:
        s = np.linalg.svd(data.T@data, compute_uv=False, hermitian=True)
    if n < d:
        s = np.linalg.svd(data@data.T, compute_uv=False, hermitian=True)
    # singular values are sorted in descending order
    min_non_zero_pos = len(s) - 1
    while min_non_zero_pos >= 0 and s[min_non_zero_pos] == 0:
        min_non_zero_pos -= 1
    if min_non_zero_pos < 0:
        return float('inf')
    return s[0] / s[min_non_zero_pos]


# ==========================
# two auxiliary functions to incorporate the use of scipy fmin_l_bfgs_b
# ==========================
def f_val_logistic(x, data, label, loss, regularizer, reg):
    return np.mean(loss.val(label, data @ x)) + reg * regularizer.val(x)


def f_grad_logistic(x, data, label, loss, regularizer, reg):
    return np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
