import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import time

caching_strategy_dir = 'caching_strategy'
eviction_strategy_dir = 'eviction_strategy'


def plot_hitrate(means: pd.Series, errors: pd.Series):
    fig, ax = plt.subplots()
    ax.set_xlabel('Write Ratio (%)')
    ax.set_ylabel('Hit Ratio')
    means.plot.line(yerr=errors, ax=ax)


def plot_everything_hit_rate(directory: str, methods: List):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Cache hit rate on various cache capacities.')

    res_100_df, err_100_df = calculate_hitrate_with_varying_methods(directory, methods, 100)
    res_100_df.plot(yerr=err_100_df, ax=ax[0][0], title='Capacity 100')
    res_1000_df, err_1000_df = calculate_hitrate_with_varying_methods(directory, methods, 1000)
    res_1000_df.plot(yerr=err_1000_df, ax=ax[0][1], title='Capacity 1000')
    res_2500_df, err_2500_df = calculate_hitrate_with_varying_methods(directory, methods, 2500)
    res_2500_df.plot(yerr=err_2500_df, ax=ax[1][0], title='Capacity 2500')
    res_5000_df, err_5000_df = calculate_hitrate_with_varying_methods(directory, methods, 5000)
    res_5000_df.plot(yerr=err_5000_df, ax=ax[1][1], title='Capacity 5000')


def save_everything_hit_rate(directory: str, methods: List, output: str, overwrite_cols: Dict[str, str]):
    capacities = [100, 1000, 2500, 5000]
    # capacities = [100]
    for capacity in capacities:
        means, errors = calculate_hitrate_with_varying_methods(directory, methods, capacity)
        means.rename(columns=overwrite_cols, inplace=True)
        errors.rename(columns=overwrite_cols, inplace=True)
        ax = means.plot.line(yerr=errors)
        ax.set_xlabel('Write Ratio (%)')
        ax.set_ylabel('Hit Ratio')
        ax.set_title(f'Cache Capacity {capacity}')
        fig = ax.get_figure()
        fig.savefig(f'{output}/hitrate_{capacity}.pdf')
        # TODO make it save to location


def save_everything_cache_rate(directory: str, methods: List, output: str, overwrite_cols: Dict[str, str]):
    capacities = [100, 1000, 2500, 5000]
    # capacities = [100]
    for capacity in capacities:
        means, errors = calculate_cache_rate_with_varying_methods(directory, methods, capacity)
        means.rename(columns=overwrite_cols, inplace=True)
        errors.rename(columns=overwrite_cols, inplace=True)
        means.rename(columns=overwrite_cols, inplace=True)
        errors.rename(columns=overwrite_cols, inplace=True)
        ax = means.plot.bar(yerr=errors)
        ax.set_xlabel('Write Ratio (%)')
        ax.set_ylabel('Cache Ratio')
        ax.set_title(f'Cache Capacity {capacity}')
        fig = ax.get_figure()
        fig.savefig(f'{output}/cache_rate_{capacity}.pdf')


def calculate_cache_rate_with_varying_methods(directory: str, methods: List, capacity: int):
    res_df = pd.DataFrame()
    errs_df = pd.DataFrame()

    for method in methods:
        res_df[f'{method}'], errs_df[f'{method}'] = calculate_cache_rate(
            f'{directory}/{method}/cache_capacity_{capacity}')

    return res_df, errs_df


def calculate_hitrate_with_varying_methods(directory: str, methods: List, capacity: int):
    res_df = pd.DataFrame()
    errs_df = pd.DataFrame()

    for method in methods:
        res_df[f'{method}'], errs_df[f'{method}'] = calculate_hitrate(f'{directory}/{method}/cache_capacity_{capacity}')

    return res_df, errs_df


def calculate_hitrate_on_varying_capacity(directory: str,
                                          capacities: List = None) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    if capacities is None:
        capacities = [100, 1000, 2500, 5000]
    res = {}
    for capacity in capacities:
        means, errors = calculate_hitrate(f'{directory}/cache_capacity_{capacity}/')
        res[f'{capacity}'] = (means, errors)

    return res


def calculate_cache_rate(directory: str):
    TIME_OF_FIX_IMPLEMENTATION = '2019_05_24_13_32'

    sub_dirs = os.listdir(directory)
    write_ratio_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    write_ratio_df = write_ratio_df.set_index('write_ratio')

    for sub_dir in sub_dirs:
        end_of_episode_stats_df = pd.read_csv(f'{directory}/{sub_dir}/end_of_episode_logger.log',
                                              names=['timestamp', 'episode_num',
                                                     'invalidate', 'hit',
                                                     'miss', 'hit_ratio',
                                                     'should_cache',
                                                     'should_not_cache',
                                                     'should_cache_ratio',
                                                     'manual_evicts',
                                                     'cache_utility'])
        if time.strptime(sub_dir, '%Y_%m_%d_%H_%M') < time.strptime(TIME_OF_FIX_IMPLEMENTATION, '%Y_%m_%d_%H_%M'):
            end_of_episode_stats_df = fix_old_bug_in_cal_cache_rate(end_of_episode_stats_df)
        cache_rate = end_of_episode_stats_df['should_cache_ratio'].drop(0)
        cache_rate.index = write_ratio_df.index
        write_ratio_df[sub_dir] = cache_rate

    means = write_ratio_df.mean(axis=1)
    errors = write_ratio_df.std(axis=1)
    return means, errors


def fix_old_bug_in_cal_cache_rate(end_of_episode_stats_df):
    df_should_cache = end_of_episode_stats_df['should_cache']
    df_should_not_cache = end_of_episode_stats_df['should_not_cache']
    df_should_cache_ratio = end_of_episode_stats_df['should_cache_ratio']
    df_cache_hit = end_of_episode_stats_df['hit']
    df_cache_miss = end_of_episode_stats_df['miss']
    df_cache_hit_ratio = end_of_episode_stats_df['hit_ratio']
    df_invalidate = end_of_episode_stats_df['invalidate']
    df_manual_evicts = end_of_episode_stats_df['manual_evicts']
    for i in range(6, 1, -1):
        df_should_cache[i] = (df_should_cache[i] - df_should_cache[i - 1]).item()
        df_should_not_cache[i] = (df_should_not_cache[i] - df_should_not_cache[i - 1]).item()
        df_should_cache_ratio[i] = (df_should_cache[i] / max(df_should_not_cache[i] + df_should_cache[i],
                                                             1)).item() * 100
        df_cache_hit[i] = (df_cache_hit[i] - df_cache_hit[i - 1]).item()
        df_cache_miss[i] = (df_cache_miss[i] - df_cache_miss[i - 1]).item()
        df_cache_hit_ratio[i] = (df_cache_hit[i] / max(df_cache_miss[i] + df_cache_hit[i], 1)).item() * 100
        df_invalidate[i] = (df_invalidate[i] - df_invalidate[i - 1]).item()
        df_manual_evicts[i] = (df_manual_evicts[i] - df_manual_evicts[i - 1]).item()

    return end_of_episode_stats_df


def calculate_hitrate(directory: str, q95=False) -> Tuple[pd.Series, pd.Series]:
    sub_dirs = os.listdir(directory)
    write_ratio_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    write_ratio_df = write_ratio_df.set_index('write_ratio')

    for sub_dir in sub_dirs:
        stats_df = pd.read_csv(f'{directory}/{sub_dir}/evaluation_logger.log',
                               names=['timestamp', 'key', 'observation', 'episode'],
                               usecols=['episode', 'observation', 'key'])
        if q95:
            stats = stats_df.groupby(['episode', 'observation', (stats_df.index // 10000)]).count().unstack(0)[
                'key'].fillna(0).transpose()
            hit_ratio_all = (stats['Hit'] / 10000).quantile(0.95, axis=1)
        else:
            stats = stats_df.groupby(['episode', 'observation']).count().unstack(0)['key'].fillna(0).transpose()
            hit_ratio_all = stats['Hit'] / stats.sum(axis=1)

        hit_ratio_all.index = write_ratio_df.index
        write_ratio_df[sub_dir] = hit_ratio_all

    means = write_ratio_df.mean(axis=1)
    errors = write_ratio_df.std(axis=1)
    return means, errors


def plot_eviction_precision(directory: str, eviction_name: str):
    eviction_performance_df = pd.read_csv(f'{directory}/eviction_strategy/{eviction_name}performance_logger.log',
                                          names=['timestamp', 'episode', 'state'])
    eviction_performance = eviction_performance_df.groupby(['episode', 'state']).count().unstack(0)['timestamp'].fillna(
        0).transpose()
    eviction_performance.index = [0, 5, 10, 25, 50, 100]
    ax = eviction_performance.plot()
    ax.set_xlabel('Write Ratio (%)')
    ax.set_ylabel('Hit Ratio')

    return eviction_performance


def calculate_f1_measure(perf):
    # https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
    # TrueEvict: True positive
    # TrueMiss: True Negative
    # FalseEvict: False Positive
    # MissEvict: False Negative
    true_negative = 0 if 'TrueMiss' not in perf.columns else perf['TrueMiss']
    false_negative = 0 if 'MissEvict' not in perf.columns else perf['MissEvict']
    true_positive = perf['TrueEvict']
    false_positive = perf['FalseEvict']

    precision = (true_positive + true_negative) / (true_positive + true_negative + false_positive)
    recall = (true_positive + true_negative) / (true_positive + true_negative + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1
