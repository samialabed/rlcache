import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

''' collection of tools to help visualise and fix missing data points in collected data.'''
caching_strategy_dir = 'caching_strategy'
eviction_strategy_dir = 'eviction_strategy'

eviction_name = {
    'simple_strategy': 'lru_eviction_strategy',
    'simple_strategy_fifo': 'fifo_eviction_strategy',
    'rl_eviction_strategy': 'rl_eviction_strategy'}


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


def save_everything_f1(directory: str, methods: List, output: str, overwrite_cols: Dict[str, str], metric='f1'):
    capacities = [100, 1000, 2500, 5000]
    label_map = {'f1': 'F1 Score',
                 'manual_evicts': 'Manual Evictions',
                 'precision': 'Precision Score'
                 }

    # capacities = [100]
    for capacity in capacities:
        means, errors = calculate_eviction_score_with_varying_methods(directory, methods, capacity, metric)
        means.rename(columns=overwrite_cols, inplace=True)
        errors.rename(columns=overwrite_cols, inplace=True)
        ax = means.plot.bar(yerr=errors)
        ax.set_xlabel('Write Ratio (%)')
        ax.set_ylabel(label_map[metric])
        ax.set_title(f'Cache Capacity {capacity}')
        fig = ax.get_figure()
        fig.savefig(f'{output}/{metric}_{capacity}.pdf')
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
            end_of_episode_stats_df = fix_cummulative_sum_in_EOE_logger(end_of_episode_stats_df)
        cache_rate = end_of_episode_stats_df['should_cache_ratio'].drop(0)
        cache_rate.index = write_ratio_df.index
        write_ratio_df[sub_dir] = cache_rate

    means = write_ratio_df.mean(axis=1)
    errors = write_ratio_df.std(axis=1)
    return means, errors


def fix_cummulative_sum_in_EOE_logger(end_of_episode_stats_df):
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


def calculate_eviction_score_with_varying_methods(directory: str, methods: List, capacity: int, metric):
    res_df = pd.DataFrame()
    errs_df = pd.DataFrame()

    for method in methods:
        precision_df, recall_df, f1_df, manual_evicts_df = calculate_eviction_score(
            f'{directory}/{method}/cache_capacity_{capacity}',
            eviction_name[method])

        if metric == 'f1':
            res_df[f'{method}'] = f1_df.mean(axis=1)
            errs_df[f'{method}'] = f1_df.std(axis=1)
        elif metric == 'manual_evicts':
            res_df[f'{method}'] = manual_evicts_df.mean(axis=1)
            errs_df[f'{method}'] = manual_evicts_df.std(axis=1)
        elif metric == 'precision':
            res_df[f'{method}'] = precision_df.mean(axis=1)
            errs_df[f'{method}'] = precision_df.std(axis=1)

    return res_df, errs_df


def calculate_eviction_score(directory: str,
                             eviction_name: str):
    MISSING_EPISODE_NUMBER_TIMESTAMP = '2019_05_19_14_27'
    BUG_IN_CUMMULATIVE_SUM_EOE_LOGGER = '2019_05_24_13_32'

    sub_dirs = os.listdir(directory)

    precision_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    precision_df = precision_df.set_index('write_ratio')
    recall_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    recall_df = recall_df.set_index('write_ratio')
    f1_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    f1_df = f1_df.set_index('write_ratio')
    manual_evicts_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    manual_evicts_df = manual_evicts_df.set_index('write_ratio')

    for sub_dir in sub_dirs:
        if time.strptime(sub_dir, '%Y_%m_%d_%H_%M') < time.strptime(MISSING_EPISODE_NUMBER_TIMESTAMP, '%Y_%m_%d_%H_%M') \
                and eviction_name == 'rl_eviction_strategy':
            eviction_performance_df = fix_missing_episode_in_eviction(f'{directory}/{sub_dir}',
                                                                      eviction_name)
        else:
            eviction_performance_df = pd.read_csv(
                f'{directory}/{sub_dir}/eviction_strategy/{eviction_name}_performance_logger.log',
                names=['timestamp', 'episode', 'state'])
        eviction_performance = eviction_performance_df.groupby(['episode', 'state']
                                                               ).count().unstack(0)['timestamp'].fillna(0).transpose()
        precision, recall, f1 = calculate_f1_measure(eviction_performance)
        precision.index = precision_df.index
        recall.index = recall_df.index
        precision_df[sub_dir] = precision
        recall_df[sub_dir] = recall

        f1.index = f1_df.index
        f1_df[sub_dir] = f1

        # count manual evicts
        end_of_episode_stats_df = pd.read_csv(f'{directory}/{sub_dir}/end_of_episode_logger.log',
                                              names=['timestamp', 'episode_num',
                                                     'invalidate', 'hit',
                                                     'miss', 'hit_ratio',
                                                     'should_cache',
                                                     'should_not_cache',
                                                     'should_cache_ratio',
                                                     'manual_evicts',
                                                     'cache_utility'])
        if time.strptime(sub_dir, '%Y_%m_%d_%H_%M') < time.strptime(BUG_IN_CUMMULATIVE_SUM_EOE_LOGGER,
                                                                    '%Y_%m_%d_%H_%M'):
            end_of_episode_stats_df = fix_cummulative_sum_in_EOE_logger(end_of_episode_stats_df)
        end_of_episode_stats_df = end_of_episode_stats_df.drop(0)
        manual_evicts_df[sub_dir] = end_of_episode_stats_df['manual_evicts'].values

    return precision_df, recall_df, f1_df, manual_evicts_df


def calculate_f1_measure(perf):
    # https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
    # TrueEvict: True positive
    # TrueMiss: True Negative
    # FalseEvict: False Positive
    # MissEvict: False Negative
    true_negative = 0 if 'TrueMiss' not in perf.columns else perf['TrueMiss']
    false_negative = 0 if 'MissEvict' not in perf.columns else perf['MissEvict']
    true_positive = 1 if 'TrueEvict' not in perf.columns else perf['TrueEvict']
    false_positive = 1 if 'FalseEvict' not in perf.columns else perf['FalseEvict']

    precision = (true_positive + true_negative) / (true_positive + true_negative + false_positive)
    recall = (true_positive + true_negative) / (true_positive + true_negative + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1


def fix_missing_episode_in_eviction(directory: str,
                                    eviction_name: str):
    # get episodes timestamps
    end_of_episode_stats_df = pd.read_csv(f'{directory}/end_of_episode_logger.log',
                                          names=['timestamp', 'episode_num',
                                                 'invalidate', 'hit',
                                                 'miss', 'hit_ratio',
                                                 'should_cache',
                                                 'should_not_cache',
                                                 'should_cache_ratio',
                                                 'manual_evicts',
                                                 'cache_utility'],
                                          usecols=['timestamp'])
    episodes_break_timestamp = end_of_episode_stats_df.drop(0).values.flatten().tolist()
    # read eviction performance
    eviction_performance_df = pd.read_csv(
        f'{directory}/eviction_strategy/{eviction_name}_performance_logger.log',
        names=['timestamp', 'state'], usecols=['timestamp', 'state'])

    # add the missing episode to the timestamp
    timestamp_df = eviction_performance_df['timestamp']
    eviction_performance_df['episode'] \
        = np.where(timestamp_df <= episodes_break_timestamp[0], 1,
                   np.where(timestamp_df <= episodes_break_timestamp[1], 2,
                            np.where(timestamp_df <= episodes_break_timestamp[2], 3,
                                     np.where(timestamp_df <= episodes_break_timestamp[3], 4,
                                              np.where(timestamp_df <= episodes_break_timestamp[4], 5,
                                                       6)
                                              )
                                     )
                            )
                   )
    return eviction_performance_df
