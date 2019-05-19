import pandas as pd

caching_strategy_dir = 'caching_strategy'
eviction_strategy_dir = 'eviction_strategy'


def plot_hitrate(directory):
    stats_df = pd.read_csv(f'{directory}/evaluation_logger.log',
                           names=['timestamp', 'key', 'observation', 'episode'],
                           usecols=['episode', 'observation', 'key'])
    stats = stats_df.groupby(['episode', 'observation']).count().unstack(0)['key'].fillna(0).transpose()
    hit_ratio_all = stats['Hit'] / stats.sum(axis=1)
    hit_ratio_all.index = hit_ratio_all.index - 1
    write_ratio_df = pd.DataFrame({'write_ratio': [0, 5, 10, 25, 50, 100]})
    write_ratio_df['hit_ratio'] = hit_ratio_all
    write_ratio_df = write_ratio_df.set_index('write_ratio')
    ax = write_ratio_df.plot()
    ax.set_xlabel('Write Ratio (%)')
    ax.set_ylabel('Hit Ratio')


def plot_eviction_precision(directory, eviction_name):
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
    #https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
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
