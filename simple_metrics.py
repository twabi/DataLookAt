import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def main():
    pass

def smooth_plot(column_x, column_y, title, x_label, y_label):
    x_new = np.linspace(column_x.min(), column_x.max(), 300)

    for value in column_y:
        spl_delay = make_interp_spline(column_x, value, k=1)
        y_smooth_delay = spl_delay(x_new)
        plt.plot(x_new, y_smooth_delay, label=value.name)

    plt.title(title)
    plt.xlabel(xlabel=x_label, fontsize=12)
    plt.ylabel(ylabel=y_label, fontsize=12)
    plt.xticks(fontsize=10, rotation=0, ha='right')
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()


    plt.show()
    #plt.savefig("C:/Users/itwab/Downloads/Video/msVtp-3.png")

def msg_size_vs_throughput():
    prod_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/msVtp/producer-payloads-msVtp-trend-4-12-3.csv")
    cons_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/msVtp/consumer-payloads-msVtp-trend-4-12-3.csv")

    all_df = prod_df.merge(cons_df, on="message_uuid")
    all_df['latency'] = all_df['consume_time'] - all_df['produce_time_x']
    all_df['time_diff'] = all_df['produce_time_x'].diff() / 1000

    diff_df = all_df.groupby(['msg_size_x'])[['latency', 'time_diff', 'consume_time']].agg({
        'latency': 'mean',
        'time_diff': 'mean',
        'consume_time' : 'count'
    }).reset_index()

    diff_df['group_id'] = diff_df.index // 3

    aggregated_df = diff_df.groupby('group_id').agg({
        'latency': 'mean',
        'msg_size_x': 'mean',
        'time_diff': 'sum',
        'consume_time': 'sum'
    }).reset_index(drop=True)
    aggregated_df.rename(columns={'consume_time': 'records'}, inplace=True)
    aggregated_df['throughput'] = ((aggregated_df['records'] * aggregated_df['msg_size_x'])/ aggregated_df['time_diff'])/1000000
    aggregated_df['record_rate'] = aggregated_df['records'] / aggregated_df['time_diff']

    aggregated_df = aggregated_df[(aggregated_df['msg_size_x'] <= 20000)]
    print(aggregated_df)
    #aggregated_df = aggregated_df.sort_values(by=['latency'])
    smooth_plot(aggregated_df['msg_size_x'], [aggregated_df['record_rate']],
                "Latency vs Msg Size", "msg size (bytes)", "Latency (ms)", )


def part_vs_throughput():
    pass

if __name__ == '__main__':
    # msg_size_vs_throughput()
    part_vs_throughput()