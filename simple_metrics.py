import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from word2number import w2n
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import json

def main():
    pass

def smooth_plot(column_x, column_y, title, x_label, y_label, filename):
    x_new = np.linspace(column_x.min(), column_x.max(), 300)

    for value in column_y:
        spl_delay = make_interp_spline(column_x, value, k=2)
        y_smooth_delay = spl_delay(x_new)
        plt.plot(x_new, y_smooth_delay, label=value.name)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel(xlabel=x_label, fontsize=12)
    plt.ylabel(ylabel=y_label, fontsize=12)
    plt.xticks(fontsize=10, rotation=0, ha='right')
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()

    plt.show()
    #plt.savefig(f"C:/Users/itwab/Downloads/Video/{filename}.png")
def time_graph(all_df, index):
    print(all_df)
    time_df = all_df.groupby(['time'])[['producer_count', 'consumer_count', 'tx_bytes_x']].mean().reset_index()
    # print(time_df)
    time_df['time'] = pd.to_datetime(time_df['time'], utc=True, unit='s').map(lambda x: x.tz_convert('Asia/Tokyo'))

    print(time_df['time'])
    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the size of the plot as needed

    plt.plot(time_df['time'], time_df['producer_count'], label='Number of Producers')
    plt.plot(time_df['time'], time_df['consumer_count'], label='Number of Consumers')

    # Formatting the date axis
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # Set major ticks to every 10 minutes
    plt.gca().xaxis_date('Asia/Tokyo')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show hours and minutes on the x-axis

    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()

    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title(f'Time Series of Producers and Consumers -> run-{index}')
    plt.legend()
    plt.grid(True)

    plt.show()
    #plt.savefig(f"C:/Users/itwab/Downloads/8-03/run-{index}.png")
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
    # aggregated_df = aggregated_df.sort_values(by=['latency'])
    smooth_plot(aggregated_df['msg_size_x'], [aggregated_df['record_rate']],
                "Latency vs Msg Size", "msg size (bytes)", "Latency (ms)", "msVtp-3")

def part_vs_throughput():
    prod_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/partVtp/producer-payloads-partVtp-trend-4-n-2.csv")
    cons_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/partVtp/consumer-payloads-partVtp-trend-4-n-2.csv")

    all_df = prod_df.merge(cons_df, on="message_uuid")

    all_df['latency'] = all_df['consume_time'] - all_df['produce_time_x']
    all_df['time_diff'] = all_df['produce_time_x'].diff() / 1000

    diff_df = all_df.groupby(['topic_x'])[['latency', 'msg_size_x', 'time_diff', 'consume_time']].agg({
        'latency': 'mean',
        'time_diff': 'mean',
        'msg_size_x' : 'mean',
        'consume_time': 'count'
    }).reset_index()

    if isinstance(diff_df['topic_x'].iloc[0], str):
        # Apply transformation row-wise
        diff_df['topic_x'] = diff_df['topic_x'].apply(lambda x: w2n.word_to_num(x.split("-")[0]))

    diff_df.rename(columns={'consume_time': 'records'}, inplace=True)
    diff_df['throughput'] = ((diff_df['records'] * diff_df['msg_size_x'])/ diff_df['time_diff'])/1000000
    #diff_df.drop(columns=['topic_x'], inplace=True)

    diff_df = diff_df.sort_values(by=['topic_x'])
    diff_df = diff_df[(diff_df['topic_x'] != 9)]

    print(diff_df)
    smooth_plot(diff_df['topic_x'], [diff_df['throughput']],
                "Partition vs Latency", "partitions", "Latency (ms)", "partVtp-4")


def rep_factor_vs_throughput():
    prod_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/repVtp/producer-payloads-repVtp-trend-4-12-1.csv")
    cons_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/repVtp/consumer-payloads-repVtp-trend-4-12-1.csv")

    all_df = prod_df.merge(cons_df, on="message_uuid")
    all_df['latency'] = all_df['consume_time'] - all_df['produce_time_x']
    all_df['time_diff'] = all_df['produce_time_x'].diff() / 1000

    diff_df = all_df.groupby(['rep_count_x'])[['latency', 'msg_size_x', 'time_diff', 'consume_time']].agg({
        'latency': 'mean',
        'time_diff': 'mean',
        'msg_size_x' : 'mean',
        'consume_time': 'count'
    }).reset_index()
    diff_df.rename(columns={'consume_time': 'records'}, inplace=True)
    diff_df['throughput'] = ((diff_df['records'] * diff_df['msg_size_x']) / diff_df['time_diff']) / 1000000

    print(diff_df)
    smooth_plot(diff_df['rep_count_x'], [diff_df['latency']],
                "Rep factor vs Latency", "rep factor", "Latency (ms)", "repVtp-2")

def consumers_vs_throughput():
    '''
    prod_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/consVtp/producer-payloads-consVtp-trend-4-12-1.csv")
    cons_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/consVtp/consumer-payloads-consVtp-trend-4-12-1.csv")

    all_df = prod_df.merge(cons_df, on="message_uuid")
    all_df['latency'] = all_df['consume_time'] - all_df['produce_time_x']
    all_df['time_diff'] = all_df['produce_time_x'].diff() / 1000

    diff_df = all_df.groupby(['consumer_count'])[['latency', 'msg_size_x', 'time_diff', 'consume_time']].agg({
        'latency': 'mean',
        'time_diff': 'mean',
        'msg_size_x' : 'mean',
        'consume_time': 'count'
    }).reset_index()
    diff_df.rename(columns={'consume_time': 'records'}, inplace=True)
    diff_df['throughput'] = ((diff_df['records'] * diff_df['msg_size_x']) / diff_df['time_diff']) / 1000000
    diff_df = diff_df[(diff_df['consumer_count'] != 11)]
    smooth_plot(diff_df['consumer_count'], [diff_df['throughput']],
                "Consumers vs Latency", "Consumers", "Throughput (mbps)", "consVtp-2")

    print(diff_df)
    '''

    prod_broke_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/consVtp/producers-consVtp-trend-4-12-1.csv")
    cons_broke_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/consVtp/consumers-consVtp-trend-4-12-1.csv")
    broke_df = prod_broke_df.merge(cons_broke_df, on="time")

    broke_df['rebalance_dict'] = broke_df['cgrp'].apply(
        lambda x: json.loads(x.replace("'", '"'))
    )

    # Extract a specific attribute (e.g., 'rebalance_cnt') from the dictionaries
    broke_df['rebalance_cnt'] = broke_df['rebalance_dict'].apply(
        lambda x: x.get("rebalance_cnt")
    )

    max_vals = broke_df.groupby('consumer_count')['rebalance_cnt'].max().reset_index(name='rebalance_cnt_max')

    broke_df = pd.merge(broke_df, max_vals, on='consumer_count', how='left')

    agg_df = broke_df.groupby(['consumer_count'])[[ 'rebalance_cnt', 'rx_bytes_y', 'tx_bytes_x','rebalance_cnt_max']].agg({
        'rebalance_cnt': 'mean',
        'rx_bytes_y': 'mean',
        'tx_bytes_x' : 'mean',
        'rebalance_cnt_max': 'mean',
    }).reset_index()
    agg_df['tx_bytes_x'] = (agg_df['tx_bytes_x'] / 100) / 1000
    agg_df['rx_bytes_y'] = (agg_df['rx_bytes_y'] / 100) / 1000
    print(agg_df)

    smooth_plot(agg_df['consumer_count'], [agg_df['tx_bytes_x'], agg_df['rx_bytes_y']],
                "Consumers vs Throughput", "Consumers", "Throughput (kbps)", "consVtp-2")

    smooth_plot(agg_df['consumer_count'], [agg_df['rebalance_cnt']],
                    "Consumers vs Rebalances", "Consumers", "Rebalances", "consVtp-3")
def broker_calculations(all_df):
    pass


def producers_vs_throughput():
    prod_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/prodVtp/producer-payloads-prodVtp-trend-4-12-1.csv")
    cons_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/prodVtp/consumer-payloads-prodVtp-trend-4-12-1.csv")

    all_df = prod_df.merge(cons_df, on="message_uuid")
    all_df['latency'] = all_df['consume_time'] - all_df['produce_time_x']
    all_df['time_diff'] = all_df['produce_time_x'].diff() / 1000

    diff_df = all_df.groupby(['producer_count_x'])[['latency', 'msg_size_x', 'time_diff', 'consume_time']].agg({
        'latency': 'mean',
        'time_diff': 'mean',
        'msg_size_x' : 'mean',
        'consume_time': 'count'
    }).reset_index()
    diff_df.rename(columns={'consume_time': 'records'}, inplace=True)
    diff_df['throughput'] = ((diff_df['records'] * diff_df['msg_size_x']) / diff_df['time_diff']) / 1000000


    diff_df = diff_df.fillna(0)

    diff_df = diff_df[(diff_df['producer_count_x'] < 40 )]

    print(diff_df)
    #smooth_plot(diff_df['producer_count_x'], [diff_df['throughput']],
                #"Producers vs Throughput", "Producers", "Throughput (mbps)", "prodVtp-3")

    prod_broke_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/prodVtp/producers-prodVtp-trend-4-12-1.csv")
    cons_broke_df = pd.read_csv(
        f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/prodVtp/consumers-prodVtp-trend-4-12-1.csv")
    broke_df = prod_broke_df.merge(cons_broke_df, on="time")

    agg_df = broke_df.groupby(['producer_count'])[ ['rx_bytes_y', 'tx_bytes_x']].agg({
        'rx_bytes_y': 'mean',
        'tx_bytes_x': 'mean',
    }).reset_index()
    agg_df['tx_bytes_x'] = (agg_df['tx_bytes_x'] / 100) / 1000
    agg_df['rx_bytes_y'] = (agg_df['rx_bytes_y'] / 100) / 1000
    print(agg_df)

    smooth_plot(agg_df['producer_count'], [agg_df['tx_bytes_x'], agg_df['rx_bytes_y']],
                "Producer vs Throughput", "Producers", "Throughput (kbps)", "prodVtp-4")




if __name__ == '__main__':
    # msg_size_vs_throughput()
    # part_vs_throughput()
    # rep_factor_vs_throughput()
    # consumers_vs_throughput()
    producers_vs_throughput()