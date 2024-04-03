import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def change_to_ms(x):
    return float(x) * 1000
def change_to_kb(x):
    return float(x) / 1000

def plot_overalls(overall_pd):


    overall_pd["avg_latency"] = overall_pd["avg_latency"].apply(change_to_ms)
    overall_pd["avg_jitter"] = overall_pd["avg_jitter"].apply(change_to_ms)
    overall_pd["avg_throughput"] = overall_pd["avg_throughput"].apply(change_to_kb)
    overall_pd["avg_msg_size"] = overall_pd["avg_msg_size"].apply(change_to_kb)

    df_grup = overall_pd.groupby(["partitions"])[["avg_latency", "avg_jitter", "avg_throughput"]].mean()

    df_res = overall_pd.groupby(["resolution"])[["avg_latency", "avg_jitter", "avg_throughput", "avg_msg_size"]].mean()

    print(df_grup)



    '''
    df_grup.plot(
        y=["avg_latency"],
        title='Brokers vs Throughput',
        kind="bar", figsize=(10, 10), legend=True)
        
    df_res.plot(
        y=["avg_throughput", "avg_msg_size"],
        title='Resolution vs Throughput',
        kind="bar", figsize=(10, 10), legend=True)

    
'''

    df_res.plot(  # x=["brokers", "partitions"],
        y=["avg_latency"],
        title='Resolution vs Latency',
        yerr="avg_jitter", capsize=5,
        kind="bar", figsize=(10, 10), legend=True)
    plt.ylabel('Bytes (kb)')
    #plt.ylabel('Latency (ms)')
    plt.show()
    #plt.savefig("C:/Users/itwab/Downloads/Video-Streams/csvs/image-6.png")

def find_closest_smaller(value, values):
    smaller_values = values[values < value]
    if len(smaller_values) > 0:
        closest_smaller = smaller_values.max()
        return value - closest_smaller
    else:
        return pd.NA

def main():
    quality_list = ['360', '720', '1080']
    brokers = ['1', '2', '4', '8', '12']
    partitions = ['1', '2', '4', '8', '10']


    df_names = []

    broker_col = []
    part_col = []
    qual_col = []
    for qual in quality_list:
        for broker in brokers:
            for part in partitions:
                sender_df = pd.read_csv(f'C:/Users/itwab/Downloads/video-data/{qual}/sender-b{broker}-p{part}-30-{qual}-250.csv')
                receiver_df = pd.read_csv(f'C:/Users/itwab/Downloads/video-data/{qual}/receiver-b{broker}-p{part}-30-{qual}-250.csv')
                sender_df.drop(['writer_error_count_total', 'writer_error_count_rate', 'writer_comp_qual'], inplace=True, axis=1,
                             errors='ignore')
                receiver_df.drop(['reader_error_count_total', 'reader_error_count_rate'], inplace=True, axis=1,
                               errors='ignore')

                var_name = f'df-{broker}-{part}-{qual}'
                df_names.append(var_name)

                broker_col.append(broker)
                part_col.append(part)
                qual_col.append(qual)
                globals()[var_name] = sender_df.join(receiver_df, how="inner")

    summary_df = pd.DataFrame()
    summary_df['resolution'] = qual_col
    summary_df['brokers'] = broker_col
    summary_df['partitions'] = part_col

    merged_df_names = set(df_names)
    for name in merged_df_names:
        temp_df = globals()[name]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        labels = name.split("-")

        temp_df['latency'] = temp_df['writer_end_time'].apply(
            lambda x: find_closest_smaller(x, temp_df['reader_end_time']))

        temp_df['jitter'] = abs(temp_df['latency'].diff())

        row_index = summary_df[(summary_df['resolution'] == labels[3])
                               & (summary_df['brokers'] == labels[1]) & (summary_df['partitions'] == labels[2])].index.tolist()[0]

        print(name, temp_df['latency'].mean())

        summary_df.loc[row_index, 'avg_latency'] = temp_df['latency'].mean()
        summary_df.loc[row_index, 'avg_jitter'] = temp_df['jitter'].mean()
        summary_df.loc[row_index, 'avg_msg_size'] = temp_df['reader_msg_size_avg'].mean()
        summary_df.loc[row_index, 'avg_count_rate'] = temp_df['reader_msg_count_rate'].mean()
        summary_df['avg_throughput'] = summary_df['avg_count_rate'] * summary_df['avg_msg_size']

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(summary_df)
    #plot_overalls(summary_df)
    summary_df.to_csv('C:/Users/itwab/Downloads/test_summary_42.csv', index=False)

if __name__ == '__main__':
    pd_df = pd.read_csv('C:/Users/itwab/Downloads/test_summary_42.csv')
    plot_overalls(pd_df)

    #main()