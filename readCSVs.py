import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from statsmodels.stats.anova import AnovaRM
import statsmodels.stats.multicomp as multi
from statsmodels.graphics.factorplots import interaction_plot



def change_to_ms(x):
    return float(x) * 1000


def change_to_kb(x):
    return float(x) / 1000


def plot_overalls(overall_pd):
    #overall_pd["avg_latency"] = overall_pd["avg_latency"].apply(change_to_ms)
    #overall_pd["avg_jitter"] = overall_pd["avg_jitter"].apply(change_to_ms)
    overall_pd["avg_throughput"] = overall_pd["avg_throughput"].apply(change_to_kb)
    overall_pd["avg_throughput"] = overall_pd["avg_throughput"] / overall_pd["sequence_count"]

    overall_pd["avg_msg_size"] = overall_pd["avg_msg_size"].apply(change_to_kb)

    #df_grup = overall_pd.groupby(["partitions"]).mean()
    df_grup_2 = overall_pd.groupby(["brokers", "partitions"]).mean()

    #df_res = overall_pd.groupby(["resolution"])[["unique_fps", "nonunique_fps"]].mean()

    #print(df_grup)
    #fig, ax = plt.subplots()

    '''
    

    df_res.plot(
        y=["avg_throughput", "avg_msg_size"],
        title='Resolution vs Throughput',
        kind="bar", figsize=(10, 10), legend=True)

    df_res.plot(  # x=["brokers", "partitions"],
        y=["nonunique_fps"],
        title='Resolution vs Average FPS (at receiver)',
        #yerr="avg_jitter", capsize=5,
        color="grey",
        kind="bar", figsize=(10, 10), legend=True)

    x_new_1 = np.linspace(df_grup.index.min(), df_grup.index.max(), 300)
    spl_delay_1 = make_interp_spline(df_grup.index, df_grup["avg_throughput"], k=1)
    y_smooth_delay = spl_delay_1(x_new_1)
    #spl_jitter_1 = make_interp_spline(df_grup.index, df_grup["avg_jitter"], k=1)
    #y_smooth_jitter = spl_jitter_1(x_new_1)

    #plt.plot(x_new_1, y_smooth_delay)
    #plt.fill_between(x_new_1, y_smooth_delay - y_smooth_jitter, y_smooth_delay + y_smooth_jitter,
     #                alpha=0.4)
'''
    df_grup_2.plot(
        y=["avg_latency"],
        #title='Brokers, Partitions vs Latency',
        yerr="avg_jitter", capsize=5,
        color="grey", #ax=ax,
        kind="bar", figsize=(10, 10), legend=True)


    #plt.ylim(0, 40000)
    #plt.ylabel('Bytes (kb)')
    #plt.xlabel('Partitions')
    plt.ylabel('Average Latency (ms)', fontsize=19)
    plt.xlabel('Brokers, Partitions', fontsize=19)
    plt.xticks(fontsize=18, rotation=90, ha='right')
    plt.yticks(fontsize=18)
    plt.legend().remove()
    plt.tight_layout()
    #plt.ylabel('Count of Msgs with same Sequence No.')
    #plt.show()
    plt.savefig("C:/Users/itwab/Downloads/image-1.png")

def plot_delay():
    quality_list = ['360', '720']
    durations = ['2-500', '1-1000']

    rand_dfs = []
    for qual in quality_list:
        for duration in durations:
            sender_df = pd.read_csv(
                f'C:/Users/itwab/Downloads/video-data/random/sender-b4-p4-rand-{qual}-{duration}.csv')
            receiver_df = pd.read_csv(
                f'C:/Users/itwab/Downloads/video-data/random/receiver-b4-p4-rand-{qual}-{duration}.csv')

            sender_df.drop(['writer_error_count_total', 'writer_error_count_rate'],
                           inplace=True, axis=1,
                           errors='ignore')
            receiver_df.drop(['reader_error_count_total', 'reader_error_count_rate'], inplace=True, axis=1,
                             errors='ignore')

            group_df1 = sender_df.groupby("sequence_num").count()

            # print(var_name, group_df1['writer_start_time'].mean() - group_df2['reader_start_time'].mean())



            comb_df = sender_df.merge(receiver_df, on="sequence_num")
            var_name = f'{qual}p'

            rand_dfs.append({"title": var_name, "df" : comb_df, "seq":group_df1['writer_start_time'].mean()})

    for df_dict in rand_dfs:
        df = df_dict['df']
        seq = df_dict['seq']
        print(seq)
        df['Delay'] = df['reader_end_time_ms'] - df['writer_end_time_ms']
        df['Jitter'] = abs(df['Delay'].diff())

        print(df.columns)
        df['base_throughput'] = (df['writer_msg_count_rate'] * df['writer_msg_size_avg'])

        print(df)
    plot_compression(rand_dfs)

def plot_compression(dfs):
    print(len(dfs))
    df_1 = dfs[1]['df']
    title_1 = dfs[1]['title']

    df_2 = dfs[3]['df']
    title_2 = dfs[3]['title']

    df_mean_1 = df_1.groupby("writer_comp_qual").mean()
    df_mean_2 = df_2.groupby("writer_comp_qual").mean()


    #df_mean_1 = df_mean_1.drop(df_mean_1[df_mean_1['Delay'] > 2000].index)
    #df_mean_2 = df_mean_2.drop(df_mean_2[df_mean_2['Delay'] > 2000].index)

    print(df_mean_1, df_mean_2)

    x_new_1 = np.linspace(df_mean_1.index.min(), df_mean_1.index.max(), 300)
    spl_delay_1 = make_interp_spline(df_mean_1.index, df_mean_1["base_throughput"], k=1)
    spl_jitter_1 = make_interp_spline(df_mean_1.index, df_mean_1["Jitter"], k=1)
    y_smooth_delay_1 = spl_delay_1(x_new_1)
    y_smooth_jitter_1 = spl_jitter_1(x_new_1)

    x_new_2 = np.linspace(df_mean_2.index.min(), df_mean_2.index.max(), 300)
    spl_delay_2 = make_interp_spline(df_mean_2.index, df_mean_2["base_throughput"], k=1)
    spl_jitter_2 = make_interp_spline(df_mean_2.index, df_mean_2["Jitter"], k=1)
    y_smooth_delay_2 = spl_delay_2(x_new_2)
    y_smooth_jitter_2 = spl_jitter_2(x_new_2)

    # plot the data
    plt.plot(x_new_1, y_smooth_delay_1, label=title_1)

    plt.plot(x_new_2, y_smooth_delay_2, label=title_2)

    # add shaded region for the Jitter column
    #plt.fill_between(x_new_1, y_smooth_delay_1 - y_smooth_jitter_1, y_smooth_delay_1 + y_smooth_jitter_1,
     #                alpha=0.4)
    #plt.fill_between(x_new_2, y_smooth_delay_2 - y_smooth_jitter_2, y_smooth_delay_2 + y_smooth_jitter_2,
     #                alpha=0.4)

    plt.xlabel('Compression Quality')
    plt.ylabel('Delay (ms)')

    plt.legend()

    #plt.ylim(0, 2000)

    plt.title("Compression Quality vs Delay")

    # display plot
    plt.show()
    #plt.savefig("C:/Users/itwab/Downloads/Video/image-5.png")

def find_closest_smaller(value, values):
    smaller_values = values[values < value]
    if len(smaller_values) > 0:
        closest_smaller = smaller_values.max()
        return value - closest_smaller
    else:
        return pd.NA


def main():
    quality_list = ['360', '720']
    brokers = ['1', '2', '4', '8', '12']
    partitions = ['1', '2', '4', '8', '10']

    df_names = []

    broker_col = []
    part_col = []
    qual_col = []
    seq_count = []
    fps_array = []
    unique_fps = []

    for qual in quality_list:
        for broker in brokers:
            for part in partitions:

                if qual == '720':
                    duration = 150
                elif qual == 'none':
                    duration = 300
                else:
                    duration = 100

                sender_df = pd.read_csv(
                    f'C:/Users/itwab/Downloads/video-data/seq_num/{qual}/sender-b{broker}-p{part}-30-{qual}-{duration}.csv')
                receiver_df = pd.read_csv(
                    f'C:/Users/itwab/Downloads/video-data/seq_num/{qual}/receiver-b{broker}-p{part}-30-{qual}-{duration}.csv')
                sender_df.drop(['writer_error_count_total', 'writer_error_count_rate', 'writer_comp_qual'],
                               inplace=True, axis=1,
                               errors='ignore')
                receiver_df.drop(['reader_error_count_total', 'reader_error_count_rate'], inplace=True, axis=1,
                                 errors='ignore')

                var_name = f'df-{broker}-{part}-{qual}'
                df_names.append(var_name)
                #pd.set_option('display.max_rows', None)
                #pd.set_option('display.max_columns', None)

                group_df1 = sender_df.groupby("sequence_num").count()
                group_df2 = receiver_df.groupby("sequence_num").count()

                df_test = sender_df.groupby("sequence_num")["writer_msg_count_rate"].count().reset_index(name="count")

                total_duration_sender = sender_df["writer_end_time"].iloc[-1] - sender_df["writer_start_time"].iloc[-1]
                unique_num = sender_df["sequence_num"].nunique()
                total_count = sender_df["sequence_num"].count()

                total_duration_reader = receiver_df["reader_end_time"].iloc[-1] - receiver_df["reader_start_time"].iloc[-1]
                unique_num_2 = receiver_df["sequence_num"].nunique()
                total_count_2 = receiver_df["sequence_num"].count()

                sender_df["sender_unique_fps"] = unique_num/total_duration_sender
                sender_df["sender_nonunique_fps"] = total_count/total_duration_sender

                receiver_df["reader_unique_fps"] = unique_num_2 / total_duration_reader
                receiver_df["reader_nonunique_fps"] = total_count_2 / total_duration_reader

                #print(var_name, group_df1['writer_start_time'].mean() - group_df2['reader_start_time'].mean())

                seq_count.append(group_df1['writer_start_time'].mean())
                broker_col.append(broker)
                part_col.append(part)
                qual_col.append(qual)
                unique_fps.append(receiver_df["reader_unique_fps"].mean())
                fps_array.append(receiver_df["reader_nonunique_fps"].mean())



                globals()[var_name] = pd.merge(sender_df, receiver_df, on='sequence_num') #sender_df.join(receiver_df, how="inner")

                #print(globals()[var_name])

    summary_df = pd.DataFrame()
    summary_df['resolution'] = qual_col
    summary_df['brokers'] = broker_col
    summary_df['partitions'] = part_col
    summary_df['sequence_count'] = seq_count
    summary_df['unique_fps'] = unique_fps
    summary_df['nonunique_fps'] = fps_array

    merged_df_names = set(df_names)
    for name in merged_df_names:
        temp_df = globals()[name]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        labels = name.split("-")

        temp_df['latency'] = temp_df['reader_end_time_ms'] - temp_df['writer_end_time_ms']
        temp_df['jitter'] = abs(temp_df['latency'].diff())
        temp_df = temp_df.groupby('sequence_num').mean()
        row_index = summary_df[(summary_df['resolution'] == labels[3])
                               & (summary_df['brokers'] == labels[1]) & (
                                           summary_df['partitions'] == labels[2])].index.tolist()[0]

        print(name, temp_df["reader_unique_fps"].mean())

        summary_df.loc[row_index, 'avg_latency'] = temp_df['latency'].mean()
        summary_df.loc[row_index, 'avg_jitter'] = temp_df['jitter'].mean()
        summary_df.loc[row_index, 'avg_msg_size'] = temp_df['reader_msg_size_avg'].mean()
        summary_df.loc[row_index, 'avg_count_rate'] = temp_df['reader_msg_count_rate'].mean()
        summary_df['avg_throughput'] = summary_df['avg_count_rate'] * summary_df['avg_msg_size']


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(summary_df)
    #plot_overalls(summary_df)
    summary_df.to_csv('C:/Users/itwab/Downloads/test_summary_72.csv', index=False)

def plot_pickle():
    brokers = ['1', '2', '4', '8', '12']
    partitions = ['1', '2', '4', '8', '10']

    none_dfs = []
    broker_col = []
    part_col = []
    seq_count = []
    for broker in brokers:
        for part in partitions:
            sender_df = pd.read_csv(
                f'C:/Users/itwab/Downloads/video-data/seq_num/none/sender-b{broker}-p{part}-none-360-300.csv')
            receiver_df = pd.read_csv(
                f'C:/Users/itwab/Downloads/video-data/seq_num/none/receiver-b{broker}-p{part}-none-360-300.csv')
            sender_df.drop(['writer_error_count_total', 'writer_error_count_rate', 'writer_comp_qual'],
                           inplace=True, axis=1,
                           errors='ignore')
            receiver_df.drop(['reader_error_count_total', 'reader_error_count_rate'], inplace=True, axis=1,
                             errors='ignore')

            group_df1 = sender_df.groupby("sequence_num").count()

            seq_count.append(group_df1['writer_start_time'].mean())

            broker_col.append(broker)
            part_col.append(part)
            comb_df = sender_df.merge(receiver_df, on="sequence_num")
            var_name = f'none-{broker}-{part}'
            none_dfs.append({"title": var_name, "df" : comb_df})


    summary_df = pd.DataFrame()
    summary_df['brokers'] = broker_col
    summary_df['partitions'] = part_col
    summary_df['sequence_count'] = seq_count
    for df_dict in none_dfs:
        temp_df = df_dict['df']
        title = df_dict['title']

        labels = title.split("-")

        temp_df['latency'] = temp_df['reader_end_time_ms'] - temp_df['writer_end_time_ms']
        temp_df['jitter'] = abs(temp_df['latency'].diff())
        temp_df = temp_df.groupby('sequence_num').mean()
        row_index = summary_df[ (summary_df['brokers'] == labels[1]) & (summary_df['partitions'] == labels[2])].index.tolist()[0]

        # print(name, temp_df['latency'].mean())

        summary_df.loc[row_index, 'avg_latency'] = temp_df['latency'].mean()
        summary_df.loc[row_index, 'avg_jitter'] = temp_df['jitter'].mean()
        summary_df.loc[row_index, 'avg_msg_size'] = temp_df['reader_msg_size_avg'].mean()
        summary_df.loc[row_index, 'avg_count_rate'] = temp_df['reader_msg_count_rate'].mean()
        summary_df['avg_throughput'] = summary_df['avg_count_rate'] * summary_df['avg_msg_size']

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(summary_df)
    #summary_df.to_csv('C:/Users/itwab/Downloads/none_summary_52.csv', index=False)




if __name__ == '__main__':
    pd_df = pd.read_csv('C:/Users/itwab/Downloads/test_summary_72.csv')
    plot_overalls(pd_df)
    #plot_delay()
    #main()

    #plot_pickle()
    #print(pd_df)
    #print(AnovaRM(data=pd_df, depvar='avg_throughput',
     #             subject='brokers', within=['partitions']).fit())

    #results = multi.pairwise_tukeyhsd(pd_df['partitions'], pd_df['avg_throughput'])

    # Output the results
    #print(results)