import pandas as pd


def main():
    quality_list = ['360', '720']
    brokers = ['1', '2', '4', '8', '12']
    partitions = ['1', '2', '4', '8', '10']

    df_names = []

    broker_col = []
    part_col = []
    qual_col = []
    seq_count = []

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

                #print(var_name, group_df1['writer_start_time'].mean() - group_df2['reader_start_time'].mean())

                globals()[var_name] = pd.merge(sender_df, receiver_df, on='sequence_num') #sender_df.join(receiver_df, how="inner")
                globals()[var_name]['latency'] = globals()[var_name][ 'reader_end_time_ms'] - globals()[var_name][ 'writer_end_time_ms']

                print(globals()[var_name].iloc[5000:7000][['sequence_num', 'latency']])

    summary_df = pd.DataFrame()

    merged_df_names = set(df_names)
    for name in merged_df_names:
        temp_df = globals()[name]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        labels = name.split("-")

        #print(labels)
        column_name = f'{labels[3]}-b{labels[1]}-p{labels[2]}'
        #temp_df['latency'] = temp_df['reader_end_time_ms'] - temp_df['writer_end_time_ms']

        #print(temp_df.iloc[:500]['sequence_num'])

        summary_df[column_name] = temp_df.iloc[5000:7000]['latency']



    #print(summary_df.columns)
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #print(summary_df)

    summary_df.to_csv('C:/Users/itwab/Downloads/jasp_summary_52.csv', index=False)


if __name__ == '__main__':
    main()
