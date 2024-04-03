import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams['timezone'] = 'Asia/Tokyo'


def barred(producer_df, consumer_df, producer_payload, consumer_payload, name):
    frames = [producer_df, consumer_df]
    result_df = pd.concat(frames)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sum_tuple = create_part_summary(consumer_payload, producer_payload)
    print(sum_tuple)

    groupedResult = result_df.fillna(0).groupby(['type'])[['age', 'tx', 'txmsgs', 'rx', 'rxmsgs', 'msg_cnt', 'msg_size',
                                                           'producer_count', 'consumer_count']].mean().reset_index()

    groupedResult['age'] = groupedResult['age'] / 1000000
    print(groupedResult)

    parts = name.split('-')[2]
    inter_df_data = {
        "partitions": [parts],
        "producers": [groupedResult['producer_count'].iloc[-1]],  # Assuming you want the last entry
        "consumers": [groupedResult['consumer_count'].iloc[0]],  # Assuming you want the first entry
        "rep factor": [1],
        "avg_latency": [sum_tuple[0]],
        "avg_throughput": [sum_tuple[2]],
    }
    inter_df = pd.DataFrame(inter_df_data)

    print(inter_df)
    return inter_df

    # groupedResult.to_csv(f'C:/Users/itwab/Downloads/grouped-run-{name}.csv', index=False)

def create_part_summary(consumer_pd, producer_pd):
    cols = ["producer_id",	"message_num",	"message_uuid",	"produce_time",	"message" ]  # Replace with your actual column names

    # Assign column names to the DataFrame
    #producer_pd.columns = cols
    print(producer_pd.columns)
    if "Unnamed" in producer_pd.columns[0] :
        producer_pd.columns = cols
    df_all = producer_pd.merge(consumer_pd.drop_duplicates(), on='message_uuid', how='left', indicator=True)

    result_df = df_all[df_all['_merge'] == 'left_only']

    df_all['latency'] = df_all['consume_time'] - df_all['produce_time_x']

    avg_latency = df_all['latency'].mean()
    total_elapsed_time = (df_all['consume_time'].iloc[-1] - df_all['consume_time'].iloc[0]) / 1000

    num_rows = len(df_all)
    avg_msgs = num_rows / total_elapsed_time

    # print(avg_latency, total_elapsed_time, avg_msgs)
    return [avg_latency, total_elapsed_time, avg_msgs, num_rows]


def main(producer_df, consumer_df, index):

    all_df = producer_df.merge(consumer_df, on="time")

    time_df = all_df.groupby(['time'])[['producer_count', 'consumer_count']].mean().reset_index()
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

    #plt.show()
    plt.savefig(f"C:/Users/itwab/Downloads/8-03/run-{index}.png")

def rand_process( prod_pd, cons_pd, prod_payloads, cons_payloads, name):
    pass

if __name__ == '__main__':

    summary_df = pd.DataFrame()
    partitions = ['1', '10', '20', '30', '40', '50']
    producers = ['n']
    consumers = ['1']
    d_frames = []

    for part in partitions:
        for prod in producers:
            for cons in consumers:

                try:
                    '''
                    prod_df = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/19-03/producers-{prod}p{cons}c-4-{part}-100.csv")
                    cons_df = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/19-03/consumers-{prod}p{cons}c-4-{part}-100.csv")
                    consumer_payloads = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/19-03/consumer-payloads-{prod}p{cons}c-4-{part}-100.csv")
                    producer_payloads = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/19-03/producer-payloads-{prod}p{cons}c-4-{part}-100.csv")
                    '''
                    prod_df = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/28-03/producers-r-{prod}p{cons}c-4-{part}-300.csv")
                    cons_df = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/28-03/consumers-r-{prod}p{cons}c-4-{part}-300.csv")
                    consumer_payloads = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/28-03/consumer-r-payloads-{prod}p{cons}c-4-{part}-300.csv")
                    producer_payloads = pd.read_csv(f"C:/Users/itwab/Downloads/Advnet/My Research/kafka/Measurements/28-03/producer-r-payloads-{prod}p{cons}c-4-{part}-300.csv")

                    index = f"{prod}-{cons}-{part}"
                    # int_df = barred(prod_df, cons_df, producer_payloads, consumer_payloads, index)
                    # d_frames.append(int_df)
                    rand_process(prod_df, cons_df, producer_payloads, consumer_payloads, index)
                except OSError as e:
                    print("Probably, file doesn't exist: ", e)

    final_df = pd.concat(d_frames).reset_index()
    print(final_df)






