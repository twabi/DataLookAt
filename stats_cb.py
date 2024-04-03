import json
import pandas as pd

def hu_export():

    hu_table_df = pd.read_csv('C:/Users/itwab/Downloads/Video/hu-export-52.csv')

    hu_table_df.columns = ["data_id","cpu_usage","humidity","light","memory_usage","noise","payload_num","payload_size",
                           "payload_time","pi_id","pressure","rx_time","temperature","wifi_bit_rate","wifi_quality","wifi_signal"]


    filtered_df = hu_table_df.loc[hu_table_df['pi_id'] == "H30-4"]
    hu_grup_num = filtered_df.groupby(["payload_num"])[["payload_time", "rx_time"]].mean()
    hu_grup_num["diff_pt"] = hu_grup_num["payload_time"].diff()
    hu_grup_num["diff_rx"] = hu_grup_num["rx_time"].diff()
    filter_diff = hu_grup_num.loc[hu_grup_num['diff_pt'] != 0]
    filter_diff["delay"] = filter_diff["rx_time"] - filter_diff["payload_time"]

    pd.options.display.float_format = '{:.0f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    print(filter_diff.iloc[0:200])

def stats_cb():
    producer_df = pd.read_csv('C:/Users/itwab/Downloads/producer_24.csv')
    brokers_df = pd.read_csv('C:/Users/itwab/Downloads/brokers_24.csv')
    partition_df = pd.read_csv('C:/Users/itwab/Downloads/partitions_24.csv')

    part_grup = partition_df.groupby(["broker", "partition"])[["txmsgs", "msgs", "rxmsgs", "msgs_inflight"]].mean()

    broke_grup = brokers_df.groupby(["nodeid"])[["int_latency_avg", "rxidle", "rtt_avg", "rx", "tx",
                                                 "outbuf_cnt", "outbuf_msg_cnt", "waitresp_cnt", "waitresp_msg_cnt"]].mean()

    producer_grup = producer_df[["age", "msg_cnt", "msg_size", "rx", "tx", "txmsgs", "rxmsgs"]].mean()

    pd.options.display.float_format = '{:.0f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(broke_grup)
    print(producer_grup)

    broke_grup.reset_index().to_csv('C:/Users/itwab/Downloads/broke_grup_51.csv', index=False)
    part_grup.reset_index().to_csv('C:/Users/itwab/Downloads/part_grup_51.csv', index=False)
    producer_grup.reset_index().to_csv('C:/Users/itwab/Downloads/producer_grup_51.csv', index=False)




if __name__ == '__main__':
    stats_cb()