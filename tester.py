import pandas as pd
import json

# Initialize lists to store the dictionaries
brokers_list = []
partition_list = []
producer_list = []

with open('D:/Downloads/txt_metrics/stats_cb_h30-3_2024-02-05.txt', 'r') as f:
    line_number = 0
    for line in f:  # Read line by line
        line_number += 1
        print(line)
        try:
            line_dict = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {line_number}: {e}")
            print(line[:150])  # Print the first 150 characters of the problematic line
            continue

        brokers_dict = line_dict.get('brokers', {})
        part_dict = line_dict.get('topics', {})
        partition_dict = part_dict.get("hu-iot", None)

        # Remove keys to prepare the producer dict
        producer_dict = {k: v for k, v in line_dict.items() if k not in ['brokers', 'topics']}
        if producer_dict:
            producer_list.append(producer_dict)

        if partition_dict:
            for value in partition_dict.get('partitions', {}).values():
                value['time'] = line_dict['time']
                partition_list.append(value)

        for v in brokers_dict.values():
            v['time'] = line_dict['time']
            v['int_latency_avg'] = v.get('int_latency', {}).get('avg', None)
            v['int_latency_cnt'] = v.get('int_latency', {}).get('cnt', None)
            v['rtt_avg'] = v.get('rtt', {}).get('avg', None)
            v['rtt_cnt'] = v.get('rtt', {}).get('cnt', None)
            v['throttle_avg'] = v.get('throttle', {}).get('avg', None)
            v['throttle_cnt'] = v.get('throttle', {}).get('cnt', None)

            # Remove keys that are no longer needed
            for key in ['throttle', 'rtt', 'int_latency', 'toppars']:
                v.pop(key, None)

            brokers_list.append(v)

# Convert lists to DataFrames
brokers_df = pd.DataFrame(brokers_list)
partition_df = pd.DataFrame(partition_list)
producer_df = pd.DataFrame(producer_list)

# Save to CSV
producer_df.to_csv('C:/Users/itwab/Downloads/producer_24.csv', index=False)
brokers_df.to_csv('C:/Users/itwab/Downloads/brokers_24.csv', index=False)
partition_df.to_csv('C:/Users/itwab/Downloads/partitions_24.csv', index=False)
