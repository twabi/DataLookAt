import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import csv
from datetime import datetime
import uuid

def change_to_ms(x):
    return float(x) * 1000
def change_to_kb(x):
    return float(x) / 1000

def plot_delay(df):
    df_mean = df.groupby("writer_comp_qual")[["Delay", "Jitter"]].mean()

    x_new = np.linspace(df_mean.index.min(), df_mean.index.max(), 300)
    spl_delay = make_interp_spline(df_mean.index, df_mean["Delay"], k=3)
    spl_jitter = make_interp_spline(df_mean.index, df_mean["Jitter"], k=3)
    y_smooth_delay = spl_delay(x_new)
    y_smooth_jitter = spl_jitter(x_new)

    # plot the data
    ax = plt.plot(x_new, y_smooth_delay, label="Smoothed Line")

    # add shaded region for the Jitter column
    plt.fill_between(x_new, y_smooth_delay - y_smooth_jitter, y_smooth_delay + y_smooth_jitter,
                     alpha=0.45)

    plt.xlabel('Compression Quality')
    plt.ylabel('Delay (ms)')

    # display plot
    plt.show()
    #plt.savefig("C:/Users/itwab/Downloads/Video-Streams/csvs/rand-fig1-new.png")



def plot_overalls():
    overall_pd = pd.read_csv('C:/Users/itwab/Downloads/Video-Streams/csvs/summary.csv')

    overall_pd["avg latency"] = overall_pd["avg latency"].apply(change_to_ms)
    overall_pd["avg jitter"] = overall_pd["avg jitter"].apply(change_to_ms)
    overall_pd["avg throughput"] = overall_pd["avg throughput"].apply(change_to_kb)
    overall_pd["avg msg size"] = overall_pd["avg msg size"].apply(change_to_kb)

    df_grup = overall_pd.groupby(["brokers", "partitions"])[["avg latency", "avg jitter", "avg throughput"]].mean()

    df_res = overall_pd.groupby(["resolution"])[["avg latency", "avg jitter", "avg throughput", "avg msg size"]].mean()

    print(df_grup)


    df_res.plot(
        y=["avg throughput", "avg msg size"],
        title='Resolution vs Throughput',
        kind="bar", figsize=(10, 10), legend=True)
    '''
    df_grup.plot(
        y=["avg throughput"],
        title='Brokers vs Throughput',
        kind="bar", figsize=(10, 10), legend=True)
        
    df_grup.plot(#x=["brokers", "partitions"],
                y=["avg latency"],
                title='Brokers vs Latency',
                yerr="avg jitter", capsize=5,
                kind="bar", figsize=(10, 10), legend=True)
    
    '''

    plt.ylabel('Bytes (kb)')
    # plt.ylabel('Latency (ms)')
    plt.show()
    # plt.savefig("C:/Users/itwab/Downloads/Video-Streams/csvs/res-g2.png")

def plot_rf():
    rf_df = pd.read_csv('C:/Users/itwab/Downloads/voltage-data.csv')

    df_grup = rf_df.groupby(["Location"])[["Voltage(V)"]].mean()

    rf_grup = rf_df.groupby(["Location"])[["Frequency", "RSSI(dBm)"]].mean()

    '''
    
    '''
    df_grup.plot(
        y=["Voltage(V)"],
        title='Location vs Voltage',
        kind="bar", figsize=(10, 10), legend=True)
    '''
    rf_grup.plot(
        y=["RSSI(dBm)"],
        title='RSSI vs Location',
        kind="bar", figsize=(10, 10), legend=True)
    '''


    xtick_labels = [label.get_text() for label in plt.gca().get_xticklabels()]

    for i, label in enumerate(xtick_labels):
        frags = label.split(" ")
        if str(frags[0]).lower().startswith("blantyre"):
            xtick_labels[i] = "BT-" + str(frags[2])
        elif str(frags[0]).lower().startswith("zomba"):
            xtick_labels[i] = "ZA-" + str(frags[2])
        elif str(frags[0]).lower().startswith("lilongwe"):
            xtick_labels[i] = "LL-" + str(frags[2])
    plt.gca().set_xticklabels(xtick_labels)

    plt.ylabel('Voltage(V)')
    plt.xticks(rotation=15, ha='right')
    plt.savefig("C:/Users/itwab/Downloads/Video/voltage-loc-2.png")
    # plt.show()

def plot_rf_volt():
    rf_df = pd.read_csv('C:/Users/itwab/Downloads/voltage-data.csv')

    df_grup = rf_df.groupby(["Frequency"])[["Voltage(V)"]].mean()
    df_grup_2 = rf_df.groupby(["RSSI(dBm)"])[["Voltage(V)"]].mean()
    df_grup_3 = rf_df.groupby(["Frequency", "RSSI(dBm)"])[["Voltage(V)"]].mean()

    print(df_grup_2)
    df_grup_2.reset_index().sort_values(by=['RSSI(dBm)'], ascending=True).plot(
        x ="RSSI(dBm)",
        y=["Voltage(V)"],
        title='Frequency vs Voltage(V)',
        kind="bar", figsize=(10, 10), legend=True)

    plt.ylabel('Voltage(V)')
    plt.xticks(rotation=15, ha='right')
    plt.savefig("C:/Users/itwab/Downloads/Video/rssi-voltage.png")
    # plt.show()

def plot_power_dev():
    rf_df = pd.read_csv('D:/Downloads/IoT/voltage-data.csv')
    rf_df['Powered Device'].fillna("N/A", inplace=True)
    pw_grup = rf_df.groupby(["Powered Device"])[["Voltage(V)", "Current(mA)", "Frequency", "RSSI(dBm)"]].mean().reset_index()

    def custom_agg(series):
        non_na_values = [val for val in series if val != "N/A"]
        non_na_values = list(set(non_na_values))
        return ", ".join(non_na_values) if non_na_values else "N/A"

    agg_func = {"Powered Device": custom_agg, }
    cols = ["Current(mA)", "Voltage(V)", "Frequency", "RSSI(dBm)"]
    for col in cols:
        agg_func[col] = 'mean'

    # Group by "Location" and aggregate "Powered Device" using the custom aggregation function
    lc_grup = rf_df.groupby("Location")[["Powered Device", "Current(mA)", "Voltage(V)", "Frequency", "RSSI(dBm)"]].agg(agg_func).reset_index()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    fig, ax = plt.subplots()

    #pw_grup.to_csv('C:/Users/itwab/Downloads/data-1.csv')
    #lc_grup.to_csv('C:/Users/itwab/Downloads/data-2.csv')
    pw_grup.dropna(inplace=True)
    print(pw_grup)
    pw_grup.plot(
        x="Powered Device",
        y=["RSSI(dBm)"],
        #title='Signal Strength for Each Powered Device',
        kind="bar", color='red', figsize=(10, 10), legend=True, ax=ax)

    pw_grup.plot(
        x="Powered Device",
        y=["Frequency"],
        #title='Signal Strength for Each Powered Device',
        kind="line", color='grey', figsize=(10, 10), legend=True, ax=ax)


    plt.title("Combo Plot")
    plt.xlabel('Powered Device')
    plt.xticks(rotation=15, ha='right')
    plt.show()
    #plt.savefig("C:/Users/itwab/Downloads/Video/rssi-dev-2.png")

def distance_plot():
    rf_df = pd.read_csv('C:/Users/itwab/Downloads/Video/Hub Dist.csv')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(rf_df)
    #rf_df['Powered Device'].fillna("N/A", inplace=True)
    rf_df.dropna(inplace=True)


    pw_grup = rf_df.groupby(["Powered Device"])[
        ["HubDist"]].mean().reset_index()
    print(pw_grup)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    pw_grup.plot(
        x="Powered Device",
        y=["HubDist"],
        title='Powered Device vs Avg Harvested Distance',
        kind="bar", figsize=(10, 10), legend=True)


    plt.ylabel('Avg Distance (m)')
    plt.xticks(rotation=15, ha='right')
    #plt.show()
    plt.savefig("C:/Users/itwab/Downloads/Video/pw-dist-2.png")

def materials_sort():
    mat_df = pd.read_csv('C:/Users/itwab/Downloads/materials.csv')

    print(mat_df.columns)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    mt_grup = mat_df.groupby("Material")[["Specifications", "Unit", "Unit Price"]].agg(lambda x: list(x)).reset_index()


    #mt_grup.to_csv('C:/Users/itwab/Downloads/materials-edit.csv')

    materials_df = pd.DataFrame(columns=["id", "created_at", "name", "dimension_ids"])
    dimensions_df = pd.DataFrame(columns=["id", "created_at", "dimension", "unit", "price_id", "material_id"])
    prices_df = pd.DataFrame(columns=["id", "created_at", "price_date", "price", "dimension_id", "material_id"])

    for index, row in mt_grup.iterrows():
        mat = row['Material']

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date = datetime.now().strftime('%Y-%m-%d')

        mat_id = uuid.uuid4()
        dim_ids = []
        for i in range(len(row['Specifications'])):
            dimension_id = uuid.uuid4()
            price_id = uuid.uuid4()
            dim = row['Specifications'][i]
            unit = row['Unit'][i]
            u_price = float(row['Unit Price'][i].replace(",", ""))


            dim_ids.append("{0}".format(dimension_id))

            #print(mat, dim, u_price, unit)
            dimensions_df.loc[len(dimensions_df)] = [dimension_id, timestamp, dim, unit, price_id, mat_id]
            prices_df.loc[len(prices_df)] = [price_id, timestamp, date, u_price, dimension_id, mat_id]

        materials_df.loc[len(materials_df)] = [mat_id, timestamp, mat, str(dim_ids).replace("'", '"')]

    print(materials_df)
    materials_df.to_csv('C:/Users/itwab/Downloads/mat-insert.csv', index=False)
    dimensions_df.to_csv('C:/Users/itwab/Downloads/dim-insert.csv',  index=False)
    prices_df.to_csv('C:/Users/itwab/Downloads/price-insert.csv',  index=False)

def calc_eff():
    rf_df = pd.read_csv('D:/Downloads/IoT/voltage-data.csv')
    rf_df['Powered Device'].fillna("N/A", inplace=True)

    def custom_agg(series):
        non_na_values = [val for val in series if val != "N/A"]
        non_na_values = list(set(non_na_values))
        return ", ".join(non_na_values) if non_na_values else "N/A"

    agg_func = {"Powered Device": custom_agg, }
    cols = ["Voltage(V)", "Frequency", "RSSI(dBm)"]
    for col in cols:
        agg_func[col] = 'mean'

    # Group by "Location" and aggregate "Powered Device" using the custom aggregation function
    pw_grup = (rf_df.groupby("Location")[[ "Voltage(V)", "Frequency", "RSSI(dBm)", "Powered Device"]]
               .agg(agg_func).reset_index())


    print(pw_grup)
    pw_grup.to_csv('C:/Users/itwab/Downloads/loc-group.csv',  index=False)

if __name__ == '__main__':
    #df_rand1 = pd.read_csv('C:/Users/itwab/Downloads/Video-Streams/csvs/rand-1.csv')
    #df_rand2 = pd.read_csv('C:/Users/itwab/Downloads/Video-Streams/csvs/rand-2.csv')

    #plot_delay(df_rand1)
    calc_eff()

