import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    file_path = 'G:/Documents/Pycharm Projects/Work_Package_1/data/lidar_data/CAR/Dangan GPS Data - 2022-05-27/2022-05-27-Scenario-2-GPS.csv'
    df = pd.read_csv(file_path)
    df = df[df['Speed'] != 0]
    df['Is_Consecutive'] = df.index.to_series().diff() == 1

    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S.%f')

    reference_time = df['Time'].iloc[0]
    df['Expected_Time'] = reference_time + pd.to_timedelta((df.index - df.index[0]) * 0.5, unit='S')
    df['Time_Error'] = (df['Time'] - df['Expected_Time']).dt.total_seconds()

    # Group by each consecutive collection of data
    df['Group'] = (~df['Is_Consecutive']).cumsum()
    for name, group in df.groupby('Group'):
        result_df = group.copy()
        result_df = result_df.reindex(columns=['Index', 'Is_Consecutive', 'Speed', 'Time', 'Expected_Time', 'Time_Error'])

        # Basic statistics of the time error for each group
        print(f"\nTime Error Statistics for Group {name}:")
        print(result_df['Time_Error'].describe())

        sns.distplot(result_df['Time_Error'])
        plt.title(f'Distribution of Time Error for Group {name}')
        plt.show()

if __name__ == "__main__":
    main()