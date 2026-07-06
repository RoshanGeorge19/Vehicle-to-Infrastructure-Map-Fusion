import numpy as np
import pandas as pd

def mahalanobis_distance(p, mean, inv_cov_matrix):
    delta = p - mean  # (p - μ)
    return np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))  # √[(p−μ)ᵀ × Σ⁻¹ × (p−μ)]

def main():
    df = pd.read_csv('G:/Documents/Pycharm Projects/Work_Package_1/src/data_processing/mahalanobis_csv.csv')
    car_df = df[df['Name'].str.contains('CAR', case=False, na=False)]
    node_df = df[df['Name'].str.contains('NODE', case=False, na=False)]

    mu = car_df[['Longitude', 'Latitude', 'Altitude']].mean()
    cov_matrix = car_df[['Longitude', 'Latitude', 'Altitude']].cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mahalanobis_distances = node_df[['Longitude', 'Latitude', 'Altitude']].apply(
        lambda row: mahalanobis_distance(row.values, mu, inv_cov_matrix), axis=1
    )

    print(mahalanobis_distances)

if __name__ == "__main__":
    main()