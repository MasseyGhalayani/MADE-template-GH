import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi


class DataLoader:
    def __init__(self, data_path='./data', datasets=None):
        self.data_path = data_path
        self.datasets = datasets if datasets else []
        self.csv_files = []

    def download_data(self):
        api = KaggleApi()
        api.authenticate()
        for dataset in self.datasets:
            api.dataset_download_files(dataset, path=self.data_path, unzip=True)
        self.csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if len(self.csv_files) < 2:
            raise ValueError("Not enough CSV files.")
        return self.csv_files

    def load_data(self):
        df1 = pd.read_csv(os.path.join(self.data_path, self.csv_files[0]), encoding='cp1252')
        df2 = pd.read_csv(os.path.join(self.data_path, self.csv_files[3]))
        return df1, df2


class DataProcessor:
    def __init__(self, temperature_data, seaice_data):
        self.temperature_data = temperature_data
        self.seaice_data = seaice_data

    def process_temperature_data(self):
        global_temp = self.temperature_data[self.temperature_data['Area'] == 'World'][
            ['Months', 'Element', 'Unit'] + [col for col in self.temperature_data.columns if col.startswith('Y')]]
        global_temp = global_temp.melt(id_vars=['Months', 'Element', 'Unit'], var_name='Year', value_name='Value')
        global_temp['Year'] = global_temp['Year'].str.extract('(\\d+)').astype(int)
        global_temp_anomalies = global_temp[
            (global_temp['Element'] == 'Temperature change') & (global_temp['Months'] == 'Meteorological year')]
        global_temp_avg = global_temp_anomalies.groupby('Year')['Value'].mean().reset_index()
        global_temp_avg.columns = ['Year', 'Temperature_Anomaly']
        return global_temp_avg

    def process_seaice_data(self):
        self.seaice_data.columns = self.seaice_data.columns.str.strip()

        if {'Year', 'Month', 'Day'}.issubset(self.seaice_data.columns):
            self.seaice_data['Date'] = pd.to_datetime(self.seaice_data[['Year', 'Month', 'Day']])
        else:
            self.seaice_data['Date'] = pd.to_datetime(self.seaice_data['Date'])

        self.seaice_data['Year'] = self.seaice_data['Date'].dt.year
        seaice_yearly_avg = self.seaice_data.groupby('Year')['Extent'].mean().reset_index()
        seaice_yearly_avg.columns = ['Year', 'Sea_Ice_Extent']
        return seaice_yearly_avg

    def merge_data(self, global_temp_avg, seaice_yearly_avg):
        merged_data = pd.merge(global_temp_avg, seaice_yearly_avg, on='Year', how='inner')
        return merged_data

    def calculate_correlation(self, merged_data):
        correlation = merged_data[['Temperature_Anomaly', 'Sea_Ice_Extent']].corr()
        return correlation


class DataVisualizer:
    def __init__(self, global_temp_avg, seaice_yearly_avg, merged_data):
        self.global_temp_avg = global_temp_avg
        self.seaice_yearly_avg = seaice_yearly_avg
        self.merged_data = merged_data

    def plot_data(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        sns.histplot(self.global_temp_avg['Temperature_Anomaly'], bins=20, color='blue', kde=True, ax=axs[0, 0])
        axs[0, 0].set_title('Distribution of Global Temperature Anomalies')
        axs[0, 0].set_xlabel('Temperature Anomaly (°C)')
        axs[0, 0].set_ylabel('Frequency')

        sns.histplot(self.seaice_yearly_avg['Sea_Ice_Extent'], bins=20, color='red', kde=True, ax=axs[0, 1])
        axs[0, 1].set_title('Distribution of Sea Ice Extent')
        axs[0, 1].set_xlabel('Sea Ice Extent (million sq km)')
        axs[0, 1].set_ylabel('Frequency')

        sns.scatterplot(x='Temperature_Anomaly', y='Sea_Ice_Extent', data=self.merged_data, color='green', ax=axs[1, 0])
        axs[1, 0].set_title('Temperature Anomalies vs. Sea Ice Extent')
        axs[1, 0].set_xlabel('Temperature Anomaly (°C)')
        axs[1, 0].set_ylabel('Sea Ice Extent (million sq km)')

        ax2 = axs[1, 1].twinx()
        axs[1, 1].plot(self.merged_data['Year'], self.merged_data['Temperature_Anomaly'], color='blue')
        ax2.plot(self.merged_data['Year'], self.merged_data['Sea_Ice_Extent'], color='red')
        axs[1, 1].set_title('Global Temperature Anomalies and Sea Ice Extent Over Time')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Temperature Anomaly (°C)', color='blue')
        ax2.set_ylabel('Sea Ice Extent (million sq km)', color='red')

        plt.tight_layout()
        plt.show()

    def plot_future(self, future_predictions):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Predicted Temperature Anomaly (°C)', color='tab:blue')
        ax1.plot(future_predictions['Year'], future_predictions['Predicted_Temperature_Anomaly'], color='tab:blue',
                 label='Temperature Anomaly')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Predicted Sea Ice Extent (million sq km)', color='tab:red')
        ax2.plot(future_predictions['Year'], future_predictions['Predicted_Sea_Ice_Extent'], color='tab:red',
                 label='Sea Ice Extent')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('Future Predictions of Temperature Anomaly and Sea Ice Extent')
        plt.savefig("future-predict.png")
        plt.show()



def prepare_data(global_temp_avg, seaice_yearly_avg):
    merged_data = pd.merge(global_temp_avg, seaice_yearly_avg, on='Year', how='inner')

    X_temp = merged_data[['Year']]
    y_temp = merged_data['Temperature_Anomaly']

    X_ice = merged_data[['Temperature_Anomaly']]
    y_ice = merged_data['Sea_Ice_Extent']

    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2,
                                                                            random_state=42)
    X_ice_train, X_ice_test, y_ice_train, y_ice_test = train_test_split(X_ice, y_ice, test_size=0.2, random_state=42)

    scaler_temp = StandardScaler().fit(X_temp_train)
    X_temp_train_scaled = scaler_temp.transform(X_temp_train)
    X_temp_test_scaled = scaler_temp.transform(X_temp_test)

    scaler_ice = StandardScaler().fit(X_ice_train)
    X_ice_train_scaled = scaler_ice.transform(X_ice_train)
    X_ice_test_scaled = scaler_ice.transform(X_ice_test)

    return X_temp_train_scaled, X_temp_test_scaled, y_temp_train, y_temp_test, X_ice_train_scaled, X_ice_test_scaled, y_ice_train, y_ice_test, scaler_temp, scaler_ice


def train_and_evaluate(X_train, X_test, y_train, y_test, models):
    best_model = None
    best_mse = float('inf')
    for model in models:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"{model.__class__.__name__} with MSE: {mse}")
        if mse < best_mse:
            best_mse = mse
            best_model = model
    return best_model, best_mse


def predict_future(years, temp_model, ice_model, scaler_temp, scaler_ice):
    future_years = pd.DataFrame({'Year': years})
    future_years_scaled = scaler_temp.transform(future_years)
    future_temp_anomalies = temp_model.predict(future_years_scaled)
    future_temp_anomalies_df = pd.DataFrame({'Temperature_Anomaly': future_temp_anomalies})
    future_temp_anomalies_scaled = scaler_ice.transform(future_temp_anomalies_df)
    future_sea_ice_extent = ice_model.predict(future_temp_anomalies_scaled)

    future_predictions = pd.DataFrame({
        'Year': years,
        'Predicted_Temperature_Anomaly': future_temp_anomalies,
        'Predicted_Sea_Ice_Extent': future_sea_ice_extent
    })

    return future_predictions


def main():
    data_path = './data'
    datasets = ["nsidcorg/daily-sea-ice-extent-data", "sevgisarac/temperature-change"]

    data_loader = DataLoader(data_path=data_path, datasets=datasets)
    data_loader.download_data()
    temperature_data, seaice_data = data_loader.load_data()

    data_processor = DataProcessor(temperature_data, seaice_data)
    global_temp_avg = data_processor.process_temperature_data()
    seaice_yearly_avg = data_processor.process_seaice_data()

    X_temp_train_scaled, X_temp_test_scaled, y_temp_train, y_temp_test, X_ice_train_scaled, X_ice_test_scaled, y_ice_train, y_ice_test, scaler_temp, scaler_ice = prepare_data(
        global_temp_avg, seaice_yearly_avg)

    models = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
    ]

    best_temp_model, best_temp_mse = train_and_evaluate(X_temp_train_scaled, X_temp_test_scaled, y_temp_train,
                                                        y_temp_test, models)
    print(f"Best Temperature Model: {best_temp_model} with MSE: {best_temp_mse}")

    best_ice_model, best_ice_mse = train_and_evaluate(X_ice_train_scaled, X_ice_test_scaled, y_ice_train, y_ice_test,
                                                      models)
    print(f"Best Sea Ice Model: {best_ice_model} with MSE: {best_ice_mse}")

    future_years = np.arange(2024, 2034)
    future_predictions = predict_future(future_years, best_temp_model, best_ice_model, scaler_temp, scaler_ice)
    print(future_predictions)

    merged_data = data_processor.merge_data(global_temp_avg, seaice_yearly_avg)
    data_visualizer = DataVisualizer(global_temp_avg, seaice_yearly_avg, merged_data)
    data_visualizer.plot_data()

    data_visualizer.plot_future(future_predictions)

    correlation = data_processor.calculate_correlation(merged_data)
    print(correlation)


if __name__ == '__main__':
    main()