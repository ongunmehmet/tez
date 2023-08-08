

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

data = pd.read_csv('/gdrive/My Drive/Colab Notebooks/Baltic Dry Index Historical Data-2008 crisis daily (1).csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

target_col = 'Value'
train_size = int(len(data) * 0.60)
val_size = int(train_size * 0.2)
train_size = train_size - val_size
test_size = len(data) - train_size - val_size

train_data = data.iloc[:train_size]
val_data = train_data.iloc[train_size - val_size:train_size]
test_data = data.iloc[train_size + val_size:]

data['Return'] = data['Value'].pct_change()

# Calculate the mean and standard deviation of returns
mean_return = data['Return'].mean()
std_deviation = data['Return'].std()

# Calculate annualized volatility (assuming 252 trading days in a year)
annualized_volatility = std_deviation * np.sqrt(252)

print("Mean Daily Return:", mean_return)
print("Standard Deviation of Daily Return:", std_deviation)
print("Annualized Volatility:", annualized_volatility)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[[target_col]])
val_data_scaled = scaler.transform(val_data[[target_col]])
test_data_scaled = scaler.transform(test_data[[target_col]])

# Convert the data into appropriate input and output format
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 5  # Number of previous time steps to use as input
X_train, y_train = create_dataset(train_data_scaled, look_back)
X_val, y_val = create_dataset(val_data_scaled, look_back)
X_test, y_test = create_dataset(test_data_scaled, look_back)

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1), kernel_regularizer=L2(0.01)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, y_train, epochs=200, batch_size=32,
          validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Predict using the trained model
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predictions = model.predict(X_test)

# # Inverse transform the predictions to get actual values

test_predictions_inscaled = scaler.inverse_transform(test_predictions)

# Calculate evaluation metrics
# Calculate AIC and BIC
n = len(y_test)  # Number of data points in the test set
residuals = y_test - test_predictions.flatten()  # Residuals

# Calculate the sum of squared residuals
ssr = np.sum(residuals**2)

# Calculate the number of parameters in the model
num_parameters = model.count_params()

# Calculate AIC and BIC
aic = n * np.log(ssr / n) + 2 * num_parameters
bic = n * np.log(ssr / n) + num_parameters * np.log(n)

print(f'AIC: {aic:.2f}')
print(f'BIC: {bic:.2f}')
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f"Test RMSE: {test_rmse}")

train_dates = train_data['Date'].values[look_back:]
test_dates = test_data['Date'].values[look_back:]
df_train_pred = pd.DataFrame({'Date': train_dates, 'Predicted': y_train_actual[:, 0]})
df_test_pred = pd.DataFrame({'Date': test_dates, 'Predicted': test_predictions[:, 0]})
df_test_real = pd.DataFrame({'Date': test_dates, 'Real': test_data[target_col].values[look_back:]})


plt.figure(figsize=(12, 6))
plt.plot(train_dates, y_train_actual, label='Eğitim Verisi')
plt.plot(test_dates, test_predictions_inscaled, label='Tahmin')
plt.plot(test_dates, test_data[target_col].values[look_back:], label='Gerçek Değerler')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.title('LSTM Covid-19 Pandemisi Tahminleme Değerleri vs. Gerçek Değerler')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualize the model architecture
plot_model(model, show_shapes=True, to_file='/gdrive/My Drive/Colab Notebooks/lstm_model.png')  # Save to a file