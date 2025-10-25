# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 25-10-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Tesla.csv')
data['Date'] = pd.to_datetime(data['Date']) # Corrected column name to 'Date'

plt.plot(data['Date'], data['Close']) # Changed to plot 'Close' price
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Tesla Close Price Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Check stationarity of the 'Close' price
check_stationarity(data['Close'])

# Plot ACF and PACF for 'Close' price
plot_acf(data['Close'])
plt.show()
plot_pacf(data['Close'])
plt.show()

# Using 'Close' price for SARIMA model
sarima_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions (Close Price)')
plt.legend()
plt.show()
```

### OUTPUT:
<img width="916" height="720" alt="image" src="https://github.com/user-attachments/assets/77758993-4e21-4d91-92e8-aba87cea012b" />
<img width="724" height="561" alt="image" src="https://github.com/user-attachments/assets/8dafcb26-5037-4059-b222-460f1c244281" />
<img width="741" height="580" alt="image" src="https://github.com/user-attachments/assets/196346d9-eeb3-4f3e-b79c-a17b1c295d30" />
<img width="873" height="574" alt="image" src="https://github.com/user-attachments/assets/b6d327af-939b-464c-8eb5-2d139fee5258" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
