import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Wczytanie danych z pliku CSV
file_path = 'AMZN.csv'
data = pd.read_csv(file_path)

# Konwersja kolumny 'Date' na datetime
data['Date'] = pd.to_datetime(data['Date'])

# Ustawienie daty jako indeks
data.set_index('Date', inplace=True)

# Wyświetlenie pierwszych kilku wierszy
print(data.head())

# Obliczanie średnich ruchomych (7 i 30 dni)
data['MA_7'] = data['Adj Close'].rolling(window=7).mean()
data['MA_30'] = data['Adj Close'].rolling(window=30).mean()

# Wizualizacja wykresu cen i średnich ruchomych
plt.figure(figsize=(10, 6))
plt.plot(data['Adj Close'], label='Cena skorygowana')
plt.plot(data['MA_7'], label='Średnia ruchoma 7 dni')
plt.plot(data['MA_30'], label='Średnia ruchoma 30 dni')
plt.legend()
plt.title('Średnie ruchome dla AMZN')
plt.grid(True)
plt.show()

# Rysowanie wykresów ACF i PACF
plt.figure(figsize=(12, 6))
plot_acf(data['Adj Close'].dropna(), lags=30, ax=plt.gca())
plt.title("Autokorelacja (ACF)")
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data['Adj Close'].dropna(), lags=30, ax=plt.gca(), method='ywm')
plt.title("Częściowa autokorelacja (PACF)")
plt.show()

# Dekompozycja szeregów czasowych
result = seasonal_decompose(data['Adj Close'].dropna(), model='additive', period=7)  # Okres 7 dni
result.plot()
plt.show()