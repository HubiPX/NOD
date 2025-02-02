import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import shapiro, probplot
from sklearn.linear_model import LinearRegression

# Wczytaj dane z pliku CSV
data = pd.read_csv('procesory.csv')

# Wyświetl kolumny dostępne w danych
print("Dostępne kolumny w danych:")
print(data.columns)

# Definiowanie zmiennych
niezalezna = 'Zegar bazowy (GHz)'  # Kolumna niezależna
zalezna = 'Zegar boost (GHz)'       # Kolumna zależna

# Wczytaj kolumny
X = data[[niezalezna]].values
y = data[zalezna].values

# Stworzenie modelu regresji liniowej
model = LinearRegression()
model.fit(X, y)

# Obliczenie przewidywanych wartości oraz reszt
y_pred = model.predict(X)
reszty = y - y_pred

# 3.1. Sprawdzenie normalności reszt (Shapiro-Wilk)
shapiro_test_stat, shapiro_p_value = shapiro(reszty)
print("Test Shapiro-Wilka dla normalności reszt:")
print(f"Statystyka testowa: {shapiro_test_stat:.4f}, p-wartość: {shapiro_p_value:.4e}")

if shapiro_p_value > 0.05:
    print("Brak podstaw do odrzucenia hipotezy zerowej: reszty są normalnie rozłożone.")
else:
    print("Odrzucenie hipotezy zerowej: reszty nie są normalnie rozłożone.")

# 3.2. Test autokorelacji reszt (Durbin-Watson)
durbin_watson_stat = durbin_watson(reszty)
print("\nTest Durbin-Watson:")
print(f"Statystyka Durbin-Watson: {durbin_watson_stat:.4f}")

# Interpretacja wyników testu Durbin-Watson
if durbin_watson_stat < 1.5:
    print("Wskazanie na autokorelację dodatnią reszt.")
elif durbin_watson_stat > 2.5:
    print("Wskazanie na autokorelację ujemną reszt.")
else:
    print("Brak istotnej autokorelacji reszt.")

# 3.3. Wykres Q-Q dla reszt
plt.figure(figsize=(8, 6))
probplot(reszty, dist="norm", plot=plt)
plt.title('Wykres Q-Q dla reszt')
plt.show()

# 3.4. Histogram reszt
plt.figure(figsize=(8, 6))
sns.histplot(reszty, kde=True, bins=10, color='blue')
plt.title('Histogram reszt')
plt.xlabel('Reszty')
plt.ylabel('Częstość')
plt.show()

# 3.5. Analiza autokorelacji reszt
plt.figure(figsize=(10, 6))
plot_acf(reszty, lags=20, ax=plt.gca())
plt.title('Wykres autokorelacji reszt')
plt.show()

# 3.6. Średnia kwadratowa błędu (MSE) dla modelu regresji liniowej
mse_lr = np.mean(reszty**2)  # MSE
print(f"\nMean Squared Error (MSE) dla regresji liniowej: {mse_lr:.2f}")