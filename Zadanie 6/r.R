# Załaduj potrzebne biblioteki
library(readr)   # Do wczytywania danych
library(ggplot2) # Do wizualizacji
library(car)     # Do testu shapiro-wilka
library(stats)   # Do testu Durbin-Watsona
library(ggpubr)  # Dla grafiki Q-Q

# Wczytaj dane z pliku CSV
data <- read_csv("procesory.csv")

# Wyświetl kolumny dostępne w danych
print("Dostępne kolumny w danych:")
print(colnames(data))

# Definiowanie zmiennych
niezalezna <- "Zegar bazowy (GHz)"  # Kolumna niezależna
zalezna <- "Zegar boost (GHz)"       # Kolumna zależna

# Wbudowanie modelu regresji liniowej
model <- lm(as.formula(paste(zalezna, "~", niezalezna)), data = data)

# Oblicz przewidywane wartości i reszty
y_pred <- predict(model)
reszty <- resid(model)

# 3.1. Test normalności reszt - Shapiro-Wilk
shapiro_test <- shapiro.test(reszty)
print("Test Shapiro-Wilka dla normalności reszt:")
print(shapiro_test)

# Interpretacja wyniku testu Shapiro-Wilka
if (shapiro_test$p.value > 0.05) {
    cat("Brak podstaw do odrzucenia hipotezy zerowej: reszty są normalnie rozłożone.\n")
} else {
    cat("Odrzucenie hipotezy zerowej: reszty nie są normalnie rozłożone.\n")
}

# 3.2. Test autokorelacji reszt - Durbin-Watson
durbin_watson_stat <- dwtest(model)
print("\nTest Durbin-Watson:")
print(durbin_watson_stat)

# 3.3. Wykres Q-Q dla reszt
ggqqplot(reszty) + ggtitle("Wykres Q-Q dla reszt")

# 3.4. Histogram reszt
ggplot(data.frame(reszty), aes(x = reszty)) +
    geom_histogram(aes(y = ..density..), bins = 10, fill = "blue", color = "black") +
    geom_density(alpha = 0.2, fill = "red") +
    labs(title = "Histogram reszt", x = "Reszty", y = "Częstość")

# 3.5. Średnia kwadratowa błędu (MSE) dla modelu regresji liniowej
mse_lr <- mean(reszty^2)  # MSE
cat(sprintf("Mean Squared Error (MSE) dla regresji liniowej: %.2f\n", mse_lr))