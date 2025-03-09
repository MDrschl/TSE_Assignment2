rm(list = ls(all = TRUE))

library(tidyverse)
library(tseries)
library(readxl) 
library(ggfortify)
library(accelerometry)
library(zoo)

set.seed(2025)

### a) Simulation ----

K <- 1000 # Number of trajectories
N <- 100 # In-sample size
H <- 3 # Forecast horizon

# coefficients of MA(3)
ma.coeffs <- c(-0.5, 0.3, -0.2)
# creating series of errors for each trajectory
errors <- matrix(rnorm(K * (N + H)), nrow = N + H, ncol = K)

# Simulate MA(3) process using the generated innovations and the coefficient vector
x <- matrix(0, nrow = N + H, ncol = K)
for (k in 1:K) {
  x[, k] <- arima.sim(model = list(ma = ma.coeffs), n = N + H, innov = errors[, k])
}

### b) Optimal MSE Forecasts ----

# initiate matrix for optimal MSE forecast
optimal.fc <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  for (t in 3:N) {
    optimal.fc[t + 1, k] <- ma.coeffs[1] * errors[t, k] + ma.coeffs[2] * errors[t - 1, k] + ma.coeffs[3] * errors[t - 2, k]
    if (t + 2 <= N + H) {
      optimal.fc[t + 2, k] <- ma.coeffs[2] * errors[t, k] + ma.coeffs[3] * errors[t - 1, k]
    }
    if (t + 3 <= N + H) {
      optimal.fc[t + 3, k] <- ma.coeffs[3] * errors[t, k]
    }
  }
}


# Initiate a vector for the MSE for the H forecast periods
optimal.mse <- numeric(H)

# loop over the three forecasting periods
for (h in 1:H) {
  
  # initiate an vector to store error terms
  errors.squ <- numeric(K)
  
  # loop over each trajectory and caluclate mean squared error
  for (k in 1:K) {
    errors.fc <- x[(N + h), k] - optimal.fc[(N + h), k]
    errors.squ[k] <- errors.fc^2
  }
  optimal.mse[h] <- mean(errors.squ)
}

print(optimal.mse)

### c) Approach 1: Naive Error Sequence Reconstruction ----

# initiate matrix for reconstructed errors
err.recon <- matrix(0, nrow = N + H, ncol = K)

approach1.fc <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  
  err.recon[1, k] <- x[1, k]
  err.recon[2, k] <- x[2, k] - ma.coeffs[1] * err.recon[1, k]
  err.recon[3, k] <- x[3, k] - ma.coeffs[1] * err.recon[2, k] - ma.coeffs[2] * err.recon[1, k]
  
  for (t in 4:(N)) {
    err.recon[t, k] <- x[t, k] - sum(ma.coeffs * err.recon[(t - 1):(t - 3), k], na.rm = TRUE)
  }
  
  for (t in 3:N) {
    approach1.fc[t + 1, k] <- ma.coeffs[1] * err.recon[t, k] + ma.coeffs[2] * err.recon[t - 1, k] + ma.coeffs[3] * err.recon[t - 2, k]
    if (t + 2 <= N + H) {
      approach1.fc[t + 2, k] <- ma.coeffs[2] * err.recon[t, k] + ma.coeffs[3] * err.recon[t - 1, k]
    }
    if (t + 3 <= N + H) {
      approach1.fc[t + 3, k] <- ma.coeffs[3] * err.recon[t, k]
    }
  }
}

approach1.mse <- numeric(H)

for (h in 1:H) {
  errors.squ <- numeric(K)
  for (k in 1:K) {
    errors.fc <- x[N + h, k] - approach1.fc[N + h, k] # Corrected line
    errors.squ[k] <- errors.fc^2
  }
  approach1.mse[h] <- mean(errors.squ)
}

print(optimal.mse)
print(approach1.mse)

### d) Approach 2: Projection Method ----
