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

# simulate MA(3) process
ma.coeffs <- c(-0.5, 0.3, -0.2)
x <- replicate(K, arima.sim(model = list(ma = ma.coeffs), n = N + H))

errors <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  for (t in 4:(N + H)) {
    errors[t, k] <- x[t, k] - sum(ma.coeffs * errors[(t-1):(t-3), k], na.rm = TRUE)
  }
}

### b) Optimal MSE Forecasts ----

optimal.fc <- array(0, dim = c(N, H, K))

for (k in 1:K) {
  for (t in 3:N) {
    optimal.fc[t, 1, k] <- ma.coeffs[1] * errors[t, k] + ma.coeffs[2] * errors[t-1, k] + ma.coeffs[3] * errors[t-2, k]
    optimal.fc[t, 2, k] <- ma.coeffs[2] * errors[t, k] + ma.coeffs[3] * errors[t-1, k]
    optimal.fc[t, 3, k] <- ma.coeffs[3] * errors[t, k]
  }
}

# Print first few forecasts for trajectory 1
print(optimal.fc[1:10, , 1])


# Initiate a vector for the MSE for the H forecast periods
optimal.mse <- numeric(H)

# loop over the three forecasting periods
for (h in 1:H) {
  
  # initiate an vector to store error terms
  errors.squ <- numeric(K)
  
  # loop over each trajectory
  for (k in 1:K) {
    # for each period to forecast, subtract the forecast from the previous period
    errors.fc <- x[(N + h), k] - (x[N,k] + optimal.fc[N,h,k])
    # square result
    errors.squ[k] <- errors.fc^2
  }
  
  # take mean over all trajectories
  optimal.mse[h] <- mean(errors.squ)
}

print(optimal.mse)


### c) Approach 1: Naive Error Sequence Reconstruction ----

err.recon <- matrix(0, nrow = N + H + 2, ncol = K)

approach1.fc <- array(0, dim = c(N, H, K))

for (k in 1:K) {
  
  err.recon[1, k] <- x[1, k]
  err.recon[2, k] <- x[2, k] - ma.coeffs[1]*err.recon[1,k]
  err.recon[3, k] <- x[3, k] - ma.coeffs[1]*err.recon[2, k] - ma.coeffs[2] * err.recon[1, k]
  
  for (t in 4:N) { 
    err.recon[t, k] <- x[t, k] - sum(ma.coeffs * err.recon[(t-1):(t-3), k], na.rm = TRUE)
  }
  
  for (t in 4:N) {
    approach1.fc[t, 1, k] <- ma.coeffs[1] * err.recon[t, k] + ma.coeffs[2] * err.recon[t-1, k] + ma.coeffs[3] * err.recon[t-2, k]
    approach1.fc[t, 2, k] <- ma.coeffs[2] * err.recon[t, k] + ma.coeffs[3] * err.recon[t-1, k]
    approach1.fc[t, 3, k] <- ma.coeffs[3] * err.recon[t, k]
  }
}

approach1.mse <- numeric(H)

for (h in 1:H) {
  errors.squ <- numeric(K)
  for (k in 1:K) {

    errors.fc <- x[N + h, k] - (x[N, k] + approach1.fc[N, h, k])
    errors.squ[k] <- errors.fc^2
  }
  
  approach1.mse[h] <- mean(errors.squ)
}

print(optimal.mse)
print(approach1.mse)

### d) Approach 2: Projection Method ----
