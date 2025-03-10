rm(list = ls(all = TRUE))

library(tidyverse)
library(tseries)
library(readxl) 
library(ggfortify)
library(accelerometry)
library(zoo)
library(MASS)

set.seed(2025)

### a) Simulation ----

K <- 1000 # Number of trajectories
N <- 100  # In-sample size
H <- 3    # Forecast horizon

# Coefficients of MA(3)
ma.coeffs <- c(-0.5, 0.3, -0.2)

# Creating series of errors for each trajectory
errors <- matrix(rnorm(K * (N + H)), nrow = N + H, ncol = K)

# Simulate MA(3) process using the generated innovations and the coefficient vector
x <- matrix(0, nrow = N + H, ncol = K)
for (k in 1:K) {
  x[, k] <- arima.sim(model = list(ma = ma.coeffs), n = N + H, innov = errors[, k])
}

### b) Optimal MSE Forecasts ----

# Initiate matrix for optimal MSE forecast
Xhat.opt <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  for (t in N:(N + H - 1)) {  
    if (t + 1 <= N + H) Xhat.opt[t + 1, k] <- ma.coeffs[1] * errors[t, k] + ma.coeffs[2] * errors[t - 1, k] + ma.coeffs[3] * errors[t - 2, k]
    if (t + 2 <= N + H) Xhat.opt[t + 2, k] <- ma.coeffs[2] * errors[t, k] + ma.coeffs[3] * errors[t - 1, k]
    if (t + 3 <= N + H) Xhat.opt[t + 3, k] <- ma.coeffs[3] * errors[t, k]
  }
}

# Initiate a vector for the MSE for the H forecast periods
optimal.mse <- numeric(H)

# Compute the Mean Squared Error (MSE) for each forecast horizon
for (h in 1:H) {
  
  # Store squared errors
  errors.squ.opt <- numeric(K)
  
  for (k in 1:K) {
    errors.squ.opt[k] <- (x[N + h, k] - Xhat.opt[N + h, k])^2
  }
  
  # Compute the mean squared error for each horizon
  optimal.mse[h] <- mean(errors.squ.opt)
}

print(optimal.mse)


### c) Approach 1: Naive Error Sequence Reconstruction ----

# Initiate matrix for reconstructed errors
err.recon <- matrix(0, nrow = N + H, ncol = K)

# Initiate matrix for forecasted values
Xhat.1 <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  
  # Reconstruct error terms
  err.recon[1, k] <- x[1, k]
  err.recon[2, k] <- x[2, k] - ma.coeffs[1] * err.recon[1, k]
  err.recon[3, k] <- x[3, k] - ma.coeffs[1] * err.recon[2, k] - ma.coeffs[2] * err.recon[1, k]
  
  for (t in 4:N) {
    err.recon[t, k] <- x[t, k] - sum(ma.coeffs * err.recon[(t - 1):(t - 3), k], na.rm = TRUE)
  }
  
  # Forecast future values using reconstructed errors
  for (t in N:(N + H - 1)) {
    if (t + 1 <= N + H) Xhat.1[t + 1, k] <- ma.coeffs[1] * err.recon[t, k] + ma.coeffs[2] * err.recon[t - 1, k] + ma.coeffs[3] * err.recon[t - 2, k]
    if (t + 2 <= N + H) Xhat.1[t + 2, k] <- ma.coeffs[2] * err.recon[t, k] + ma.coeffs[3] * err.recon[t - 1, k]
    if (t + 3 <= N + H) Xhat.1[t + 3, k] <- ma.coeffs[3] * err.recon[t, k]
  }
}

# Calculate MSE for Approach 1
approach1.mse <- numeric(H)

for (h in 1:H) {
  errors.squ.1 <- numeric(K)
  for (k in 1:K) {
    errors.squ.1[k] <- (x[N + h, k] - Xhat.1[N + h, k])^2
  }
  approach1.mse[h] <- mean(errors.squ.1)
}

### d) Approach 2: Projection Method ----

M <- 5
h <- H:(H+M - 1)

gamma.func <- function(h, macoeffs,sigmasqu) {
  theta <- c(1, macoeffs)
  if (h > 3) {
    return(0)
  } else {
    sum_val <- 0
    for (j in 0:(3 - h)) {
      sum_val <- sum_val + theta[j + 1] * theta[j + h + 1]
    }
    return(sigmasqu * sum_val)
  }
}

gamma.m <- sapply(h, gamma.func, macoeffs = ma.coeffs, sigmasqu = 1)

# Construct covariance matrix Gamma_m (MxM)
Gamma_m <- matrix(0, nrow = M, ncol = M)
for (i in 1:M) {
  for (j in 1:M) {
    Gamma_m[i, j] <- gamma[abs(i - j) + 1]
  }
}

# Invert Gamma_m
Gamma_m_inv <- ginv(Gamma_m)  # Moore-Penrose inverse to avoid singularity issues

# Initialize forecast matrix
Xhat.2 <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  # Construct X_t^(m) for each trajectory
  X_t_m <- matrix(0, nrow = N - M, ncol = M)
  
  for (t in (M + 1):N) {
    X_t_m[t - M, ] <- x[t:(t - M + 1), k]
  }
  
  for (h in 1:H) {
    # Create gamma_h^(m) vector
    gamma_h_m <- gamma[(h + 1):(h + M)]
    
    # Compute optimal coefficients α_h^(m)
    alpha_h_m <- Gamma_m_inv %*% gamma_h_m
    
    # Forecast using projection formula
    for (t in N:(N + H - 1)) {
      if (t - M + 1 > 0) {
        Xhat.2[t + h, k] <- sum(alpha_h_m * x[t:(t - M + 1), k])
      }
    }
  }
}

# Compute MSE for Approach 2
approach2.mse <- numeric(H)

for (h in 1:H) {
  errors.squ.2 <- numeric(K)
  for (k in 1:K) {
    errors.squ.2[k] <- (x[N + h, k] - Xhat.2[N + h, k])^2
  }
  approach2.mse[h] <- mean(errors.squ.2)
}

print(approach2.mse)











print(optimal.mse)
print(approach1.mse)
print(approach2.mse)
