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

# Initiate a matrix for the MSE for each approach
mse.matrix <- matrix(0, nrow = H, ncol = 3)

# Initiate matrix for optimal MSE forecast
Xhat.opt <- matrix(0, nrow = N + H, ncol = K)

for (k in 1:K) {
  for (t in N:(N + H - 1)) {  
    if (t + 1 <= N + H) Xhat.opt[t + 1, k] <- ma.coeffs[1] * errors[t, k] + ma.coeffs[2] * errors[t - 1, k] + ma.coeffs[3] * errors[t - 2, k]
    if (t + 2 <= N + H) Xhat.opt[t + 2, k] <- ma.coeffs[2] * errors[t, k] + ma.coeffs[3] * errors[t - 1, k]
    if (t + 3 <= N + H) Xhat.opt[t + 3, k] <- ma.coeffs[3] * errors[t, k]
  }
}

# Compute the Mean Squared Error (MSE) for each forecast horizon
for (h in 1:H) {
  
  # Store squared errors
  errors.squ.opt <- numeric(K)
  
  for (k in 1:K) {
    errors.squ.opt[k] <- (x[N + h, k] - Xhat.opt[N + h, k])^2
  }
  
  # Compute the mean squared error for each horizon
  mse.matrix[h, 1] <- mean(errors.squ.opt)
}

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
for (h in 1:H) {
  errors.squ.1 <- numeric(K)
  for (k in 1:K) {
    errors.squ.1[k] <- (x[N + h, k] - Xhat.1[N + h, k])^2
  }
  mse.matrix[h, 2] <- mean(errors.squ.1)
}

### d) Approach 2: Projection Method ----

sigma.squ <- 1 # Variance of white noise
M <- 5 # Number of past values used for forecasting
theta <- c(1, -0.5, 0.3, -0.2) # Including theta_0 = 1 by convention

# Function to compute autocovariance of an MA(3) process
gamma_h <- function(h, theta, sigma.squ) {
  if (h > 3) return(0)
  sum <- 0
  for (j in 0:(3 - h)) {
    sum <- sum + theta[j + 1] * theta[j + h + 1]
  }
  return(sigma.squ * sum)
}

# Compute autocovariance matrix Gamma_m
Gamma_m <- matrix(0, nrow = M, ncol = M)
for (i in 1:M) {
  for (j in 1:M) {
    Gamma_m[i, j] <- gamma_h(abs(i - j), theta, sigma.squ)
  }
}

# Matrix for forecasted values using projection method
Xhat.2 <- matrix(0, nrow = N + H, ncol = K)

# Compute forecasts for each simulation
for (k in 1:K) {
  for (h in 1:H) {
    # Compute gamma_h vector for each h
    gamma_h_vec <- sapply(0:(M - 1), function(j) gamma_h(h + j, theta, sigma.squ))
    alpha_h <- solve(Gamma_m, gamma_h_vec)
    
    # Use last M observations for forecasting
    X_t <- x[N:(N - M + 1), k]
    Xhat.2[N + h, k] <- sum(alpha_h * X_t)
  }
}

# Compute MSE for Approach 2
for (h in 1:H) {
  errors.squ.2 <- numeric(K)
  for (k in 1:K) {
    errors.squ.2[k] <- (x[N + h, k] - Xhat.2[N + h, k])^2
  }
  mse.matrix[h, 3] <- mean(errors.squ.2)
}

### e) Compare Forecasting Methods ----

df.mse <- data.frame(
  Horizon = 1:H,
  Optimal = mse.matrix[,1],
  Approach1 = mse.matrix[,2],
  Approach2 = mse.matrix[,3]
)

print(df.mse)

ggplot(df.mse, aes(x = Horizon)) +
  geom_line(data = df.mse, aes(x = Horizon, y = Optimal), 
            size = 1, linetype = "dashed", ,color = "black") +
  geom_line(aes(y = Approach1, color = "Approach 1"), size = 1) +
  geom_line(aes(y = Approach2, color = "Approach 2 (M=5)"), size = 1) +
  labs(title = "Comparison of MSEs for Forecasting Approaches",
       x = "Forecast Horizon (h)", 
       y = "Mean Squared Error (MSE)") +
  scale_color_manual(values = c("red", "blue", "black")) +
  theme_minimal()

M.grid <- 1:25
mse.results <- matrix(0, nrow = H, ncol = length(M.grid))

for (j in seq_along(M.grid)) {
  M <- M.grid[j]
  
  # Compute Gamma_m matrix
  Gamma_m <- matrix(0, nrow = M, ncol = M)
  for (i in 1:M) {
    for (j in 1:M) {
      Gamma_m[i, j] <- gamma_h(abs(i - j), theta, sigma.squ)
    }
  }
  
  # Compute MSE for each forecast horizon h
  for (h in 1:H) {
    gamma_h_vec <- sapply(0:(M - 1), function(j) gamma_h(h + j, theta, sigma.squ))
    alpha_h <- solve(Gamma_m, gamma_h_vec)
    
    errors <- numeric(K)
    for (k in 1:K) {
      X_t <- tail(x[1:N, k], M)
      X_hat <- sum(alpha_h * X_t)
      errors[k] <- (x[N + h, k] - X_hat)^2
    }
    
    mse.results[h, j] <- mean(errors)
  }
}

# Convert MSE results to dataframe for plotting
df.mse.m <- data.frame(
  Horizon = rep(1:H, times = length(M.grid)),
  MSE = as.vector(mse.results),
  M = rep(M.grid, each = H)
)

ggplot(df.mse.m, aes(x = Horizon, y = MSE, color = factor(M))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_line(data = df.mse, aes(x = Horizon, y = Optimal), 
            size = 1, linetype = "dashed", color = "black") +
  labs(title = "Effect of M on Projection Method Accuracy",
       x = "Forecast Horizon (h)", 
       y = "Mean Squared Error (MSE)",
       color = "M value") +
  theme_minimal()


df.mse.horizon <- df.mse.m %>%
  filter(Horizon %in% c(1, 2, 3))
ggplot(df.mse.horizon, aes(x = M, y = MSE, color = factor(Horizon))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Effect of M on MSE for Different Forecast Horizons",
       x = "M value (Number of Past Observations)",
       y = "Mean Squared Error (MSE)",
       color = "Horizon") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue", "green")) 
