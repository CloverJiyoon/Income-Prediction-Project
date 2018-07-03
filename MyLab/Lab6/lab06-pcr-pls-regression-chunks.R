# R script for lab 06: Regression with Dimension Reduction
# Prof. Gaston Sanchez
# Stat 154, Fall 2017

# packages
library(pls)
library(ISLR)


## @knitr data_hitters
data(Hitters)

## @knitr head_hitters
head(Hitters)

## @knitr str_hitters
str(Hitters, vec.len = 1)

## @knitr pcr_fit
# principal component regression
pcr_fit <- pcr(Salary ~ ., data = Hitters, scale = TRUE, validation = "none")
names(pcr_fit)


## @knitr na_omit
# remove missing values
hitters <- na.omit(Hitters)


## @knitr model_matrix
# model matrix
MM <- model.matrix(Salary ~ ., data = hitters)

## @knitr predictors
# exclude 1st column of model matrix
X <- scale(MM[ ,-1])

## @knitr response
y <- hitters$Salary

## @knitr svd
# SVD on X (without intercept term)
SVD <- svd(X)
U <- SVD$u
V <- SVD$v
D <- diag(SVD$d)

## @knitr pc_components
# PCs
Z <- U %*% D

## @knitr comps_vs_scores
# compare Z vs pcr-scores
head(cbind(pcr_fit$scores[,1], Z[,1]))
head(cbind(pcr_fit$scores[,19], Z[,19]))


## @knitr pcreg_z1
# regression with first component
pcreg_1 <- lm(y ~ Z[,1])
b1_pcr <- pcreg_1$coefficients

## @knitr fitted_pcr1
# compare y-hat with pcr() output
head(cbind(pcreg_1$fitted.values, pcr_fit$fitted.values[,,1]))

## @knitr b1_pcr
# PCR coeffs with PC1, in terms of predictors
pcreg_1$coefficients[-1] * V[,1]

# compare with output from pcr()
pcr_fit$coefficients[,,1]


## @knitr pcreg_z12
# regression with PC1 and PC2
pcreg_12 <- lm(y ~ Z[,1:2])

## @knitr b12_pcr
# regression coeffs in terms of predictors
beta_12 <- V[,1:2] %*% pcreg_12$coefficients[-1] 
cbind(beta_12, pcr_fit$coefficients[,,2])

## @knitr fitted_12
# comapre your coeffs vs those provided by pcr()
cbind(pcreg_12$fitted.values, pcr_fit$fitted.values[,,2])


## @knitr pcreg_all
# regression with all PCs
pcreg_all <- lm(y ~ Z)

## @knitr fitted_all
# comapre your coeffs vs those provided by pcr()
head(cbind(pcreg_all$fitted.values, pcr_fit$fitted.values[,,19]))


## @knitr all_pcr_coeffs
coeffs <- matrix(0, nrow = ncol(X), ncol = ncol(Z))
for (k in 1:ncol(Z)) {
  pcreg_k <- lm(y ~ Z[ ,1:k])
  if (k == 1) {
    coeffs[ ,k] <- pcreg_k$coefficients[-1] * V[,1] 
  } else {
    coeffs[ ,k] <- V[,1:k] %*% pcreg_k$coefficients[-1]
  }
}
rownames(coeffs) <- colnames(X)
colnames(coeffs) <- paste0('Z', 1:ncol(Z))

# compare agains pcr()
cbind(coeffs[ ,1:3], pcr_fit$coefficients[,,1:3])


## @knitr pcr_cv
# =========================================================
# PCR with Cross-Validation: 10-folds
# =========================================================
# size of folds
num_obs <- nrow(hitters)
fold_size <- num_obs %/% 10
remainder <- num_obs %% 10

folds <- seq(1, num_obs, by = fold_size)
folds[length(folds)] <- num_obs
folds

# starting and ending positions of folds
begin <- folds[-length(folds)]
end <- folds[-1] - 1
end[length(end)] <- num_obs
cbind(start, end)

# randomize observations
set.seed(1)
randomized <- sample(1:num_obs, size = num_obs)

# vector of MSEs
mse_cv <- rep(0, 10)

for (k in 1:10) {
  exclude <- randomized[begin[k]:end[k]]
  Xk <- scale(MM[-exclude,-1])
  yk <- y[-exclude]
  SVDk <- svd(Xk)
  Vk <- SVDk$v
  Dki <- diag(1/SVDk$d)
  Uk <- SVDk$u
  
  bk <- Vk %*% Dki %*% t(Uk) %*% yk
  yk_hat <- Xk %*% bk
  residk <- yk - yk_hat
  mse_cv[k] <- sqrt(mean(residk^2))
}
mse_cv
which.min(mse_cv)


set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(hitters), replace = TRUE)
test <- (!train)


set.seed(1)
pcr.fit = pcr(Salary ~ ., data = Hitters, subset = train, scale = TRUE, 
              validation = "CV")


pcr.fit = pcr(Salary ~ ., data = Hitters, scale = TRUE, validation = "CV")



# # =========================================================
# # PLSR "classic" version
# # =========================================================
# 
# pls_fit <- plsr(Salary ~ ., data = Hitters, scale = TRUE, validation = "none")
# 
# Z <- matrix(0, nrow(X), ncol(X))  # PLS components
# P <- matrix(0, ncol(X), ncol(X))  # PLS loadings
# W <- matrix(0, ncol(X), ncol(X))  # PLS weights
# d <- rep(0, ncol(X))              # regression coeffs
#   
# Xh <- X
# yh <- y
# 
# for (h in 1:ncol(X)) {
#   w <- t(Xh) %*% yh
#   w_norm <- sqrt(sum(w*w))
#   w <- w / w_norm
#   # weights
#   W[,h] <- w
#   # pls components
#   z <- Xh %*% w
#   Z[,h] <- z
#   # pls loadings
#   p <- t(Xh) %*% z / sum(z*z)
#   P[,h] <- p
#   # pls coeffs
#   dh <- sum(yh * z) / sum(z*z)
#   d[h] <- dh
#   # deflations
#   Xh <- Xh - (z %*% t(p))
#   yh <- yh - (dh * z)
# }
# 
# 
# 
# 
# # =========================================================
# # PLSR Helland's algorithm
# # =========================================================
# 
# Z <- matrix(0, nrow(X), ncol(X))  # PLS components
# P <- matrix(0, ncol(X), ncol(X))  # PLS loadings
# W <- matrix(0, ncol(X), ncol(X))  # PLS weights
# 
# Xh <- X
# 
# for (h in 1:ncol(X)) {
#   w <- t(Xh) %*% y
#   w_norm <- sqrt(sum(w*w))
#   w <- w / w_norm
#   # weights
#   W[,h] <- w
#   # pls components
#   z <- Xh %*% w
#   Z[,h] <- z
#   # pls loadings
#   p <- t(Xh) %*% z / sum(z*z)
#   P[,h] <- p
#   # deflation
#   Xh <- Xh - z %*% t(p)
# }
# 
# 
# 
