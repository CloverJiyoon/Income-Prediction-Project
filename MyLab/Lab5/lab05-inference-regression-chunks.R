# R script for lab 05: Inference in Linear Regression and Model Assessment
# Johnny Hong
# Stat 154, Fall 2017



## @knitr fit
reg <- lm(mpg ~ disp + hp, data = mtcars)
summary(reg)
reg$coefficients["disp"] + 
  c(-1, 1) * qt(0.975, reg$df.residual) * 
  coef(summary(reg))["disp", "Std. Error"]
confint(reg, "disp", 0.95)

## @knitr hypothesis


## @knitr insample
degs <- 1:6
models <- lapply(degs, function(deg) {
  lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars)
})
inMSEs <- sapply(models, function(model) {
  mean(model$residuals^2)
})
plot(degs, inMSEs, type="o", main="In-sample MSE vs Order of Polynomials",
     xlab="Order", ylab="in-sample MSE")

## @knitr insample
degs <- 1:6
models <- lapply(degs, function(deg) {
  lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars)
})
inMSEs <- sapply(models, function(model) {
  mean(model$residuals^2)
})
plot(degs, inMSEs, type="o", main="In-sample MSE vs Order of Polynomials",
     xlab="Order", ylab="in-sample MSE")

## @knitr holdout
set.seed(32901)
p <- 0.2
holdoutInd <- sample(1:nrow(mtcars), floor(nrow(mtcars)) * p)
sapply(degs, function(deg) {
  model <- lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars[-holdoutInd, ])
  mean((predict(model, mtcars[holdoutInd, ]) - mtcars[holdoutInd, "mpg"])^2)
})

## @knitr cv
kfoldMSEs <- sapply(folds, function(ind) {
  sapply(degs, function(deg) {
    model <- lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars[-ind, ])
    mean((predict(model, mtcars[ind, ]) - mtcars[ind, "mpg"])^2)
  })
})
cvMSEs <- rowMeans(kfoldMSEs)
cvMSEs
plot(degs, cvMSEs, type="o", main="CV-MSE vs Order of Polynomials",
     xlab="Order", ylab="cv-/MSE")
plot(mpg ~ disp, data=mtcars, main="mpg vs disp")

## @knitr cv5
set.seed(13490)
folds <- createFolds(mtcars$mpg, k = 5)
kfoldMSEs <- sapply(folds, function(ind) {
  sapply(degs, function(deg) {
    model <- lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars[-ind, ])
    mean((predict(model, mtcars[ind, ]) - mtcars[ind, "mpg"])^2)
  })
})
cvMSEs <- rowMeans(kfoldMSEs)
cvMSEs
plot(degs, cvMSEs, type="o", main="CV-MSE vs Order of Polynomials",
     xlab="Order", ylab="cv-/MSE")
plot(mpg ~ disp, data=mtcars, main="mpg vs disp")

## @knitr cvn
set.seed(41903)
folds <- createFolds(mtcars$mpg, k = nrow(mtcars))
folds
kfoldMSEs <- sapply(folds, function(ind) {
  sapply(degs, function(deg) {
    model <- lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars[-ind, ])
    mean((predict(model, mtcars[ind, ]) - mtcars[ind, "mpg"])^2)
  })
})
cvMSEs <- rowMeans(kfoldMSEs)
cvMSEs
plot(degs, cvMSEs, type="o", main="CV-MSE vs Order of Polynomials",
     xlab="Order", ylab="cv-/MSE")
plot(mpg ~ disp, data=mtcars, main="mpg vs disp")

# @knitr bootstrap
set.seed(114390)
resamples <- createResample(mtcars$mpg, 400)
resamplepMSEs <- sapply(resamples, function(resample) {
  sapply(degs, function(deg) {
    model <- lm(mpg ~ poly(disp, deg, raw=TRUE), data=mtcars[resample, ])
    mean((predict(model, mtcars[-resample, ]) - mtcars[-resample, "mpg"])^2)
  })
})
bootstrapMSEs <- rowMeans(resamplepMSEs)
bootstrapMSEs
plot(degs, bootstrapMSEs, type="o", main="Bootstrap MSE vs Order of Polynomials",
     xlab="Order", ylab="bootstrap sample MSE")
plot(degs, apply(resamplepMSEs, 1, sd), type="o", 
     main="SD of resample MSEs vs Order of Polynomials",
     xlab="Order", ylab="SD of resample MSEs")