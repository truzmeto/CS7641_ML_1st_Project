#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library("rpart")

## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = "", header = TRUE)

## extract 10% of data and performe hyperparameter tuning 
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.01, list=FALSE),]

# break sub data into train test and validation sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 

##----------------------------------- Experiment 1 ------------------------------------##
## Support Vector Machines
## Fit SVM model
#TrainCtrl <- trainControl(method = "repeatedcv", number = 5,repeats=0,verbose = FALSE)
TrainCtrl <- trainControl(method = "cv", number = 5,verbose = FALSE)


set.seed(300) 
SVMgrid <- expand.grid(sigma = c(0.03, 0.035), C = (1:5)*0.1 + 1.0)

model_svm <- train(factor(loan_status) ~ .,
                     data = training, 
                     method="svmRadial",
                     trControl=TrainCtrl,
                     tuneGrid = SVMgrid,
                     preProc = c("scale","center"),
                     verbose=FALSE)


prediction_svm <- predict(model_svm, testing)
#Accuracy(prediction_svm, testing$loan_status)
confusionMatrix(prediction_svm, testing$loan_status)$overall[1]





