#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library(doMC)
registerDoMC(cores = 4)


## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = "", header = TRUE)
data[data$loan_status == 0,]$loan_status <- "No"
data[data$loan_status == 1,]$loan_status <- "Yes"

## extract 10% of data and perfor hyperparameter tuning 
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.1, list=FALSE),]

# break sub data into train test and validation sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 


##############################################################################################
##--------------------------------- Experiment 1 -------------------------------------------##
## Cross validation
## Neural Network Model
set.seed(2017)
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 2, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)

nnetGrid <- expand.grid(size = seq(from = 5, to = 15, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

nnetFit <- train(factor(loan_status) ~ .,
                 data = training,
                 method = "nnet",
                 metric = "ROC",
                 trControl = fitControl,
                 tuneGrid = nnetGrid,
                 verbose = FALSE)

prediction_nnet <- predict(nnetFit, newdata=testing, type = "raw")
con_mat_nnet <- confusionMatrix(prediction_nnet, testing$loan_status)
con_mat_nnet$overall[1]
plot(nnetFit)
