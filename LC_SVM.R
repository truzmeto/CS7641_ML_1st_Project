#!/usr/bin/Rscript
  
##loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library("rpart")
library("kernlab")

## setting seed for random number generator
set.seed(300)
	
## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = "", header = TRUE)
     	
## extract some part of data and performe hyperparameter tuning 
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.1, list=FALSE),]

## break sub data into train test and validation sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 

##----------------------------------- Experiment 1 ------------------------------------##
## Support Vector Machines
## Fit SVM model
##TrainCtrl <- trainControl(method = "repeatedcv", number = 5,repeats=0,verbose = FALSE)
TrainCtrl <- trainControl(method = "cv", number = 10,verbose = FALSE)

set.seed(300) 
SVMgrid <- expand.grid(sigma = c(0.03,0.033,0.035,0.037), C = (1:10)*0.1 + 1.0)

model_svm <- train(factor(loan_status) ~ .,
                     data = training, 
                     method="svmRadial",
                     trControl=TrainCtrl,
                     tuneGrid = SVMgrid,
                     preProc = c("scale","center"),
                     verbose=FALSE)


prediction_svm <- predict(model_svm, testing)
con_mat <- confusionMatrix(prediction_svm, testing$loan_status)

## output confusion matrix
write.table(con_mat$table, file = "output/confusion_mat_svm.txt", row.names = TRUE, col.names = TRUE, sep = "  ")

#plot and save
pdf("figs/svm_acc_cost_sigma.pdf")
plot(model_svm)
dev.off()

