#!/usr/bin/Rscript
  
##loading required libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library("rpart")
library("kernlab")
library(doMC)
registerDoMC(cores = 4)

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
## Fit SVM model with two different kernels: Radial and Polynomial
##TrainCtrl <- trainControl(method = "repeatedcv", number = 5,repeats=0,verbose = FALSE)
TrainCtrl <- trainControl(method = "cv", number = 5, verbose = FALSE)


## Fit Radial Kernel----------------------------------------------------------
set.seed(300) 
SVMgridRad <- expand.grid(C = (1:10)*0.2 + 0.5, sigma = c(0.030,0.033,0.035))
model_svmRad <- train(factor(loan_status) ~ .,
                     data = training, 
                     method = "svmRadial",
                     trControl = TrainCtrl,
                     tuneGrid = SVMgridRad,
                     preProc = c("scale","center"),
                     verbose = FALSE)

best_sigma <- model_svmRad$bestTune$sigma
best_C <- model_svmRad$bestTune$C
prediction_svm_Rad <- predict(model_svmRad, testing)
con_mat_Rad <- confusionMatrix(prediction_svm_Rad, testing$loan_status)

## Fit Polynomial Kernel-------------------------------------------------------------------------
set.seed(300)
SVMgridPoly <- expand.grid(C = (1:10)*0.2 + 0.5, degree = 1:3, scale = (1:2)*2) 


model_svmPoly <- train(factor(loan_status) ~ .,
                      data = training, 
                      method = 'svmPoly',
                      trControl = TrainCtrl,
                      tuneGrid = SVMgridPoly,
                      preProc = c("scale","center"),
                      verbose = FALSE)

#best_sigma <- model_svm$bestTune$sigma
#best_C <- model_svm$bestTune$C
prediction_svm_Poly <- predict(model_svmPoly, testing)
con_mat_Poly <- confusionMatrix(prediction_svm_Poly, testing$loan_status)

## compare two models
# collect models
result_models <- resamples(list(Radial=model_svmRad, Polynomial=model_svmPoly))

# summarize the distributions
summary(result_models)

# boxplots of results
#plot and save
pdf("figs/LC_svm_model_compare.pdf")
bwplot(result_models)
dev.off()

#-------------------------------------------------------------------------------------------------------------------------!!!!!!!!!!!!!!!!!!


## output confusion matrix
write.table(con_mat_Rad$table, file = "output/LC_confusion_mat_svmRad.txt", row.names = TRUE, col.names = TRUE, sep = "  ")
write.table(con_mat_Poly$table, file = "output/LC_confusion_mat_svmPoly.txt", row.names = TRUE, col.names = TRUE, sep = "  ")

#plot and save
pdf("figs/LC_svm_acc_cost_sigma.pdf")
plot(model_svmRad)
dev.off()


##-------------------------------- Experiment 2 -------------------------------
# Learning Curve
# Vary trainig set size and and observe how accuracy of prediction affected

N_iter <- 20  #|> number of iterations for learning curve

# initilzing empty array for some measures
test_accur <- 0
test_kap <- 0
train_accur <- 0
train_kap <- 0
cpu_time <- 0
data_size <- 0
set.seed(500)   #|> setting random seed
training1 <- training

TrainCtrl <- trainControl(method = "none")
SVMgrid <- expand.grid(sigma = best_sigma, C = best_C)


for (i in 1:N_iter) { 
  
  new_train <- training1 
  training1 <- new_train[createDataPartition(y = new_train$loan_status, p = 0.8, list = FALSE),]
  validation1 <- training[createDataPartition(y = training1$loan_status, p = 0.3, list = FALSE), ]
  
  
  start_time <- Sys.time() ## start the clock------------------------------------------------------
  svmFit <- train(factor(loan_status) ~ .,
                     data = training1, 
                     method = "svmRadial",
                     trControl = TrainCtrl,
                     tuneGrid = SVMgrid,
                     preProc = c("scale","center"),
                     verbose = FALSE)
  
  end_time <- Sys.time()  ## end the clock---------------------------------------------------------
  
  
  ## making a prediction
  prediction_svm_test <- predict(svmFit, testing, type = "raw")
  con_mat_test <- confusionMatrix(prediction_svm_test, testing$loan_status)
  
  prediction_svm_train <- predict(svmFit, validation1, type = "raw")
  con_mat_train <- confusionMatrix(prediction_svm_train, validation1$loan_status)
  
  cpu_time[i] <- round(as.numeric(end_time - start_time),3)
  data_size[i] <- nrow(training1)
  test_accur[i] <- round(as.numeric(con_mat_test$overall[1]),3)
  test_kap[i] <- round(as.numeric(con_mat_test$overall[2]),3)
  train_accur[i] <- round(as.numeric(con_mat_train$overall[1]),3)
  train_kap[i] <- round(as.numeric(con_mat_train$overall[2]),3)
}

results <- data.frame(test_accur,test_kap,train_accur,train_kap, cpu_time, data_size)
write.table(results, file = "output/LC_learning_results_svm.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


## plot some results
pl <- ggplot(results, aes(x=data_size)) +
  geom_line(aes(y = train_accur, colour = "train")) + 
  geom_line(aes(y = test_accur, colour = "test")) +
  geom_point(aes(y = train_accur,colour = "train")) + 
  geom_point(aes(y = test_accur,colour = "test")) +
  theme_bw() +
  labs(title = "Learning Curve SVM", x = "Training Size", y = "Accuracy", color="") +
  theme(legend.position = c(0.6,0.8),
        axis.title = element_text(size = 16.0),
        axis.text = element_text(size=10, face = "bold"),
        plot.title = element_text(size = 15, hjust = 0.5),
        axis.text.x = element_text(colour="black"),
        axis.text.y = element_text(colour="black"))

#plot and save
png("figs/LC_svm_learning_curve.png", width = 5.0, height = 4.0, units = "in", res = 800)
pl
dev.off()



