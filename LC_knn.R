#!/usr/bin/Rscript

#loading reuired libraries
library(ggplot2)
library("lattice") 
library("caret")
library("plyr")

## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = " ", header = TRUE)

# function to convert factor features to numeric
FacToString <- function(input) {
  for(i in 1:ncol(input)){
    if(class(input[,i]) == "factor") { 
      input[,i] <- as.integer(as.factor(input[,i])) 
    }
  }
  input
}

# convert factors to numeric
data <- FacToString(data)

# convert "loan_status" col. back to factor
data$loan_status <- as.factor(data$loan_status)

##-------------------------------- Experiment 1 -------------------------------------## 
# Find optimum k value, which correspond to highest accuracy

## extract 2% of data and perfor hyperparameter tuning with 5 fold cross validation
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.02, list=FALSE),]

# break sub data into train and test sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 

## apply KNN algorithm
set.seed(400)
ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5)
                    #,summaryFunction = twoClassSummary)


knnFit <- train(loan_status ~ .,
                data = training,
                method = "knn",
                trControl = ctrl,
                preProcess = c("center","scale"),
                tuneLength = 20)


#Output of kNN fit
knnFit

#plot and save
pdf("figs/knn_acc_naigh.pdf")
plot(knnFit)
dev.off()

# predict with knn
prediction_knn <- predict(knnFit, newdata = testing)
con_mat<- confusionMatrix(prediction_knn, testing$loan_status)

#print confusion matrix table
con_mat$table



##----------------------------------------- Experiment 2 ------------------------------------------##
# Learning Curve
# Performe knn prediction by increasing data size. Plot data size vs error, and performance

N_iter <- 10    #|> number of iterations for learning curve

# initilzing empty array for some measures
test_err <- 0
test_accur <- 0
test_kap <- 0
train_err <- 0
train_accur <- 0
train_kap <- 0

cpu_time <- 0
best_k <- 0
data_size <- 0

set.seed(500)   #|> setting random seed
data_frac <- 0.8 

for (i in 1:N_iter) { 
    
  
    training <- training[createDataPartition(y=data$loan_status, p = data_frac, list=FALSE),]
    
    ## apply KNN algorithm
    ctrl <- trainControl(method = "repeatedcv",
                        repeats = 5)#,summaryFunction = twoClassSummary)
    
    start_time <- Sys.time() #start the clock
    knnFit <- train(loan_status ~ .,
                    data = training,
                    method = "knn",
                    trControl = ctrl,
                    preProcess = c("center","scale"),
                    tuneLength = 20)
    
    end_time <- Sys.time() # end the clock
    cpu_time[i] <- as.numeric(end_time - start_time)
    
    # predict with knn on test set
    prediction_knn_test <- predict(knnFit, newdata = testing)
    con_mat_test <- confusionMatrix(prediction_knn_test, testing$loan_status)
    
    # predict with knn on train set
    prediction_knn_train <- predict(knnFit, newdata = training)
    con_mat_train <- confusionMatrix(prediction_knn_train, training$loan_status)
    
    
    best_k[i] <-  as.numeric(knnFit$bestTune[1])
    data_size[i] <- nrow(training)
    
    test_err[i] <-  1 - as.numeric(con_mat_test$overall[1])
    test_accur[i] <- as.numeric(con_mat_test$overall[1])
    test_kap[i] <- as.numeric(con_mat_test$overall[2])
    
    train_err[i] <-  1 - as.numeric(con_mat_train$overall[1])
    train_accur[i] <- as.numeric(con_mat_train$overall[1])
    train_kap[i] <- as.numeric(con_mat_train$overall[2])
}

results <- data.frame(test_err,test_accur,test_kap,train_err,train_accur,train_kap,cpu_time, best_k, data_size)

#plot some results
#library(extrafont)
library("Rmisc")

p1 <- ggplot(results, aes(x=data_size)) +
          geom_line(aes(y = train_err, colour = "train")) + 
          geom_line(aes(y = test_err, colour = "test")) +
          #geom_point() +
          theme_bw() +
          #ylim(0.0, 1.) +
          #xlim(0.0, 1) +
          labs(title = "Learning Curve", x = "Data Size", y = "Error", color="") +
          theme(legend.position = c(0.2,0.8),
              axis.title = element_text(size = 16.0),
              axis.text = element_text(size=10, face = "bold"),
              plot.title = element_text(size = 15, hjust = 0.5),
              #text = element_text(family="Times New Roman"),
              axis.text.x = element_text(colour="black"),
              axis.text.y = element_text(colour="black"))

p2 <- ggplot(results, aes(x=data_size, y=cpu_time)) +
          geom_line() + 
          geom_point(colour="red") +
          theme_bw() +
          labs(title = "Performance Benchmarking", x = "Data Size", y = "Clock Time", color="") +
          theme(axis.title = element_text(size = 16.0),
                axis.text = element_text(size=10, face = "bold"),
                plot.title = element_text(size = 15, hjust = 0.5),
              #  text = element_text(family="Times New Roman"),
                axis.text.x = element_text(colour="black"),
                axis.text.y = element_text(colour="black"))

#plot and save
#png("figs/knn_learning_curve.png", width=8.0, height = 4.0, units = "in", res=800)
suppressWarnings(multiplot(p1, p2, cols=2))
#dev.off()
results
