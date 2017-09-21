#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library("Rmisc")
library(doMC)
registerDoMC(cores = 4)

## setting seed for random number generator
set.seed(300)

## loading cleaned data
data <- read.table("clean_data/loan.txt", sep = "", header = TRUE)

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

## extract some % of data and perfor hyperparameter tuning with 5 fold cross validation
sub_data <- data[createDataPartition(y=data$loan_status, p = 0.1, list=FALSE),]

# break sub data into train and test sets
indx <- createDataPartition(y=sub_data$loan_status, p = 0.70, list=FALSE)
training <- sub_data[indx, ]
testing <- sub_data[-indx, ] 

## apply KNN algorithm
set.seed(400)
ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 2)
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

## output confusion matrix
write.table(con_mat$table, file = "output/LC_confusion_mat_knn.txt", row.names = TRUE, col.names = TRUE, sep = "  ")
write.table(knnFit, file = "output/LC_Fit_Info_knn.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


##----------------------------------------- Experiment 2 ------------------------------------------##
# Learning Curve
# Performe knn prediction by increasing data size. Plot data size vs error, and performance

N_iter <- 20  #|> number of iterations for learning curve

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
train_frac <- 0.8 

training1 <- training

ctrl <- trainControl(method = "cv")
grid <-  expand.grid(k = 41)

for (i in 1:N_iter) { 
    
    new_train <- training1 
    training1 <- new_train[createDataPartition(y=new_train$loan_status, p = train_frac, list=FALSE),]
    
    ## apply KNN algorithm
    start_time <- Sys.time() #start the clock--------------------------------------------------------------
    knnFit <- train(loan_status ~ .,
                    data = training1,
                    method = "knn",
                    trControl = ctrl,
                    preProcess = c("center","scale"),
                    #tuneLength = 2,
                    tuneGrid=grid)
    
    end_time <- Sys.time() # end the clock-----------------------------------------------------------------

    cpu_time[i] <- as.numeric(end_time - start_time)
    
    ## predict with knn on test set
    prediction_knn_test <- predict(knnFit, newdata = testing)
    con_mat_test <- confusionMatrix(prediction_knn_test, testing$loan_status)
    
    ## predict with knn on validation set
    validation <- training1[createDataPartition(y=training1$loan_status, p = 0.3, list=FALSE),]
    prediction_knn_train <- predict(knnFit, newdata = validation)
    con_mat_train <- confusionMatrix(prediction_knn_train, validation$loan_status)
    
    data_size[i] <- nrow(training1)
    test_accur[i] <- round(as.numeric(con_mat_test$overall[1]),3)
    test_kap[i] <- round(as.numeric(con_mat_test$overall[2]),3)
    train_accur[i] <- round(as.numeric(con_mat_train$overall[1]),3)
    train_kap[i] <- round(as.numeric(con_mat_train$overall[2]),3)
}

results <- data.frame(test_accur, test_kap, train_accur, train_kap, cpu_time, data_size)
write.table(results, file = "output/LC_learning_results_knn.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


#plot some results
pl <- ggplot(results, aes(x=data_size)) +
          geom_line(aes(y = train_accur, colour = "train")) + 
          geom_line(aes(y = test_accur, colour = "test")) +
          geom_point(aes(y = train_accur,colour = "train")) + 
          geom_point(aes(y = test_accur,colour = "test")) +
          #geom_point() +
          theme_bw() +
          #ylim(0.0, 1.) +
          #xlim(0.0, 1) +
          labs(title = "Learning Curve KNN", x = "Training Size", y = "Accuracy", color="") +
          theme(legend.position = c(0.2,0.8),
              axis.title = element_text(size = 16.0),
              axis.text = element_text(size=10, face = "bold"),
              plot.title = element_text(size = 15, hjust = 0.5),
              #text = element_text(family="Times New Roman"),
              axis.text.x = element_text(colour="black"),
              axis.text.y = element_text(colour="black"))

#plot and save
png("figs/LC_knn_learning_curve.png", width=8.0, height = 4.0, units = "in", res=800)
pl
dev.off()
