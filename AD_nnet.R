#!/usr/bin/Rscript

#loading reuired libraries
library("ggplot2")
library("lattice") 
library("caret")
library("plyr")
library(doMC)
registerDoMC(cores = 4)


## loading cleaned data
training <- read.table("clean_data/adult_train.txt", sep = "", header = TRUE)
testing <- read.table("clean_data/adult_test.txt", sep = "", header = TRUE)


## temp sub data for debugging --------------------------------------------------------------
sub_data <- training[createDataPartition(y=training$income, p = 0.04, list=FALSE),]
training <- sub_data
##--------------------------------------------------------------------------------------------

## setting seed for random number generator
set.seed(300)

training[training$income == 0,]$income <- "No"
training[training$income == 1,]$income <- "Yes"
testing[testing$income == 0,]$income <- "No"
testing[testing$income == 1,]$income <- "Yes"



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

nnetGrid <- expand.grid(size = seq(from = 2, to = 5, by = 1),
                         decay = seq(from = 0.005, to = 0.02, by = 0.005)) #0.005 to 0.02 in steps of 0.0025

nnetFit <- train(factor(income) ~ .,
                 data = training,
                 method = "nnet",
                 metric = "ROC",
                 trControl = fitControl,
                 preProcess = c("center","scale"),
                 tuneGrid = nnetGrid,
                 verbose = FALSE)

prediction_nnet <- predict(nnetFit, newdata=testing, type = "raw")
con_mat_nnet <- confusionMatrix(prediction_nnet, testing$income)

#plot and save
pdf("figs/AD_nnet_ROC_units_weight.pdf")
plot(nnetFit)
dev.off()

best_size <- as.numeric(nnetFit$bestTune[1])
best_decay <- as.numeric(nnetFit$bestTune[2])

## outputting some results
write.table(con_mat_nnet$table, file = "output/AD_confusion_mat_nnet.txt", row.names = TRUE, col.names = TRUE, sep = "  ")
write.table(con_mat_nnet$overall, file = "output/AD_nnet_accuracy.txt", row.names = TRUE, col.names = TRUE, sep = "  ")
write.table(nnetFit$bestTune, file = "output/AD_bestTune.txt", row.names = TRUE, col.names = TRUE, sep = "  ")


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
train_frac <- 0.8

training1 <- training


for (i in 1:N_iter) { 
  
  
  new_train <- training1 
  training1 <- new_train[createDataPartition(y = new_train$income, p = 0.8, list = FALSE),]
  validation1 <- training[createDataPartition(y = training1$income, p = 0.3, list = FALSE), ]
  
  
  start_time <- Sys.time() #start the clock--------------------------------------------------------
  ## building a net model  
  nnetGrid1 <- expand.grid(size = best_size,
                          decay = best_decay)
  
  fitControl1 <- trainControl(method = "none", 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary)
  
  
  nnetFit1 <- train(factor(income) ~ .,
                   data = training1,
                   method = "nnet",
                   metric = "ROC",
                   trControl = fitControl1,
                   preProcess = c("center","scale"),
                   tuneGrid = nnetGrid1,
                   verbose = FALSE)
  end_time <- Sys.time() # end the clock-------------------------------------------------------------
  
  
  ## making a prediction
  prediction_nnet_test <- predict(nnetFit1, testing, type = "raw")
  con_mat_test <- confusionMatrix(prediction_nnet_test, testing$income)
  
  prediction_nnet_train <- predict(nnetFit1, validation1, type = "raw")
  con_mat_train <- confusionMatrix(prediction_nnet_train, validation1$income)
  
  
  cpu_time[i] <- round(as.numeric(end_time - start_time),3)
  data_size[i] <- nrow(training1)
  test_accur[i] <- round(as.numeric(con_mat_test$overall[1]),3)
  test_kap[i] <- round(as.numeric(con_mat_test$overall[2]),3)
  train_accur[i] <- round(as.numeric(con_mat_train$overall[1]),3)
  train_kap[i] <- round(as.numeric(con_mat_train$overall[2]),3)
  
}

results <- data.frame(test_accur,test_kap,train_accur,train_kap, cpu_time, data_size)
write.table(results, file = "output/AD_learning_results_nnet.txt", row.names = TRUE, col.names = TRUE, sep = "  ")



#plot some results
pl <- ggplot(results, aes(x=data_size)) +
  geom_line(aes(y = train_accur, colour = "train")) + 
  geom_line(aes(y = test_accur, colour = "test")) +
  geom_point(aes(y = train_accur,colour = "train")) + 
  geom_point(aes(y = test_accur,colour = "test")) +
  theme_bw() +
  labs(title = "Learning Curve NNet Adult Data", x = "Training Size", y = "Accuracy", color="") +
  theme(legend.position = c(0.8,0.8),
        axis.title = element_text(size = 16.0),
        axis.text = element_text(size=10, face = "bold"),
        plot.title = element_text(size = 15, hjust = 0.5),
        #text = element_text(family="Times New Roman"),
        axis.text.x = element_text(colour="black"),
        axis.text.y = element_text(colour="black"))


#plot some results
#plot and save
png("figs/AD_nnet_learning_curve.png", width = 5.0, height = 4.0, units = "in", res = 800)
pl
dev.off()



