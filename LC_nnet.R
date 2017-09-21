












## Neural Network Model
set.seed(2017)
model_nnet <- nnet(factor(status) ~  grade + int_rate + term +revol_util, 
                   data = train, size = 40, 
                   decay = 5e-4, maxit = 50)

prediction_nnet <- predict(model_nnet, newdata=test, type = "class")
conf_nnet <- confusionMatrix(prediction_nnet, test$status)
conf_nnet$overall[1]
