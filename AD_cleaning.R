#!/usr/bin/Rscript

# loading locally stored data
train <- read.csv("data/adult.data", na.strings = c("NA",""), header = FALSE)
test <- read.csv("data/adult.test", na.strings = c("NA",""), header = FALSE, skip = 1)

# assign column names
colnames <- c("age","type_employer", "fnlwgt", "education", 
              "education_num","marital", "occupation",
              "relationship", "race","sex","capital_gain",
              "capital_loss", "hr_per_week","country", "income")
colnames(train) <- colnames
colnames(test) <- colnames
train$set <- "train"
test$set <- "test"
data <- rbind(train,test)
data$education_num <- NULL
data$fnlwgt <- NULL

data$type_employer = as.character(data$type_employer)
data$occupation = as.character(data$occupation)
data$country = as.character(data$country)
data$marital = as.character(data$marital)
data$marital[data$marital=="Never-married"] = "Never-Married"
data$marital[data$marital=="Married-spouse-absent"] = "Not-Married"
data$country[data$country=="India"] = "British-Commonwealth"
data$country[data$country=="Iran"] = "Other"
data$country[data$country=="Ireland"] = "British-Commonwealth"
data$country[data$country=="Italy"] = "Euro_1"
data$country[data$country=="Jamaica"] = "Latin-America"
data$country[data$country=="Japan"] = "Other"
data$country[data$country=="Laos"] = "SE-Asia"
data$country[data$country=="Mexico"] = "Latin-America"
data$country[data$country=="Nicaragua"] = "Latin-America"
data$country[data$country=="Outlying-US(Guam-USVI-etc)"] = "Latin-America"
data$country[data$country=="Peru"] = "South-America"
data$country[data$country=="Philippines"] = "SE-Asia"
data$country[data$country=="Poland"] = "Euro_2"
data$country[data$country=="Portugal"] = "Euro_2"
data$country[data$country=="Puerto-Rico"] = "Latin-America"
data$country[data$country=="Scotland"] = "British-Commonwealth"
data$country[data$country=="South"] = "Euro_2"
data$country[data$country=="Taiwan"] = "China"
data$country[data$country=="Thailand"] = "SE-Asia"
data$country[data$country=="Trinadad&Tobago"] = "Latin-America"
data$country[data$country=="United-States"] = "United-States"
data$country[data$country=="Vietnam"] = "SE-Asia"
data$country[data$country=="Yugoslavia"] = "Euro_2"

data$education = gsub("^10th","Dropout",data$education)
data$education = gsub("^11th","Dropout",data$education)
data$education = gsub("^12th","Dropout",data$education)
data$education = gsub("^1st-4th","Dropout",data$education)
data$education = gsub("^5th-6th","Dropout",data$education)
data$education = gsub("^7th-8th","Dropout",data$education)
data$education = gsub("^9th","Dropout",data$education)
data$education = gsub("^Assoc-acdm","Associates",data$education)
data$education = gsub("^Assoc-voc","Associates",data$education)
data$education = gsub("^Bachelors","Bachelors",data$education)
data$education = gsub("^Doctorate","Doctorate",data$education)
data$education = gsub("^HS-Grad","HS-Graduate",data$education)
data$education = gsub("^Masters","Masters",data$education)
data$education = gsub("^Preschool","Dropout",data$education)
data$education = gsub("^Prof-school","Prof-School",data$education)
data$education = gsub("^Some-college","HS-Graduate",data$education)

data$type_employer = gsub("^Federal-gov","Federal-Govt",data$type_employer)
data$type_employer = gsub("^Local-gov","Other-Govt",data$type_employer)
data$type_employer = gsub("^State-gov","Other-Govt",data$type_employer)
data$type_employer = gsub("^Private","Private",data$type_employer)
data$type_employer = gsub("^Self-emp-inc","Self-Employed",data$type_employer)
data$type_employer = gsub("^Self-emp-not-inc","Self-Employed",data$type_employer)
data$type_employer = gsub("^Without-pay","Not-Working",data$type_employer)
data$type_employer = gsub("^Never-worked","Not-Working",data$type_employer)

data$occupation = gsub("^Adm-clerical","Admin",data$occupation)
data$occupation = gsub("^Armed-Forces","Military",data$occupation)
data$occupation = gsub("^Craft-repair","Blue-Collar",data$occupation)
data$occupation = gsub("^Exec-managerial","White-Collar",data$occupation)
data$occupation = gsub("^Farming-fishing","Blue-Collar",data$occupation)
data$occupation = gsub("^Handlers-cleaners","Blue-Collar",data$occupation)
data$occupation = gsub("^Machine-op-inspct","Blue-Collar",data$occupation)
data$occupation = gsub("^Other-service","Service",data$occupation)
data$occupation = gsub("^Priv-house-serv","Service",data$occupation)
data$occupation = gsub("^Prof-specialty","Professional",data$occupation)
data$occupation = gsub("^Protective-serv","Other-Occupations",data$occupation)
data$occupation = gsub("^Sales","Sales",data$occupation)
data$occupation = gsub("^Tech-support","Other-Occupations",data$occupation)
data$occupation = gsub("^Transport-moving","Blue-Collar",data$occupation)
#data[data$race == "Amer-Indian-Eskimo",]$race <- "Amer-Indian"
#data[data$race == "Asian-Pac-Islander",]$race <- "Asian"

data[["capital_gain"]] <- ordered(cut(data$capital_gain,
                                      c(-Inf,
                                        0,
                                        median(data[["capital_gain"]][data[["capital_gain"]] >0]),
                                        Inf)),
                                  labels = c("None", "Low", "High"))

data[["capital_loss"]] <- ordered(cut(data$capital_loss,
                                      c(-Inf,
                                        0,
                                        median(data[["capital_loss"]][data[["capital_loss"]] >0]),
                                        Inf)),
                                  labels = c("None", "Low", "High"))

is.na(data) = data=='?'
is.na(data) = data==' ?'
data = na.omit(data)

data$marital = factor(data$marital)
data$education = factor(data$education)
data$country = factor(data$country)
data$type_employer = factor(data$type_employer)
data$occupation = factor(data$occupation)
data$race = factor(data$race)
data$sex = factor(data$sex)
data$relationship = factor(data$relationship)

data$country <- NULL
#------------ reduce levels from 4 to 2 ----------------------------------
data$income <- gsub("K.", "", data$income)
data$income <- gsub("K", "", data$income)
data$income <- as.factor(data$income)
data$income = as.factor(ifelse(data$income == data$income[1],"low","high"))
#-------------------------------------------------------------------------

## train test split
train <- data[data$set == "train",]
test <- data[data$set == "test",]
train$set <- NULL
test$set <- NULL

write.table(train, file = "clean_data/adult_train.txt", row.names = FALSE, col.names = TRUE, sep = "  ")
write.table(test, file = "clean_data/adult_test.txt", row.names = FALSE, col.names = TRUE, sep = "  ")
