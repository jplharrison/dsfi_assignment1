libraries <- 'libraries.R'
source(libraries)
setpwd()
load(file='sonaTokens.RData')

#### TF-IDF Data ####
#Data Split 60-20-20 sentences for each president to ensure proportional representation
# the train set will be used for model fitting
# the validation set will be used for model selection.
# the test set will be set aside for OOS performance of final model
set.seed(2023)
n <- dim(idf_bag)[1];train_size <- round(0.6*n);val_size <- round(0.2*n);test_size <- round(0.2*n);

train_set <- idf_bag %>% 
  group_by(president_speaker) %>% 
  slice_sample(prop=0.6) %>% 
  ungroup() %>%
  select(sentence_id)

val_set <- idf_bag %>% anti_join(train_set, by = 'sentence_id') %>%
  group_by(president_speaker) %>% 
  slice_sample(prop=0.5) %>% 
  ungroup() %>%
  select(sentence_id)
  
test_set <- idf_bag %>%
  anti_join(train_set, by = 'sentence_id') %>% 
  anti_join(val_set, by = 'sentence_id') %>%
  select(sentence_id) 
  
test_d <- idf_bag %>% 
  right_join(test_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

val_d <- idf_bag %>% 
  right_join(val_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

train_d <- idf_bag %>% 
  right_join(train_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

## TF-IDF Training Data in Matrix Format
train_X <- as.matrix(train_d[,-1])
train_y <- as.integer(train_d$president_speaker)-1

val_X <- as.matrix(val_d[,-1])
val_y <- as.integer(val_d$president_speaker)-1
# Change format of training data to matrix for xgboost
# 0        1       2     3         4         5
# deKlerk  Mandela Mbeki Motlanthe Ramaphosa Zuma

# TF-IDF Training and Validation Responses - One-Hot Encoding
train_yoh <- to_categorical(train_y, num_classes = 6)
val_yoh <- to_categorical(val_y, num_classes = 6)


#### TF-IDF - Extreme Gradient Boosting Classification ####
xgb_models <- list()
xgb_model_summary <- matrix(0, ncol=6, nrow=7); colnames(xgb_model_summary) <- c('Model', 'eta', 'depth','nrounds','train_%','val_%')
for(i in 1:7){
parms <- list(
  objective = "multi:softmax",
  num_class = length(levels(train_d$president_speaker)),
  eta = (i/10),
  max_depth = 6
)
n_rounds <- 150
m_xgb <- xgboost(data = train_X, label = train_y,
                 nrounds = n_rounds, params = parms)
xgb_models[[i]] <- m_xgb

train_fit_xgb  <- predict(m_xgb, train_X)
val_fit_xgb    <- predict(m_xgb, val_X)
train_pred_xgb <- table(train_y, train_fit_xgb); colnames(train_pred_xgb) <- levels(train_d$president_speaker); row.names(train_pred_xgb) <- levels(train_d$president_speaker);
val_pred_xgb   <- table(val_y, val_fit_xgb); colnames(val_pred_xgb) <- levels(train_d$president_speaker); row.names(val_pred_xgb) <- levels(train_d$president_speaker);

#train_pred_xgb
train_pred_xgb_pres  <- diag(train_pred_xgb)/rowSums(train_pred_xgb)  # accuracy per president
train_pred_xgb_total <- sum(diag(train_pred_xgb))/sum(train_pred_xgb) # overall accuracy 0.972
#val_pred_xgb
val_pred_xgb_pres <- diag(val_pred_xgb)/rowSums(val_pred_xgb)  # accuracy per president
val_pred_xgb_total <- sum(diag(val_pred_xgb))/sum(val_pred_xgb) # overall validation accuracy 0.561

xgb_model_summary[i,] <- c(i, parms$eta, parms$max_depth, n_rounds, round(train_pred_xgb_total,3), round(val_pred_xgb_total,3))
}

plot(xgb_model_summary[,2], xgb_model_summary[,5], type='p', ylim=c(0.5,1), pch=19, col='red',
     xlab='eta', ylab='Prediction Error', las=1, main='XGB Model Performance')
legend('right', legend=c('Training Error','Validation Error'), col=c('red','blue'), pch=19, bty='n')
points(xgb_model_summary[,2], xgb_model_summary[,6], pch=19, col='blue')

#save.image(file='xgb_models.RData')
#load(file='xgb_models.RData')

#### TF-IDF Multi-layer perceptron model ####
set.seed(4096)
n_mlp=4

#### TF-IDF MLP with a 1-4 layers of 64 nodes ####
mlp_metrics <- matrix(NA, nrow=n_mlp, ncol=30)
mlp_validation <- matrix(NA, nrow=n_mlp, ncol=2)
m_mlp_base <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = dim(train_X)[2], activation = "relu")

for(m in 1:n_mlp){
  
  m_mlp_base <- m_mlp_base %>%
    layer_dense(units = 64, activation = "relu")
  
  m_mlp_out <- m_mlp_base %>%
    layer_dense(units = 6, activation = "softmax") %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = "accuracy")
  
  fit_mlp <- m_mlp_out %>% fit(train_X, train_yoh, epochs = 30, batch_size = 128, verbose = 0) 
  val_pred_mlp <- m_mlp_out %>% evaluate(val_X, val_yoh)
  
  mlp_metrics[m,] <- fit_mlp$metrics$accuracy
  mlp_validation[m,] <- val_pred_mlp
}

#### TF-IDF MLP with 1-4 layers of 32 nodes ####
mlp_metrics32 <- matrix(NA, nrow=n_mlp, ncol=30)
mlp_validation32 <- matrix(NA, nrow=n_mlp, ncol=2)

m_mlp_base <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = dim(train_X)[2], activation = "relu")

par(mfrow=c(ceiling(n_mlp/2), 2))
for(m in 1:n_mlp){
  
  m_mlp_base <- m_mlp_base %>%
    layer_dense(units = 32, activation = "relu")
  
  m_mlp_out <- m_mlp_base %>%
    layer_dense(units = 6, activation = "softmax") %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = "accuracy")
  
  fit_mlp <- m_mlp_out %>% fit(train_X, train_yoh, epochs = 30, batch_size = 128, verbose = 0) 
  val_pred_mlp <- m_mlp_out %>% evaluate(val_X, val_yoh)
  
  mlp_metrics32[m,] <- fit_mlp$metrics$accuracy
  mlp_validation32[m,] <- val_pred_mlp
}

# TF-IDF MLP Prediction Accuracy
par(mfrow=c(1,1))
plot(mlp_validation[,2], xlab='n layers', col='blue', pch=4, ylim=c(0.4,0.6), ylab='Prediction Accuracy', lwd=2)
points(mlp_validation32[,2], col='red', pch=3, lwd=2)
legend('bottomleft', legend=c('Validation: 64 node','Validation: 32 nodes'), pch=c(4,3), col=c('blue','red'), bty='n')
points(mlp_metrics[,30], col='blue', pch=4)
points(mlp_metrics[,30], col='red', pch=3)

save.image(file='mlp_models.RData')

#### TF-IDF - Multinomial Logit Regression ####

m_multinom <- multinom(train_y~train_X)

train_fit_multinom <- predict(m_multinom, newdata = train_X, "class")
val_fit_multinom   <- predict(m_multinom, newdata = val_X, "class")
train_pred_multinom<- table(train_y, train_fit_multinom); colnames(train_pred_multinom) <- levels(train_d$president_speaker); row.names(train_pred_multinom) <- levels(train_d$president_speaker);
val_pred_multinom  <- table(val_y, val_fit_multinom); colnames(val_pred_multinom) <- levels(train_d$president_speaker); row.names(val_pred_multinom) <- levels(train_d$president_speaker);

train_pred_multinom_total <- sum(diag(train_pred_multinom))/sum(train_pred_multinom) 
val_pred_multinom_total <- sum(diag(val_pred_multinom))/sum(val_pred_multinom) 

train_accuracy <- accuracy(train_fit_multinom, train_d[,1])
val_accuracy <- accuracy(val_fit_multinom, val_d[,1])
f1_score <- F1_Score(predictions, test_data$response_column, positive = NULL)
confusion_matrix <- confusionMatrix(predictions, test_data$response_column)








#### BAG OF WORDS ####

#### BAG OF WORDS Data ####
set.seed(2023)
n <- dim(bag_of_words)[1];train_size <- round(0.6*n);val_size <- round(0.2*n);test_size <- round(0.2*n);

BoW_train_set <- bag_of_words %>% 
  group_by(president_speaker) %>% 
  slice_sample(prop=0.6) %>% 
  ungroup() %>%
  select(sentence_id)

BoW_val_set <- bag_of_words %>% anti_join(BoW_train_set, by = 'sentence_id') %>%
  group_by(president_speaker) %>% 
  slice_sample(prop=0.5) %>% 
  ungroup() %>%
  select(sentence_id)

BoW_test_set <- bag_of_words %>%
  anti_join(BoW_train_set, by = 'sentence_id') %>% 
  anti_join(BoW_val_set, by = 'sentence_id') %>%
  select(sentence_id) 

BoW_test_d <- bag_of_words %>% 
  right_join(BoW_test_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

BoW_val_d <- bag_of_words %>% 
  right_join(BoW_val_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

BoW_train_d <- bag_of_words %>% 
  right_join(BoW_train_set, by = 'sentence_id') %>%
  select(-sentence_id) %>% mutate(president_speaker = factor(president_speaker))

## BoW Training Data in Matrix Format
BoW_train_X <- as.matrix(BoW_train_d[,-1])
BoW_train_y <- as.integer(BoW_train_d$president_speaker)-1

BoW_val_X <- as.matrix(BoW_val_d[,-1])
BoW_val_y <- as.integer(BoW_val_d$president_speaker)-1
# Change format of training data to matrix for xgboost
# 0        1       2     3         4         5
# deKlerk  Mandela Mbeki Motlanthe Ramaphosa Zuma

# BoW training and validation Responses - One-Hot Encoding
BoW_train_yoh <- to_categorical(BoW_train_y, num_classes = 6)
BoW_val_yoh <- to_categorical(BoW_val_y, num_classes = 6)


#### BAG OF WORDS - Extreme Gradient Boosting Classification ####
BoW_xgb_models <- list()
BoW_xgb_model_summary <- matrix(0, ncol=6, nrow=7); colnames(BoW_xgb_model_summary) <- c('Model', 'eta', 'depth','nrounds','BoW_train_%','BoW_val_%')
for(i in 1:7){
  parms <- list(
    objective = "multi:softmax",
    num_class = length(levels(BoW_train_d$president_speaker)),
    eta = (i/10),
    max_depth = 6
  )
  n_rounds <- 150
  m_xgb <- xgboost(data = BoW_train_X, label = BoW_train_y,
                   nrounds = n_rounds, params = parms)
  BoW_xgb_models[[i]] <- m_xgb
  
  BoW_val_X <- as.matrix(BoW_val_d[,-1])
  BoW_val_y <- as.integer(BoW_val_d$president_speaker)-1
  BoW_train_fit_xgb  <- predict(m_xgb, BoW_train_X)
  BoW_val_fit_xgb    <- predict(m_xgb, BoW_val_X)
  BoW_train_pred_xgb <- table(BoW_train_y, BoW_train_fit_xgb); colnames(BoW_train_pred_xgb) <- levels(BoW_train_d$president_speaker); row.names(BoW_train_pred_xgb) <- levels(BoW_train_d$president_speaker);
  BoW_val_pred_xgb   <- table(BoW_val_y, BoW_val_fit_xgb); colnames(BoW_val_pred_xgb) <- levels(BoW_train_d$president_speaker); row.names(BoW_val_pred_xgb) <- levels(BoW_train_d$president_speaker);
  
  #BoW_train_pred_xgb
  BoW_train_pred_BoW_xgb_pres  <- diag(BoW_train_pred_xgb)/rowSums(BoW_train_pred_xgb)  # accuracy per president
  BoW_train_pred_BoW_xgb_total <- sum(diag(BoW_train_pred_xgb))/sum(BoW_train_pred_xgb) # overall accuracy 0.972
  #BoW_val_pred_xgb
  BoW_val_pred_BoW_xgb_pres <- diag(BoW_val_pred_xgb)/rowSums(BoW_val_pred_xgb)  # accuracy per president
  BoW_val_pred_BoW_xgb_total <- sum(diag(BoW_val_pred_xgb))/sum(BoW_val_pred_xgb) # overall BoW_validation accuracy 0.561
  
  BoW_xgb_model_summary[i,] <- c(i, parms$eta, parms$max_depth, n_rounds, round(BoW_train_pred_BoW_xgb_total,3), round(BoW_val_pred_BoW_xgb_total,3))
}

plot(BoW_xgb_model_summary[,2], BoW_xgb_model_summary[,5], type='p', ylim=c(0.5,1), pch=19, col='red',
     xlab='eta', ylab='Prediction Error', las=1, main='XGB Model Performance')
legend('right', legend=c('BoW_training Error','BoW_validation Error'), col=c('red','blue'), pch=19, bty='n')
points(BoW_xgb_model_summary[,2], BoW_xgb_model_summary[,6], pch=19, col='blue')

#save.image(file='xgb_models.RData')
#load(file='xgb_models.RData')

#### BAG OF WORDS - Multilayer Perceptron ####
set.seed(4096)
n_mlp=4

#### BAG OF WORDS MLP with a 1-4 layers of 64 nodes ####
BoW_mlp_metrics <- matrix(NA, nrow=n_mlp, ncol=30)
BoW_mlp_validation <- matrix(NA, nrow=n_mlp, ncol=2)
m_BoW_mlp_base <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = dim(BoW_train_X)[2], activation = "relu")

for(m in 1:n_mlp){
  
  m_BoW_mlp_base <- m_BoW_mlp_base %>%
    layer_dense(units = 64, activation = "relu")
  
  m_BoW_mlp_out <- m_BoW_mlp_base %>%
    layer_dense(units = 6, activation = "softmax") %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = "accuracy")
  
  fit_mlp <- m_BoW_mlp_out %>% fit(BoW_train_X, BoW_train_yoh, epochs = 30, batch_size = 128, verbose = 0) 
  BoW_val_pred_mlp <- m_BoW_mlp_out %>% evaluate(BoW_val_X, BoW_val_yoh)
  
  BoW_mlp_metrics[m,] <- fit_mlp$metrics$accuracy
  BoW_mlp_validation[m,] <- BoW_val_pred_mlp
}

#### BAG OF WORDS MLP with 1-4 layers of 32 nodes ####
BoW_mlp_metrics32 <- matrix(NA, nrow=n_mlp, ncol=30)
BoW_mlp_validation32 <- matrix(NA, nrow=n_mlp, ncol=2)

m_BoW_mlp_base <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = dim(BoW_train_X)[2], activation = "relu")

par(mfrow=c(ceiling(n_mlp/2), 2))
for(m in 1:n_mlp){
  
  m_BoW_mlp_base <- m_BoW_mlp_base %>%
    layer_dense(units = 32, activation = "relu")
  
  m_BoW_mlp_out <- m_BoW_mlp_base %>%
    layer_dense(units = 6, activation = "softmax") %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = "accuracy")
  
  fit_mlp <- m_BoW_mlp_out %>% fit(BoW_train_X, BoW_train_yoh, epochs = 30, batch_size = 128, verbose = 0) 
  BoW_val_pred_mlp <- m_BoW_mlp_out %>% evaluate(BoW_val_X, BoW_val_yoh)
  
  BoW_mlp_metrics32[m,] <- fit_mlp$metrics$accuracy
  BoW_mlp_validation32[m,] <- BoW_val_pred_mlp
}

# BAG OF WORDS MLP Prediction Accuracy
par(mfrow=c(1,1))
plot(BoW_mlp_validation[,2], xlab='n layers', col='blue', pch=4, ylim=c(0.4,0.6), ylab='Prediction Accuracy', lwd=2)
points(BoW_mlp_validation32[,2], col='red', pch=3, lwd=2)
legend('bottomleft', legend=c('BoW_validation: 64 node','BoW_validation: 32 nodes'), pch=c(4,3), col=c('blue','red'), bty='n')
points(BoW_mlp_metrics[,30], col='blue', pch=4)
points(BoW_mlp_metrics[,30], col='red', pch=3)

save.image(file='BoW_mlp_models.RData')



#### MODEL COMPARISON #### 
par(mfrow=c(1,2))
plot(BoW_mlp_validation[,2], main='MLP Model Validation', xaxt='n', xlab='n Hidden Layers', col='blue', pch=1, ylim=c(0.48,0.62), ylab='Validation Accuracy', lwd=2)
legend('bottomleft', bty='n', cex=0.9,
       legend=c('MLP BoW 64 nodes','MLP BoW 32 nodes','MLP TF-IDF 64 nodes','MLP TF-IDF 32 nodes'), 
       pch=c(1,3,4,5), 
       col=c('blue','blue','red','red'))
points(BoW_mlp_validation32[,2], col='blue', pch=3, lwd=2)
points(mlp_validation32[,2], col='red', pch=4, lwd=2)
points(mlp_validation[,2], col='red', pch=5, lwd=2)
axis(1, at = c(1,2,3,4), labels = c(1,2,3,4))

plot(BoW_xgb_model_summary[,6], main='XGB Model Validation', xaxt='n', xlab='eta', col='blue', pch=3, ylim=c(0.48,0.62), ylab='Validation Accuracy', lwd=2)
legend('bottomleft', bty='n', cex=0.9,
       legend=c('XGB BoW','XGB TF-IDF'), 
       pch=c(1,3,4,5),
       col=c('blue','red'))
points(xgb_model_summary[,6], col='red', pch=4, lwd=2)
axis(1, at = 1:7, labels = seq(0.1,0.7,0.1))




points(BoW_mlp_metrics[,30], col='blue', pch=4)
points(BoW_mlp_metrics[,30], col='red', pch=3)
