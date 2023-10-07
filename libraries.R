rm(list=ls())
setpwd <- function(){setwd(dirname(rstudioapi::getActiveDocumentContext()$path))}
library(reticulate)
#use_python("C:/Users/User/AppData/Local/Programs/Python/Python311/python.exe", required = TRUE)
library(keras)
#install_keras()
library(tensorflow)
#install_tensorflow()
#install_tensorflow(envname = "r-pytensorflow")
#use_virtualenv("r-pytensorflow", required = T)
#use_virtualenv("r-tensorflow", required = T)
#use_virtualenv("C:/Users/User/anaconda3/envs/r-tensorflow/python.exe", required = T)
#install_keras(envname = "r-pytensorflow")
library(tidyverse)
library(ggplot2)
library(plotly)
library(stringr)
library(rvest)
library(lubridate)
library(tidytext)
library(rpart)
library(xgboost)
library(nnet)
library(caret)
library(Metrics)








