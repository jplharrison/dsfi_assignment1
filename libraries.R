rm(list=ls())
setpwd <- function(){setwd(dirname(rstudioapi::getActiveDocumentContext()$path))}
library(keras)
library(tensorflow)
library(tidyverse)
library(ggplot2)
library(plotly)
library(stringr)
library(rvest)
library(lubridate)
library(tidytext)
library(rpart)