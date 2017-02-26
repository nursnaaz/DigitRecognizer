# Clean workspace
rm(list=ls())
setwd()

# Loading data and set up
#-------------------------------------------------------------------------------

# Load train and test datasets
train <- read.csv("train_sample.csv")
test <- read.csv("test_sample.csv")

###########################################
########################################### Viewing the Images ########################
#train = subset(train,label==1)
train_mat <- as.matrix(train)

## Color ramp def.
colors <- c('white','black')
cus_col <- colorRampPalette(colors=colors)

## Plot the random 12 images
par(mfrow=c(4,3),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
sm = sample(nrow(train_mat), 12)
for(di in sm)
{
  
  z <- array(train_mat[di,-1],dim=c(28,28))
  z <- z[,28:1] ##right side up
  z <- matrix(as.numeric(z), 28, 28)
  image(1:28,1:28,z,main=train_mat[di,1],col=cus_col(256))
}  
  




library(h2o)
h2o.shutdown()
h2o.init(nthreads = -1,max_mem_size = '5g')
rm(list=ls())
train <- h2o.importFile("train.csv")
test  <- h2o.importFile("test.csv")
str(train)

y <- "label"
x <- setdiff(colnames(train),y)
train["label"]=as.factor(train["label"])
dlModel <- h2o.deeplearning(x = x, y = y, 
                            training_frame=train,
                            activation = "RectifierWithDropout",
                            hidden = c(250,150),
                            input_dropout_ratio = 0.01,
                            l1 = 1e-5,
                            epochs = 200)

# View specified parameters of the deep learning model
dlModel@parameters

# Examine the performance of the trained model
dlModel # display all performance metrics

# Metrics
h2o.performance(dlModel) 

# Get MSE only
h2o.mse(dlModel)

# Classify the test set (predict class labels)
# This also returns the probability for each class
pred = h2o.predict(dlModel, test)


finaltestres <- pred[,1]
result = as.data.frame(finaltestres)
write.csv(x = result, file = "result.csv")


h2o.shutdown(prompt = F)
