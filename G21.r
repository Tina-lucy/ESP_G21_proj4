netup <- function(d) {
  # this function sets up a list representing the network
  
  # the parameter d is a vector giving the number of nodes in each layer

  # the network list containing
  # h: a list of nodes for each layer;
  # W: a list of weight matrices;
  # b: a list of offset vectors

  h <- W <- b <- list()
  for(i in 1:(length(d)-1)) {
    # a vector of length d[l] which contains the node values for layer l
    h <- c(h, list(rep(0, d[i])))

    # initialize the elements of weight matrix and offset vector that 
    # linking layer l to layer l+1 with Uni(0, 0.2) random deviates
    W <- c(W, list(matrix(runif(d[i+1] * d[i], 0, 0.2), d[i+1])))
    b <- c(b, list(runif(d[i+1], 0, 0.2)))
  }

  # add the vector representing last layer to the list h
  h <- c(h, list(rep(0, d[length(d)])))
  
  list(h = h, W = W, b = b)
}

forward <- function(nn, inp) {
  # this function is for computing 
  # the remaining node values implied by given data

  # nn is the network list
  # inp is the given data
  
  # the values of the nodes in first layer are set as given data
  nn$h[[1]] <- inp

  # calculate the values of nodes in each layer
  for(i in 1:length(nn$W)) {
    # get h^{l+1} = W^{l} * h^{l} + b^{l}, and set negative values to 0
    z <- nn$W[[i]] %*% nn$h[[i]] + nn$b[[i]]
    z[z < 0] <- 0
    nn$h[[i+1]] <- z
  }

  # return the updated network list
  nn
}

backward <- function(nn, k) {
  # this function is for computing the derivatives of the loss 
  # corresponding to output class for network

  # nn is the network list returned from forward function
  # k is the output class

  # get array of exp(h)
  exp_h <- exp(nn$h[[length(nn$h)]])

  # if j = k, dh^{L}_j = exp(h_j) / sum(exp(h_i)) - 1
  # else dh^{L}_j = exp(h_j) / sum(exp(h_i))
  dh_L <- exp_h/sum(exp_h)
  dh_L[k] <- dh_L[k] - 1
  dh <- list(dh_L)
  
  # create empty list for dW and db
  dW <- db <- list()
  
  # loop from higher layer
  for(l in length(nn$W):1) {
    # create empty array for d^{l+1}
    d_next_lv <- rep(0, length(nn$h[[l+1]]))

    # set d^{l+1}_j = dh^{l+1}_j when dh^{l+1}_j is positive
    h_positive <- nn$h[[l+1]] > 0
    d_next_lv[h_positive] <- dh[[1]][h_positive]

    # update dh, with element dh^{l} = W^{lT} * d^{l+1}
    # update dW, with element dW^{l} = d^{l+1} * h^{lT}
    # update db, with element db^{l} = d^{l+1}

    # notice that new element is added to the beginning of each list,
    # therefore after updating, the first element of dh list is dh^{l},
    # so in the next loop, dh[[1]] = dh^{l+1}
    dh <- c(list(t(nn$W[[l]]) %*% d_next_lv), dh)
    dW <- c(list(d_next_lv %*% t(nn$h[[l]])), dW)
    db <- c(list(d_next_lv), db)
  }
  
  # add dh, dW and db to the network list, return updated network list
  c(nn, list(dh = dh, dW = dW, db = db))
}

train <- function(nn, inp, k, eta = .01, mb = 10, nstep = 10000) {
  # this function is for training the network
  
  # nn is the original network list
  # inp is the data matrix
  # k is the vector of corresponding labels (1, 2, 3 . . . )
  # eta is the step size
  # mb is the number of data to randomly sample to compute the gradient
  # nstep is the number of optimization steps to take
  
  # the trained network list will be returned
  
  # optimize for nstep loops
  for(step in 1:nstep) {
    
    # in each loop, take mb random samples to compute the gradient
    for(i in 1:mb) {
      
      # get one random sample data and label
      ind <- sample(length(k), 1)
      sample_data <- inp[ind, ]
      label <- k[ind]
      
      # use data and label to derive the gradient
      update_nn <- backward(forward(nn, sample_data), label)
      
      # update matrix W and vector b with step size eta and
      # the average of gradients get from corresponding random data
      for(l in 1:length(update_nn$dW)) {
        nn$W[[l]] <- nn$W[[l]] - eta * update_nn$dW[[l]]/mb
        nn$b[[l]] <- nn$b[[l]] - eta * update_nn$db[[l]]/mb
      }
    }
  }
  nn
}

test <- function(inp, k, d = c(4, 8, 7, 3), interval = 5) {
  # this function is for testing the trained network list
  
  # inp is the data containing training part and testing part
  # k is the corresponding label (can not be numeric)
  # d is the number of nodes of network in each layer
  # interval is the sampling interval of the test set
  
  # the output is the misclassification rate
  
  # set up the original network list
  nn <- netup(d)
  
  # get the index of test set
  test_ind <- 1:(nrow(iris)/interval)*interval
  
  # separate train set and test set from data and label
  train_inp <- inp[-test_ind, ]
  train_label <- as.numeric(k)[-test_ind]
  test_inp <- inp[test_ind, ]
  test_label <- as.numeric(k)[test_ind]
  
  # get the trained model with training data and training label
  trained_model <- train(nn, train_inp, train_label)

  # create a vector to store predicting label
  result <- rep(0, length(test_ind))
  
  # predict for each data
  for(i in 1:length(test_ind)) {
    # put testing data into trained model
    n_temp <- forward(trained_model, test_inp[i, ])
    
    # get the class with highest probability as predicted class
    result[i] <- which.max(n_temp$h[[length(d)]])
  }
  
  # compare with real label and get the misclassification rate
  sum(result != test_label)/length(test_label) * 100
}

set.seed(73)
# separate the feature and species of raw data
feature <- as.matrix(iris[, 1:4])
species <- factor(as.array(iris$Species))

# get the error rate from test function, print it
error_rate <- test(feature, species)
cat("The misclassification rate is ", error_rate, "%.", sep = '')