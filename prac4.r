netup <- function(d) {
  # Determine the number of layers
  layer_n = length(d)
  
  # First generate the first layer h
  h = rep(0,d[1])
  
  # Generate all remaining h
  for (i in 2:layer_n){
    hi = rep(0, d[i])
    h = list(h, hi)
  }
  
  # First generate the weights from the first layer to the second layer
  W_con = runif(d[2]*d[1], 0, 0.2)
  W = matrix(W_con,d[2],d[1])
   
  # Generate all remaining weights
  for (i in 2:((layer_n-1))){
    Wi_con = runif(d[i+1]*d[i], 0, 0.2)
    Wi = matrix(Wi_con, d[i+1],d[i])
    W = list(W, Wi)
  }
  # First generate the offset vector from the first layer to the second layer
  b = runif(d[2], 0, 0.2)
  
  # Generate all offset vectors
  for (i in 2:(layer_n-1)){
    bi = runif(d[i+1], 0, 0.2)
    b = list(b, bi)
  }
  network = list(h=h, W=W, b=b)
  return(network)
}

forward <- function(nn, inp){
  # Take h, W, b from nn
  h = nn$h
  W = nn$W
  b = nn$b
  
  # Determine the number of neural network layers
  layer_n = length(W)
  
  # Put the data into the first layer (input layer)
  h[[1]] = inp
  
  # Update data at each level
  for (i in 1:layer_n){
    h_next =as.matrix(W[[i]]) %*% as.matrix(h[[i]]) + as.matrix(b[[i]])
    h_next[h_next < 0 ] = 0
    h[[i+1]] = h_next
  }
  return(h = h)
}

backward <- function(nn, k){
  # Take out the h updated by the forward function
  h = nn$h
  
  # Determine the number of neural network layers
  layer_n = length(h)
  #p_k = exp(h[layer_n][k])/sum(exp(h[layer_n]))
  #L_i = -log(p_k)
  
  # Define a function to find the derivative of loss
  der_lossh <- function(h_l, k){
    d_l = rep(0, length(hl))
    for (j in 1:length(hl)){
      if (j %in% k){
        d_l[j] = exp(hl[j])/sum(exp(hl))
      }else{
        d_l[j] = exp(hl[j])/sum(exp(hl)) - 1
      }
    }
    return(d_l = d_l)
  }
  
  # Find the derivative of the last layer (output layer)
  d_L = der_lossh(h[[layer_n]], k) 
  
  # Find the derivatives of each layer with respect to h, W, b
  for (l in 1:(layer_n-1)){
    # Find d^(l+1)
    d_l1 = der_lossh(h[[l+1]])
    for (j in 1:length(h[[l+1]])){
      if (h[[l+1]][j] <= 0){
        d_l1[j] = 0 
      }
    }
    
    # Since the list is not given in advance, when l=1, the value is assigned first
    if (l == 1){
      dh = t(W[[l]]) %*% d_l1
      db = d_l1
      dW = d_l1 %*% t(h[[l]])
    
      # Put the derivatives of the remaining layers into the list
    }else{
      dh = list(dh, t(W[[l]]) %*% d_l1)
      db = list(db, d_l1)
      dw = list(db, d_l1 %*% t(h[[l]]))
    }
  }
  
  # Put the derivative of the last layer into dh
  dh = list(dh, d_L)
  d_hWb = list(dh = dh, dW = dW, db = db)
  return(d_hWb)
}

train <- function(nn, inp, k, eta = 0.01, mb = 10, nstep = 10000){
  # Randomly remove mb rows of data
  inp_mb = inp[sample(nrow(inp), mb),]
  
  # Loop nstep times to find the appropriate W, b
  for (nst in 1:nstep){
      for (i in 1:mb){
        # Since the list is not given in advance, when l=1, the value is assigned first
        if (i == 1){
          fw = forward(nn, inp_mb[i,])
          bw = backward(fw, k)
          
          # Put the variables obtained from the remaining data into the list
        }else{
          fw = append(fw, forward(nn, inp_mb[i,]))
          bw = append(bw, backward(fw, k))
        }
      }
      
      # Create a list for storing dw, db and
      dW_sum = db_sum = bw[[1]]$dW
      
      # Create a list for storing dw, db
      dW = db = bw[[1]]$dw
      
      # Sum the weights and offset vectors of each layer
      for ( j in 1:mb){
        for (m in 1:length(bw[[j]]$dw)){
            dW_sum[[m]] = dW_sum[[m]] + bw[[j]]$dW[[m]]
            db_sum[[m]] = db_sum[[m]] + bw[[j]]$db[[m]]
        }
      }
      
      # Find the average of the weights and offset vectors of each layer of 10 rows of data
      for (i in 1:len(dw)){
        dW[[i]] = dW_sum[[i]]/mb
        db[[i]] = db_sum[[i]]/mb
      }
      
      # Update weights and offset vectors
      for (i in 1:len(dw)){
        nn$W[[i]] = nn$W[[i]] - eta*dW[[i]]
        nn$b[[i]] = nn$b[[i]] - eta*db[[i]]
      }
  }
}
set.seed(73)
d <- c(4, 8, 7, 3)
nn <- netup(d)

test_ind <- 1:(nrow(iris)/5)*5
species <- factor(as.array(iris$Species))

train_feature <- as.matrix(iris[-test_ind, 1:4])
train_species <- as.numeric(species)[-test_ind]
test_feature <- as.matrix(iris[test_ind, 1:4])
test_species <- as.numeric(species)[test_ind]

trained_model <- train(nn, train_feature, train_species)
result <- rep(0, length(test_ind))

for(i in 1:length(test_ind)) {
  n_temp <- forward(trained_model, test_feature[i, ])
  result[i] <- which.max(n_temp$h[[length(d)]])
}

error_rate <- sum(result != test_species)/length(test_species)
print(result)
print(test_species)
print(error_rate)
