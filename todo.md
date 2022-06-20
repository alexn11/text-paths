


# OVERVIEW
- create paths with 10-20 steps from one sentence to the other
- at each step, its a double inverse problem:
 1. from the output produce input internal representation by reversing the core model
 2. from the internal representation, inverse the embedding layers to get a vector in the 1-hot space (but not 1-hot itself) (+ position embedding?)
- reduce the vector in 1-hot space to the closest actual r1-hot and use the vocab to translate into words
- at each step can start the optimization process with the previous step input

# TODO
- training for the inverse: the loss doesnt decrease a lot if at all.
- solve the reverse the embedding problem (look what the embedding is made of)
- the above requires 1-hot encoding and processing position embedding
- the path generation, with steps etc

# WHAT IS READY
- the reverse the core problem
- the bit after
```>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BERT
```
  is a self contained demo of how its done,
  it perturbs a sentence and solve the inverse problem for the core of the model


# COULD DO
- make the thing work on inceptionv3 as well :)
  (so far not successfull)

# DONE

- add tensorboard to survey progress