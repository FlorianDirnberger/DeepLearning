This manual ensures not burning down the HPC and making ourselves very unpopular at DTU ;) 

Manual: How to set up parametrization with wandb sweep:

1. Set your personal wandb login key in an .env file (have a look in the gitignore)

2. In the model add the variable you want to paramterize in the init constructor 

3. add it to the sweep config as the examples show 


Sweep-variables: 

First of all we should do research about suitable parameters and decide on a strategy on how to minimze runs (even on hpc one run is ~2 min)

List the parameters here following the template 

parameter: [Interval_min, Interval_max], [Chosen_val1, chosen_val2, ... ] # Argumentation + source

#tbd here# 


## Implemented ##
learning_rate: [1e-3, 1e-6], [1e-3, 1e-4, 1e-5, 1e-6] # We could consider 1e-3, since I used it in one of the DL exercises.

activation_fn: ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Swish', 'Mish'] # There are many more activation functions, maybe we can add two more popular ones, but there are many more to investigate like GELU or Softplus.

weight_init: [uniform_xavier, normal_xavier, uniform_he, normal_he] # Use better fitting weight inits for specific activation function (Xavier: Sigmoid, Tanh; He: ReLU, LeakyReLU, Swish, Mish), not just a general one (weights_init_uniform_rule). Maybe also have a try with a random init.

kernel_size: [1x3, 3x1, 3x5, 5x3, 3x7, 7x3] # Choosing non-squared kernels may achieve different results, at least worth trying.


## Further ideas ##
lr_scheduler: [None, Step, Exponential] # Gives additional hyperparameter to momentum parameter or adam optimizer, but also provides a more handcrafted control over the learning rate.
weight_init [random]


optimizer:  SGD + weight decay + momentum                       # can handel large data sets effectively,
                                                                # regularization (weight_decay): prevents overfitting (1e-5, 1e-4, 1e-3),
                                                                # momentum accelerates SGD to find minimum (between 0.5 and 0.99),
            
            Adam                                                # works for a wide range of models
                                                                # weight decay is incorporated into the gradient update step
                                                                # - can lead to wrong regularization, better AdamW
            
            AdamW (incorporats weight_decay)                    # more effective and stable when regularization should be applied
                                                                # decouples weight decay from the optimization steps 
                                                                # and directly applies the regularization to the weights
                                                                # weight_decay in [1e-5,1e-1]
            
            AdaGrad                                             # learning rate is adjusted automatically 
                                                                # - smaller for frequently occurring features, larger for infrequent features
                                                                # good for data with sparse features 
                                                                # maybe good for our case because we just want to finde one point (initial velocity),
                                                                # but I'm not sure if that is what meant by sparse data


weight_decay:   [1e-5, 1e-1], [1e-5, 1e-4, 1e-3, 1e-2]          # prevents overfitting (can be adapted with SGD and AdamW)
                                                                # Didn't choose: 1e-1 (very strong regularization probably not necessary)

momentum:       (0,1),         [0.5, 0.6, 0.7, 0.8]             # smaller values leads to noisy updates, that's why values are choosen like this
