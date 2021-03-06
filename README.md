This is the experiment of distributed lasso CTA algorithm on high dimensional space.

# Quick Start
```shell
python main.py
```

# Configuration instruction
The parameters are all controlled in *config.json* file. This json file contains following section:
* multiprocessing
* experiment
* data
* network
* solver

## multiprocessing

## experiment
This section is a high level capsulation of experiment. 

### exp
There are several variables we can experiment on, but each experiment only contains one experiment variable. The valid experiment parameters contain:
* N
* p
* s
* k
* sigma
* m
* method
* step_size
* constraint_param(lmda or r)
### plot
this section decide what kind of line figure you want to plot
x_axis and y_axis set the variable you want to plot.\
x_axis usually put the experiment variable or iteration;\
y_axis usually put the loss. There are two kinds of loss you may want to plot: optimization_log_loss or statistic_log_loss.\
stack shows which variable you will use to put them in the same line figure. For instance, you may want to put model or experiment parameter.\
## data
## network
## solver

