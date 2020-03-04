# PID_autotuning_relaybase_Adam
A PID autotuning based on relay model tuning and gradient descent with Adam
I just modified the code of [hirschmann](https://github.com/hirschmann/pid-autotune) and add a gradient descend algorithm(Adam) in it
And thanks Swan BOSCS for his help

## Relay model
[PID autotuning algorithm](http://brettbeauregard.com/blog/2012/01/arduino-pid-autotune-library/)
We can find a PID parameter closed to our target but it need to be fine tuned

## Adam
[Adam algorithm](https://ruder.io/optimizing-gradient-descent/)
We now have a pre-tuned pid from relay model
we need to define a cost function and use gradient descend to find a optimal PID
But gradient descent is easy to be trapped in local minimum, so Adam algorithm will be helpful


## Simulation autotuning

`sim_adam.py --atune`

![2020-03-04 18-45-19屏幕截图.png](https://i.loli.net/2020/03/05/PbHkSrEJB416gox.png)



## Todo
- clean the code and make some comments
- implement with C
- try with different cost function
- try with reinforcement learning
