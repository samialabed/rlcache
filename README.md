# RLCache
Reinforcement Learning Cache Manager



# TODOs:


## Observer
Should there be a global observer, but for each type of strategy have their own implementation?

Have hook_observers method that takes list of observers and send them requests when observation is detected.


## Refactoring TODO
rlcache.observers: shouldn't observer keep track of stats instead of having it floating around.
rlcache.observers.observe: do I need an update?
rlcache.backend.inmemory: What is the point of that?
rlcache.cache_constants: Needs rethinking, what's difference between it and observers? 
__init__ in rlcache.[backend, strategieies]: Should use magic to factory build based on configs.

## Flask low priority
Config retrival in __init__,py under rlcache/server seems hacky.
Replace print with logger
figure out difference between insert and update
Distinguish between /close for load and /close for workload

## Nice to do TODO 
Integrate with redis.