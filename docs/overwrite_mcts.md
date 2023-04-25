(overwrite_mcts)=
# Overwriting MCTS

Monte Carlo Tree Search (MCTS) is the choice of planning algorithm in IGP2.

MCTS is a tree-based iterative algorithm that performs forward simulations of the world and accumulates experience from these simulations to select the best possible sequence of macro actions for a vehicle.

As a tree-based method, MCTS is described by a `Tree` which is composed of `Nodes`. 
MCTS also plans over a set of `MCTSAction` derived directly from macro actions.

Sometimes, you may wish to customise how rollouts are performed, how nodes are selected, or how actions are stored. 
Fortunately, MCTS supports overwriting many of its built-in functionalities.

In the following, we give a brief description of the various (overrideable) components of MCTS and what their function is within the entirety of MCTS.
This is not an exhaustive description of MCTS, however, if you are interested in learning more than you can look at the code in the module `igp2.planning`.

**Note: Overwriting MCTS is not necessary for new macro actions and maneuvers to work properly. This is merely an extra option to add new features to IGP2.**

## Reward function

MCTS uses a reward function to calculate the reward from having executed a given trace of macro actions.

Rewards are calculated using the `igp2.planning.reward.Reward` class.
You can create your own instance of this class (by using a configuration file most simply), or create a new subclass to overwrite the built-in `Reward` class.
The new class then can be given to MCTS by passing in the `reward` argument during call to `__init__`, or you can set the MCTS.reward field directly.

## Tree 

The `igp2.planning.tree.Tree` class is directly responsible for managing nodes of the rollout, sampling from macro actions at the start of a rollout, and backpropagating reward information across the search trace.
Trees also store some information regarding the MCTS simulation. 
They store which actions were sampled for non-ego agents and the distributions that those samplings followed.

The `Tree` class also determines what policy to follow when selecting an action (`action_policy`) and what policy to follow when selecting the final plan at the end of the MCTS rollouts (`plan_policy`).

It also exposes the `Tree.on_finish()` method which can be used to perform post-MCTS tasks, such as writing to files, or running calculations using rollouts.

All of the above functionality can be overriden by creating a new `Tree` subclass and specifying the relevant parameters.

## Node
 The `igp2.planning.node.Node` class represents a particular state of the ego vehicle at which point the last macro action in its current trace has terminated.
 From this node, a new macro action is selected based on the Q-values stored in the node. 
 
Nodes are characterised by a `key`.
Keys have to be unique and hashable for each node.
The method `MCTS.to_key()` can be overwritten if necessary to produce a new key representation, however, the preferred way to do this is to overwrite the `MCTSAction` class (see section below).

The state of the rollout up to this point is stored in the node in the `run_results` property, which can be access later on (after MCTS terminated) to retrieve all information about rollouts.

If the node represents a sequence of macro actions that ends the rollout, then the property `reward_results` will contain the reward information about from that rollout.

## MCTSAction

MCTS does not directly plan over macro actions, rather, it puts them in a wrapper called `igp2.planning.mctsaction.MCTSAction`. 

The most important method of this class is the `__repr__` class which is used to set up the keys for nodes.
Overwriting this method will change what keys are created by the `MCTS.to_key()` method.