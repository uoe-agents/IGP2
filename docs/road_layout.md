(road_layout)=
# Road layouts

Road layouts, defined using the [ASAM OpenDrive 1.6](https://www.asam.net/standards/detail/opendrive/) standard, form the core of each scenario.
They are the source of rich semantic information, such as lane midlines, junction priorities, and road geometries, which are essential to performing the planning and prediction tasks within IGP2.

Road layouts can be designed using external tools.
The most feature rich is [RoadRunner by MathWorks](https://uk.mathworks.com/products/roadrunner.html?requestedDomain=), however this is not easy to get access to.
A more accessible alternative is [TrueVision Designer by truevision](https://www.truevision.ai/designer).

We do not aim to give a detailed tutorial on how to use these tools, or how to use them in conjunction with CARLA. 
You should consult the documentation of your chosen tool first.
The CARLA documentation also has a nice page on how to get started with [new maps in CARLA](https://carla.readthedocs.io/en/latest/tuto_content_authoring_maps/).
Following the previous link, you can learn to design your own maps with buildings, foliage, and props as well if you so wish.

To use OpenDrive road layouts with IGP2 you should make sure to have done the following steps:
1. Added junction priorities.
2. Added junction groups for each roundabout present in the road layout definition.

The following sections describe in detail how to do these steps.

## Roundabout layouts

Roundabouts are somewhat tricky to define in OpenDrive, and external tools take a rather liberal approach in coming up with their own ways of automatically converting roundabouts to OpenDrive road layouts.

In our implementation, we assume that roundabouts consist of distinct non-overlapping junctions separated by at least one road segment (see [Figure 84](https://www.asam.net/standards/detail/opendrive/) in the OpenDrive standard).
This allows best for our macro actions to work properly.


## Additional necessary annotations

This section describes the essential components necessary for IGP2, which are not currently supported by the tools mentioned above. 
These can be added in by hand after exporting the map from the tool without too much work.

**Junction connecting road priorities**: IGP2 relies on behaviour, rather than signals, to infer the goals of vehicles. However, much of people's behaviour on the road relies on junction priorities, which are not currently supported by external tools. 

Junction priorities should be added according to the actual road priorities using the &lt;priority&gt; tag. More can be read about this tag in the [standard](https://www.asam.net/standards/detail/opendrive/).

**Junction groups**: IGP2 also works in roundabouts. The OpenDrive standard uses junction groups to denote roundabouts, however these are not currently supported by external tools.  If not roundabouts are present in the road layout, then this part can be ignored.

Junction groups should be added for each roundabout in the scenario. Each junction group should contain all the junctions that relate to that particular roundabout. 
More about junction groups is available in the [standard](https://www.asam.net/standards/detail/opendrive/).

## Other remarks

The following are some further remarks about what our implementation of the OpenDrive standard assumes.

1. Speed limits: We assume there is a global speed limit in each scenario. While the OpenDrive standard allows for defined speed limits on roads, we do not currently support this functionality.
2. Signals: Signals are not supported by IGP2.