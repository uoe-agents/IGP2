import matplotlib.pyplot as plt

from igp2.opendrive.map import Map


def plot_map(odr_map: Map, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """ Draw the road layout of the map

    Args:
        odr_map: The Map to plot
        ax: Axes to draw on

    Keyword Args:
        midline: True if the midline of roads should be drawn (default: False)
        road_ids: If True, then the IDs of roads will be drawn (default: False)
        markings: If True, then draw LaneMarkers (default: False)
        road_color: Plot color of the road boundary (default: black)
        junction_color: Face color of junctions (default: [0.941, 1.0, 0.420, 0.5])
        midline_color: Color of the midline

    Returns:
        The axes onto which the road layout was drawn
    """
    colors = plt.get_cmap("tab10").colors

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.set_xlim([odr_map.west, odr_map.east])
    ax.set_ylim([odr_map.south, odr_map.north])

    for road_id, road in odr_map.roads.items():
        boundary = road.boundary.boundary
        if boundary.geom_type == "LineString":
            ax.plot(boundary.xy[0],
                    boundary.xy[1],
                    color=kwargs.get("road_color", "k"))
        elif boundary.geom_type == "MultiLineString":
            for b in boundary:
                ax.plot(b.xy[0],
                        b.xy[1],
                        color=kwargs.get("road_color", "orange"))

        color = kwargs.get("midline_color", colors[road_id % len(colors)] if kwargs.get("road_ids", False) else "r")
        if kwargs.get("midline", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    ax.plot(lane.midline.xy[0],
                            lane.midline.xy[1],
                            color=color)

        if kwargs.get("road_ids", False):
            mid_point = len(road.midline.xy) // 2
            ax.text(road.midline.xy[0][mid_point],
                    road.midline.xy[1][mid_point],
                    road.id,
                    color=color, fontsize=15)

        if kwargs.get("markings", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    for marker in lane.markers:
                        line_styles = marker.type_to_linestyle
                        for i, style in enumerate(line_styles):
                            df = 0.13  # Distance between parallel lines
                            side = "left" if lane.id <= 0 else "right"
                            line = lane.reference_line.parallel_offset(i * df, side=side)
                            ax.plot(line.xy[0], line.xy[1],
                                    color=marker.color_to_rgb,
                                    linestyle=style,
                                    linewidth=marker.plot_width)

    for junction_id, junction in odr_map.junctions.items():
        if junction.boundary.geom_type == "Polygon":
            ax.fill(junction.boundary.boundary.xy[0],
                    junction.boundary.boundary.xy[1],
                    color=kwargs.get("junction_color", (0.941, 1.0, 0.420, 0.5)))
        else:
            for polygon in junction.boundary:
                ax.fill(polygon.boundary.xy[0],
                        polygon.boundary.xy[1],
                        color=kwargs.get("junction_color", (0.941, 1.0, 0.420, 0.5)))
    return ax
