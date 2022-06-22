# -*- coding: utf-8 -*-

import numpy as np
import logging
from lxml import etree
from igp2.opendrive.elements.opendrive import OpenDrive, Header
from igp2.opendrive.elements.road import Road
from igp2.opendrive.elements.road_link import (
    Predecessor as RoadLinkPredecessor,
    Successor as RoadLinkSuccessor,
    Neighbor as RoadLinkNeighbor,
)
from igp2.opendrive.elements.road_type import (
    RoadType,
    Speed as RoadTypeSpeed,
)
from igp2.opendrive.elements.road_elevation_profile import (
    ElevationRecord as RoadElevationProfile,
)
from igp2.opendrive.elements.road_lateral_profile import (
    Superelevation as RoadLateralProfileSuperelevation,
    Crossfall as RoadLateralProfileCrossfall,
    Shape as RoadLateralProfileShape,
)
from igp2.opendrive.elements.road_lanes import (
    LaneOffset as RoadLanesLaneOffset,
    Lane as RoadLaneSectionLane,
    LaneSection as RoadLanesSection,
    LaneWidth as RoadLaneSectionLaneWidth,
    LaneBorder as RoadLaneSectionLaneBorder,
    LaneMarker as RoadLaneSectionMarker
)
from igp2.opendrive.elements.junction import (
    Junction,
    Connection as JunctionConnection,
    JunctionLaneLink as JunctionConnectionLaneLink, JunctionPriority, JunctionGroup,
)

logger = logging.getLogger(__name__)


def parse_opendrive(root_node) -> OpenDrive:
    """Tries to parse XML tree, returns OpenDRIVE object

    Args:
      root_node: The root node of a parsed OpenDrive XML tree

    Returns:
      The object representing an OpenDrive specification.

    """

    # Only accept lxml element
    if not etree.iselement(root_node):
        raise TypeError("Argument root_node is not a xml element")

    opendrive = OpenDrive()

    # Header
    header = root_node.find("header")
    if header is not None:
        parse_opendrive_header(opendrive, header)

    # Load roads
    for road in root_node.findall("road"):
        parse_opendrive_road(opendrive, road)

    # Load Junctions
    for junction in root_node.findall("junction"):
        parse_opendrive_junction(opendrive, junction)

    # Load JunctionGroups
    for junction_group in root_node.findall("junctionGroup"):
        parse_opendrive_junction_group(opendrive, junction_group)

    # Load Road Links
    for road in root_node.findall("road"):
        parse_opendrive_road_link(opendrive, road)

    # Load Junction Lane Links
    for junction in opendrive.junctions:
        load_junction_lane_links(junction)

    # Load additional objects
    for road in opendrive.roads:
        # Load Junction references
        if road.junction is not None:
            road.junction = opendrive.get_junction(road.junction)

        # Load Lane Links for current Road
        load_road_lane_links(road)

    return opendrive


def parse_opendrive_road_link(opendrive, road):
    opendrive_road_link = road.find("link")
    road = opendrive.get_road(int(road.get("id")))
    if opendrive_road_link is None:
        return

    predecessor = opendrive_road_link.find("predecessor")

    if predecessor is not None:
        road.link.predecessor = RoadLinkPredecessor(
            predecessor.get("elementType"),
            predecessor.get("elementId"),
            predecessor.get("contactPoint"),
        )

        if road.link.predecessor.element_type == "road":
            road.link.predecessor.element = opendrive.get_road(int(predecessor.get("elementId")))
        elif road.link.predecessor.element_type == "junction":
            road.link.predecessor.element = opendrive.get_junction(int(predecessor.get("elementId")))

    successor = opendrive_road_link.find("successor")

    if successor is not None:
        road.link.successor = RoadLinkSuccessor(
            successor.get("elementType"),
            successor.get("elementId"),
            successor.get("contactPoint"),
        )
        if road.link.successor.element_type == "road":
            road.link.successor.element = opendrive.get_road(int(successor.get("elementId")))
        elif road.link.successor.element_type == "junction":
            road.link.successor.element = opendrive.get_junction(int(successor.get("elementId")))

    for neighbor in opendrive_road_link.findall("neighbor"):
        new_neighbor = RoadLinkNeighbor(
            neighbor.get("side"), neighbor.get("elementId"), neighbor.get("direction")
        )
        new_neighbor.element = opendrive.get_road(int(neighbor.get("elementId")))

        road.link.neighbors.append(new_neighbor)


def parse_opendrive_road_type(road, opendrive_xml_road_type: etree.ElementTree):
    """Parse opendrive road type and append to road object.

    Args:
      road: Road to append the parsed road_type to types.
      opendrive_xml_road_type: XML element which contains the information.
      opendrive_xml_road_type: etree.ElementTree:

    """
    speed = None
    if opendrive_xml_road_type.find("speed") is not None:
        speed = RoadTypeSpeed(
            max_speed=opendrive_xml_road_type.find("speed").get("max"),
            unit=opendrive_xml_road_type.find("speed").get("unit"),
        )

    road_type = RoadType(
        s_pos=opendrive_xml_road_type.get("s"),
        use_type=opendrive_xml_road_type.get("type"),
        speed=speed,
    )
    road.types.append(road_type)


def parse_opendrive_road_geometry(new_road, road_geometry):
    start_coord = [float(road_geometry.get("x")), float(road_geometry.get("y"))]

    if road_geometry.find("line") is not None:
        new_road.plan_view.add_line(
            start_coord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
        )

    elif road_geometry.find("spiral") is not None:
        new_road.plan_view.add_spiral(
            start_coord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("spiral").get("curvStart")),
            float(road_geometry.find("spiral").get("curvEnd")),
        )

    elif road_geometry.find("arc") is not None:
        new_road.plan_view.add_arc(
            start_coord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("arc").get("curvature")),
        )

    elif road_geometry.find("poly3") is not None:
        new_road.plan_view.add_poly3(
            start_coord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("poly3").get("a")),
            float(road_geometry.find("poly3").get("b")),
            float(road_geometry.find("poly3").get("c")),
            float(road_geometry.find("poly3").get("d")),
        )
        # raise NotImplementedError()

    elif road_geometry.find("paramPoly3") is not None:
        if road_geometry.find("paramPoly3").get("pRange"):

            if road_geometry.find("paramPoly3").get("pRange") == "arcLength":
                p_max = float(road_geometry.get("length"))
            else:
                p_max = None
        else:
            p_max = None

        new_road.plan_view.add_param_poly3(
            start_coord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("paramPoly3").get("aU")),
            float(road_geometry.find("paramPoly3").get("bU")),
            float(road_geometry.find("paramPoly3").get("cU")),
            float(road_geometry.find("paramPoly3").get("dU")),
            float(road_geometry.find("paramPoly3").get("aV")),
            float(road_geometry.find("paramPoly3").get("bV")),
            float(road_geometry.find("paramPoly3").get("cV")),
            float(road_geometry.find("paramPoly3").get("dV")),
            p_max,
        )

    else:
        raise Exception("invalid xml")


def parse_opendrive_road_elevation_profile(new_road, road_elevation_profile):
    for elevation in road_elevation_profile.findall("elevation"):
        new_elevation = (
            RoadElevationProfile(
                float(elevation.get("a")),
                float(elevation.get("b")),
                float(elevation.get("c")),
                float(elevation.get("d")),
                start_pos=float(elevation.get("s")),
            ),
        )

        new_road.elevation_profile.elevations.append(new_elevation)


def parse_opendrive_road_lateral_profile(new_road, road_lateral_profile):
    for superelevation in road_lateral_profile.findall("superelevation"):
        new_superelevation = RoadLateralProfileSuperelevation(
            float(superelevation.get("a")),
            float(superelevation.get("b")),
            float(superelevation.get("c")),
            float(superelevation.get("d")),
            start_pos=float(superelevation.get("s")),
        )

        new_road.lateral_profile.superelevations.append(new_superelevation)

    for crossfall in road_lateral_profile.findall("crossfall"):
        new_crossfall = RoadLateralProfileCrossfall(
            float(crossfall.get("a")),
            float(crossfall.get("b")),
            float(crossfall.get("c")),
            float(crossfall.get("d")),
            side=crossfall.get("side"),
            start_pos=float(crossfall.get("s")),
        )

        new_road.lateral_profile.crossfalls.append(new_crossfall)

    for shape in road_lateral_profile.findall("shape"):
        new_shape = RoadLateralProfileShape(
            float(shape.get("a")),
            float(shape.get("b")),
            float(shape.get("c")),
            float(shape.get("d")),
            start_pos=float(shape.get("s")),
            start_pos_t=float(shape.get("t")),
        )

        new_road.lateral_profile.shapes.append(new_shape)


def parse_opendrive_road_lane_offset(new_road, lane_offset):
    new_lane_offset = RoadLanesLaneOffset(
        float(lane_offset.get("a")),
        float(lane_offset.get("b")),
        float(lane_offset.get("c")),
        float(lane_offset.get("d")),
        start_pos=float(lane_offset.get("s")),
    )

    new_road.lanes.lane_offsets.append(new_lane_offset)


def parse_opendrive_road_lane_section(new_road, lane_section_id, lane_section, program=None):
    new_lane_section = RoadLanesSection(road=new_road)

    # Manually enumerate lane sections for referencing purposes
    new_lane_section.idx = lane_section_id

    new_lane_section._start_ds = float(lane_section.get("s"))
    new_lane_section.single_side = lane_section.get("singleSide")

    sides = dict(
        left=new_lane_section.left_lanes,
        center=new_lane_section.center_lanes,
        right=new_lane_section.right_lanes,
    )

    drivable = False

    for sideTag, newSideLanes in sides.items():

        side = lane_section.find(sideTag)

        # It is possible one side is not present
        if side is None:
            continue

        for lane in side.findall("lane"):

            new_lane = RoadLaneSectionLane(
                parent_road=new_road, lane_section=new_lane_section
            )
            new_lane.id = lane.get("id")
            new_lane.type = lane.get("type")
            drivable = drivable or (new_lane.type == "driving")

            # In some sample files the level is not specified according to the OpenDRIVE spec
            new_lane.level = (
                "true" if lane.get("level") in [1, "1", "true"] else "false"
            )

            # Lane Links
            if lane.find("link") is not None:

                if lane.find("link").find("predecessor") is not None:
                    new_lane.link.predecessor_id = (
                        lane.find("link").find("predecessor").get("id")
                    )

                if lane.find("link").find("successor") is not None:
                    new_lane.link.successor_id = (
                        lane.find("link").find("successor").get("id")
                    )

                # RoadRunner does not take lane direction into account when outputting lane links so need to flip
                #  lane IDs for left-lanes
                if program == "RoadRunner" and new_lane.id > 0:
                    new_lane.link.predecessor_id, new_lane.link.successor_id = \
                        new_lane.link.successor_id, new_lane.link.predecessor_id

            # Width
            for widthIdx, width in enumerate(lane.findall("width")):
                new_width = RoadLaneSectionLaneWidth(
                    float(width.get("a")),
                    float(width.get("b")),
                    float(width.get("c")),
                    float(width.get("d")),
                    idx=widthIdx,
                    start_offset=float(width.get("sOffset")),
                )

                new_lane.widths.append(new_width)

            # Border
            for borderIdx, border in enumerate(lane.findall("border")):
                new_border = RoadLaneSectionLaneBorder(
                    float(border.get("a")),
                    float(border.get("b")),
                    float(border.get("c")),
                    float(border.get("d")),
                    idx=borderIdx,
                    start_offset=float(border.get("sOffset")),
                )

                new_lane.borders.append(new_border)

            if lane.find("width") is None and lane.find("border") is not None:
                new_lane.widths = new_lane.borders
                new_lane.has_border_record = True

            # Road Marks
            for markerIdx, marker in enumerate(lane.findall("roadMark")):
                new_marker = RoadLaneSectionMarker(
                    width=float(marker.get("width", 0.0)),
                    color=marker.get("color"),
                    weight=marker.get("weight"),
                    type=marker.get("type"),
                    idx=markerIdx,
                    start_offset=float(marker.get("sOffset"))
                )
                new_lane.markers.append(new_marker)

            # Material
            # TODO implementation

            # Visiblility
            # TODO implementation

            # Speed
            # TODO implementation

            # Access
            # TODO implementation

            # Lane Height
            # TODO implementation

            # Rules
            # TODO implementation

            newSideLanes.append(new_lane)

    new_lane_section._drivable = drivable
    new_road.lanes.lane_sections.append(new_lane_section)


def parse_opendrive_road(opendrive, road):
    new_road = Road()

    new_road.id = int(road.get("id"))
    new_road.name = road.get("name")

    junction_id = int(road.get("junction")) if road.get("junction") != "-1" else None
    new_road.junction = junction_id

    # TODO verify road length
    new_road._length = float(road.get("length"))

    # Type
    for opendrive_xml_road_type in road.findall("type"):
        parse_opendrive_road_type(new_road, opendrive_xml_road_type)

    # Plan view
    for road_geometry in road.find("planView").findall("geometry"):
        parse_opendrive_road_geometry(new_road, road_geometry)

    # Elevation profile
    road_elevation_profile = road.find("elevationProfile")
    if road_elevation_profile is not None:
        parse_opendrive_road_elevation_profile(new_road, road_elevation_profile)

    # Lateral profile
    road_lateral_profile = road.find("lateralProfile")
    if road_lateral_profile is not None:
        parse_opendrive_road_lateral_profile(new_road, road_lateral_profile)

    # Lanes
    lanes = road.find("lanes")

    if lanes is None:
        raise Exception("Road must have lanes element")

    # Lane offset
    for lane_offset in lanes.findall("laneOffset"):
        parse_opendrive_road_lane_offset(new_road, lane_offset)

    # Lane sections
    for lane_section_id, lane_section in enumerate(road.find("lanes").findall("laneSection")):
        parse_opendrive_road_lane_section(new_road, lane_section_id, lane_section, opendrive.header.program)

    # Objects
    # TODO implementation

    # Signals
    # TODO implementation
    calculate_lane_section_lengths(new_road)

    opendrive.roads.append(new_road)


def parse_opendrive_header(opendrive, header):
    parsed_header = Header(
        header.get("revMajor"),
        header.get("revMinor"),
        header.get("name"),
        header.get("version"),
        header.get("date"),
        header.get("north"),
        header.get("south"),
        header.get("east"),
        header.get("west"),
        header.get("vendor"),
    )
    # Reference
    if header.find("geoReference") is not None:
        parsed_header.geo_reference = header.find("geoReference").text

    # Find whether file was created with RoadRunner
    user_data = header.find("userData")
    if user_data is not None:
        vector_scene = user_data.find("vectorScene")
        if vector_scene is not None:
            parsed_header.program = vector_scene.get("program")

    opendrive.header = parsed_header


def parse_opendrive_junction(opendrive, junction):
    new_junction = Junction()

    new_junction.id = int(junction.get("id"))
    new_junction.name = str(junction.get("name"))

    for connection in junction.findall("connection"):

        new_connection = JunctionConnection()

        new_connection.id = connection.get("id")
        incoming_road_id = int(connection.get("incomingRoad"))
        new_connection.incoming_road = opendrive.get_road(incoming_road_id)

        connecting_road_id = int(connection.get("connectingRoad"))
        new_connection.connecting_road = opendrive.get_road(connecting_road_id)

        new_connection.contact_point = connection.get("contactPoint")

        for laneLink in connection.findall("laneLink"):
            new_lane_link = JunctionConnectionLaneLink()

            new_lane_link.from_id = laneLink.get("from")
            new_lane_link.to_id = laneLink.get("to")

            new_connection.add_lane_link(new_lane_link)

        new_junction.add_connection(new_connection)

    for priority in junction.findall("priority"):
        low_id = int(priority.get("low"))
        high_id = int(priority.get("high"))
        new_priority = JunctionPriority(high_id, low_id)

        new_priority.low = opendrive.get_road(low_id)
        new_priority.high = opendrive.get_road(high_id)

        new_junction.add_priority(new_priority)

    opendrive.junctions.append(new_junction)


def parse_opendrive_junction_group(opendrive, junction_group):
    new_junction_group = JunctionGroup(
        junction_group.get("name"),
        int(junction_group.get("id")),
        junction_group.get("type")
    )

    for junction_reference in junction_group.findall("junctionReference"):
        junction_id = int(junction_reference.get("junction"))
        junction = opendrive.get_junction(junction_id)

        if junction is None:
            raise ValueError(f"Junction with ID {junction_id}")

        new_junction_group.add_junction(junction)
        junction.junction_group = new_junction_group

    opendrive.junction_groups.append(new_junction_group)


def calculate_lane_section_lengths(new_road):
    # OpenDRIVE does not provide lane section lengths by itself, calculate them by ourselves
    for lane_section in new_road.lanes.lane_sections:

        # Last lane section in road
        if lane_section.idx + 1 >= len(new_road.lanes.lane_sections):
            lane_section.length = new_road.plan_view.length - lane_section.start_distance

        # All but the last lane section end at the succeeding one
        else:
            lane_section.length = (
                    new_road.lanes.lane_sections[lane_section.idx + 1].start_distance
                    - lane_section.start_distance
            )

    # OpenDRIVE does not provide lane width lengths by itself, calculate them by ourselves
    for lane_section in new_road.lanes.lane_sections:
        for lane in lane_section.all_lanes:
            widths_poses = np.array(
                [x.start_offset for x in lane.widths] + [lane_section.length]
            )
            widths_lengths = widths_poses[1:] - widths_poses[:-1]

            for widthIdx, width in enumerate(lane.widths):
                width.length = widths_lengths[widthIdx]


def load_junction_lane_links(junction):
    for connection in junction.connections:
        for link in connection.lane_links:
            if connection.contact_point == "start":
                link.to_lane = connection.connecting_road.lanes.lane_sections[0].get_lane(link.to_id)
            elif connection.contact_point == "end":
                link.to_lane = connection.connecting_road.lanes.lane_sections[-1].get_lane(link.to_id)

            if link.to_lane is None:
                logger.debug(f"Connecting Lane in {junction} for {link} not found.")


def load_road_lane_links(road):
    previous_element = road.link.predecessor.element if road.link.predecessor else None
    previous_contact_point = road.link.predecessor.contact_point if road.link.predecessor else None
    next_element = road.link.successor.element if road.link.successor else None
    next_contact_point = road.link.successor.contact_point if road.link.successor else None
    num_sections = len(road.lanes.lane_sections)

    for lane_section_idx, lane_section in enumerate(road.lanes.lane_sections):
        for lane_idx, lane in enumerate(lane_section.right_lanes):
            if lane.link.predecessor_id is not None:
                previous_lane_section = None
                if lane_section_idx == 0 and previous_element is not None:
                    if previous_contact_point == "start":
                        previous_lane_section = previous_element.lanes.lane_sections[0]
                    elif previous_contact_point == "end":
                        previous_lane_section = previous_element.lanes.lane_sections[-1]
                elif lane_section_idx > 0:
                    previous_lane_section = road.lanes.lane_sections[lane_section_idx - 1]

                if previous_lane_section is not None:
                    lane.link.predecessor = [previous_lane_section.get_lane(lane.link.predecessor_id)]
                if lane.link.predecessor is None:
                    logger.debug(f"Road {road.id} - Lane {lane.id}: Predecessor {lane.link.predecessor_id} not found")

            if lane.link.successor_id is not None:
                next_lane_section = None
                if lane_section_idx == num_sections - 1 and next_element is not None:
                    if next_contact_point == "start":
                        next_lane_section = next_element.lanes.lane_sections[0]
                    elif next_contact_point == "end":
                        next_lane_section = next_element.lanes.lane_sections[-1]
                elif lane_section_idx < num_sections - 1:
                    next_lane_section = road.lanes.lane_sections[lane_section_idx + 1]

                if next_lane_section is not None:
                    lane.link.successor = [next_lane_section.get_lane(lane.link.successor_id)]
                if lane.link.successor is None:
                    logger.debug(f"Road {road.id} - Lane {lane.id}: Successor {lane.link.successor_id} not found")
            elif next_element is not None and isinstance(next_element, Junction):
                lane.link.successor = next_element.get_all_connecting_lanes(lane)

        previous_element, next_element = next_element, previous_element
        previous_contact_point, next_contact_point = next_contact_point, previous_contact_point

        for lane_idx, lane in enumerate(lane_section.left_lanes):
            if lane.link.predecessor_id is not None:
                previous_lane_section = None
                if lane_section_idx == num_sections - 1 and previous_element is not None:
                    if previous_contact_point == "start":
                        previous_lane_section = previous_element.lanes.lane_sections[0]
                    elif previous_contact_point == "end":
                        previous_lane_section = previous_element.lanes.lane_sections[-1]
                elif lane_section_idx < num_sections - 1:
                    previous_lane_section = road.lanes.lane_sections[lane_section_idx + 1]

                if previous_lane_section is not None:
                    lane.link.predecessor = [previous_lane_section.get_lane(lane.link.predecessor_id)]
                if lane.link.predecessor is None:
                    logger.debug(f"Road {road.id} - Lane {lane.id}: Predecessor {lane.link.predecessor_id} not found")

            # Find successor Lanes
            if lane.link.successor_id is not None:
                next_lane_section = None
                if lane_section_idx == 0 and next_element is not None:
                    if next_contact_point == "start":
                        next_lane_section = next_element.lanes.lane_sections[0]
                    elif next_contact_point == "end":
                        next_lane_section = next_element.lanes.lane_sections[-1]
                elif lane_section_idx > 0:
                    next_lane_section = road.lanes.lane_sections[lane_section_idx - 1]

                if next_lane_section is not None:
                    lane.link.successor = [next_lane_section.get_lane(lane.link.successor_id)]
                if lane.link.successor is None:
                    logger.debug(f"Road {road.id} - Lane {lane.id}: Successor {lane.link.successor_id} not found")
            elif next_element is not None and isinstance(next_element, Junction):
                lane.link.successor = next_element.get_all_connecting_lanes(lane)

        previous_element, next_element = next_element, previous_element
        previous_contact_point, next_contact_point = next_contact_point, previous_contact_point
