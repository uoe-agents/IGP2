# -*- coding: utf-8 -*-

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.2.0"
__maintainer__ = "Sebastian Maierhofer"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class OpenDrive:
    """ An object storing the parsed OpenDrive file """

    def __init__(self):
        self.header = None
        self._roads = []
        self._controllers = []
        self._junctions = []
        self._junctionGroups = []
        self._stations = []

    @property
    def roads(self):
        """ Get all roads of the OpenDrive file"""
        return self._roads

    def get_road(self, road_id):
        """ Get a Road object by ID

        Args:
          road_id: The ID of the required Road

        Returns:
            A Road object or None
        """
        for road in self._roads:
            if road.id == road_id:
                return road
        return None

    @property
    def controllers(self):
        """ Get all controllers of the OpenDrive file"""
        return self._controllers

    @property
    def junctions(self):
        """ Get all junctions of the OpenDrive file """
        return self._junctions

    def get_junction(self, junction_id):
        """ Get a Junction object by ID

        Args:
          junction_id: The ID of the required Junction

        Returns:
            A Junction object or None
        """
        for junction in self._junctions:
            if junction.id == junction_id:
                return junction
        return None

    @property
    def junction_groups(self):
        """ """
        return self._junctionGroups

    @property
    def stations(self):
        """ """
        return self._stations


class Header:
    """ """

    def __init__(
        self,
        rev_major=None,
        rev_minor=None,
        name: str = None,
        version=None,
        date=None,
        north=None,
        south=None,
        east=None,
        west=None,
        vendor=None,
    ):
        self.revMajor = rev_major
        self.revMinor = rev_minor
        self.name = name
        self.version = version
        self.date = date
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.vendor = vendor
        self.geo_reference = None
