# -*- coding: utf-8 -*-
from dataclasses import dataclass


class OpenDrive:
    """ An object storing the parsed OpenDrive file """

    def __init__(self):
        self.header = None
        self._roads = []
        self._controllers = []
        self._junctions = []
        self._junction_groups = []
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
        return self._junction_groups

    @property
    def stations(self):
        """ """
        return self._stations


@dataclass
class Header:
    """ Dataclass holding header information of the OpenDrive file """
    rev_major: str = None
    rev_minor: str = None
    name: str = None
    version: str = None
    date: str = None
    north: str = None
    south: str = None
    east: str = None
    west: str = None
    vendor: str = None
    geo_reference: str = None
    program: str = None
