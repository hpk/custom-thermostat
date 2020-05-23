from typing import List

from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
import homeassistant.helpers.config_validation as cv

import voluptuous as vol


METHOD_ALWAYS = "always"
METHOD_MOTION = "motion"
METHOD_WEIGHTED_MOTION = "weighted_motion"
METHODS = [METHOD_ALWAYS, METHOD_MOTION, METHOD_WEIGHTED_MOTION]

CONF_NAME = "name"
CONF_HEAT_TARGET = "heat_target"
CONF_COOL_TARGET = "cool_target"
CONF_SENSORS = "sensors"
CONF_TEMPERATURE_ENTITY = "temperature"
CONF_MOTION_ENTITY = "motion"
CONF_METHOD = "method"
CONF_MINUTES = "interval_minutes"

DEFAULT_INTERVAL_MINUTES = 60

SENSOR_SCHEMA = vol.Schema({
    vol.Required(CONF_TEMPERATURE_ENTITY): cv.entity_domain(SENSOR_DOMAIN),
    vol.Optional(CONF_MOTION_ENTITY): cv.entity_domain(BINARY_SENSOR_DOMAIN),
    vol.Optional(CONF_METHOD): vol.In(METHODS),
})

PRESET_SCHEMA = vol.All(
    cv.ensure_list,
    [
        vol.All(
            vol.Schema(
                {
                    vol.Required(CONF_NAME): cv.string,
                    vol.Required(CONF_SENSORS): vol.All(cv.ensure_list, [vol.All(SENSOR_SCHEMA)]),
                    vol.Optional(CONF_HEAT_TARGET): vol.Coerce(float),
                    vol.Optional(CONF_COOL_TARGET): vol.Coerce(float),
                    vol.Optional(CONF_MINUTES, default=DEFAULT_INTERVAL_MINUTES): vol.Coerce(int),
                }
            )
        )
    ]
)

def build_presets(items):
    presets = []
    for raw in items:
        presets.append(
            Preset(
                raw["name"],
                raw.get(CONF_HEAT_TARGET),
                raw.get(CONF_COOL_TARGET),
                raw.get(CONF_MINUTES),
                build_sensors(raw.get(CONF_SENSORS)),
            )
        )
    return presets

def build_sensors(items):
    sensors = []
    for raw in items:
        sensors.append(
            Sensor(
                raw[CONF_TEMPERATURE_ENTITY],
                raw.get(CONF_MOTION_ENTITY),
                raw.get(CONF_METHOD),
            )
        )
    return sensors


class Sensor:
    def __init__(
        self,
        temp_sensor: str,
        motion_sensor: str,
        method: str,
    ):
        self._temp_sensor = temp_sensor
        self._motion_sensor = motion_sensor

        if not method:
            if not motion_sensor:
                method = METHOD_ALWAYS
            else:
                method = METHOD_MOTION
        self._method = method

    @property
    def temp_sensor(self):
        return self._temp_sensor

    @property
    def motion_sensor(self):
        return self._motion_sensor

    @property
    def method(self):
        return self._method

class Preset:
    def __init__(
        self,
        name: str,
        heat_target: float,
        cool_target: float,
        interval_minutes: int,
        sensors: List[Sensor],
    ):
        self._name = name
        self._heat_target = heat_target
        self._cool_target = cool_target
        self._sensors = sensors
        self._minutes = interval_minutes

    @property
    def name(self):
        return self._name
    
    @property
    def heat_target(self):
        return self._heat_target

    @property
    def cool_target(self):
        return self._cool_target

    @property
    def sensors(self):
        return self._sensors

    @property
    def minutes(self):
        return self._minutes
