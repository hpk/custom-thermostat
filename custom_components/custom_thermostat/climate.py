import asyncio
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Set, Tuple

import voluptuous as vol

import homeassistant.helpers.config_validation as cv
from homeassistant.components import climate
from homeassistant.components.climate import (
    PLATFORM_SCHEMA,
    ClimateEntity,
)
from homeassistant.components.climate.const import (
    ATTR_CURRENT_TEMPERATURE,
    ATTR_FAN_MODE,
    ATTR_HVAC_ACTION,
    ATTR_HVAC_MODE,
    ATTR_PRESET_MODE,
    CURRENT_HVAC_COOL,
    CURRENT_HVAC_HEAT,
    CURRENT_HVAC_IDLE,
    CURRENT_HVAC_OFF,
    DOMAIN,
    FAN_OFF,
    HVAC_MODES,
    HVAC_MODE_COOL,
    HVAC_MODE_HEAT,
    HVAC_MODE_OFF,
    PRESET_AWAY,
    PRESET_NONE,
    SERVICE_SET_TEMPERATURE,
    SERVICE_SET_HVAC_MODE,
    SERVICE_SET_PRESET_MODE,
    SERVICE_SET_FAN_MODE,
    SUPPORT_PRESET_MODE,
    SUPPORT_TARGET_TEMPERATURE,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    TEMP_CELSIUS,
    TEMP_FAHRENHEIT,
    CONF_NAME,
    CONF_TEMPERATURE_UNIT,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import DOMAIN as HA_DOMAIN, callback
from homeassistant.helpers.event import async_track_state_change
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import HomeAssistantType, ConfigType

from .preset import (
    Preset,
    build_presets,
    PRESET_SCHEMA,
    METHOD_ALWAYS,
    METHOD_MOTION,
    METHOD_WEIGHTED_MOTION,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Custom Thermostat"

CONF_HEAT_THERMOSTAT = "heat_thermostat"
CONF_COOL_THERMOSTAT = "cool_thermostat"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_INITIAL_PRESET = "initial_preset"
CONF_MAX_TEMP = "max_temp"
CONF_MIN_TEMP = "min_temp"
CONF_PRECISION = "precision"
CONF_DEADBAND = "deadband"
CONF_TARGET_TEMP = "target_temp"
CONF_PRESETS = "presets"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Inclusive(CONF_HEAT_THERMOSTAT, "thermostats"): cv.entity_id,
        vol.Inclusive(CONF_COOL_THERMOSTAT, "thermostats"): cv.entity_id,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVAC_MODE_COOL, HVAC_MODE_HEAT, HVAC_MODE_OFF]
        ),
        vol.Optional(CONF_INITIAL_PRESET): cv.string,
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_DEADBAND, default=0.5): vol.Coerce(float),
        vol.Optional(CONF_PRESETS): PRESET_SCHEMA
    }
)

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the custom thermostat platform."""
    name = config.get(CONF_NAME)
    heat_thermostat = config.get(CONF_HEAT_THERMOSTAT)
    cool_thermostat = config.get(CONF_COOL_THERMOSTAT)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)
    target_temp = config.get(CONF_TARGET_TEMP)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    initial_preset = config.get(CONF_INITIAL_PRESET)
    precision = config.get(CONF_PRECISION)
    deadband = config.get(CONF_DEADBAND)
    presets = build_presets(config.get(CONF_PRESETS))
    unit = hass.config.units.temperature_unit

    thermostat = CustomThermostat(
        name,
        heat_thermostat,
        cool_thermostat,
        min_temp,
        max_temp,
        target_temp,
        initial_hvac_mode,
        presets,
        initial_preset,
        precision,
        deadband,
        unit,
    )

    async_add_entities(
        [
            thermostat,
        ]
    )

class CustomThermostat(ClimateEntity, RestoreEntity):
    def __init__(
        self,
        name: str,
        heat_thermostat: str,
        cool_thermostat: str,
        min_temp: float,
        max_temp: float,
        target_temp: float,
        initial_hvac_mode: str,
        presets: List[Preset],
        initial_preset: str,
        precision: float,
        deadband: float,
        unit: str,
    ):
        self._name = name
        self._unit = unit
        self._target_temp = target_temp
        self._heat_slave = heat_thermostat
        self._cool_slave = cool_thermostat
        self._min_temp = min_temp
        self._max_temp = max_temp

        self._sensor_temps: Dict[str, float] = {}
        self._motion_history: Dict[str, List[datetime]] = {}
        self._calc_temp = None
        self._slave_cur_temp = None
        self._slave_target_temp = None
        self._hvac_mode = initial_hvac_mode
        self._fan_mode = FAN_OFF
        self._precision = precision
        self._deadband = deadband
        self._action = None
        self._presets: Dict[str, Preset] = {p.name: p for p in presets}
        self._preset_mode = initial_preset
        self._temp_lock = asyncio.Lock()

        self._active = False
        self._slave_override = False
        self._largest_lookback_period = max([p.minutes for p in self._presets.values()])
        self._attrs: Dict[str, object] = {}

    async def async_added_to_hass(self):
        await super().async_added_to_hass()

        self._async_unsub_sensor_state_changed = async_track_state_change(
            self.hass, self._sensor_entity_ids(), self._async_sensor_changed
        )
        self._async_unsub_motion_sensors = async_track_state_change(
            self.hass, self._motion_sensor_entity_ids(), self._async_motion_sensor_changed
        )

        thermostat_entities = []
        if self._heat_slave:
            thermostat_entities.append(self._heat_slave)
        if self._cool_slave:
            thermostat_entities.append(self._cool_slave)
        self._async_unsub_slave_state_changed = async_track_state_change(
            self.hass, thermostat_entities, self._async_slave_changed
        )

        @callback
        def _async_startup(event):
            """Init on startup."""
            self.async_schedule_update_ha_state(True)

        self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_START, _async_startup)

        # Check if we have old state
        old_state = await self.async_get_last_state()
        _LOGGER.info("climate.%s old state: %s", self._name, old_state)

        if self._preset_mode is None:
            if old_state is not None:
                self._preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
        if self._preset_mode is None:
            self._preset_mode = next(iter(self._presets.keys()))

        if self._hvac_mode is None:
            if old_state is not None:
                self._hvac_mode = old_state.state
        if self._hvac_mode is None:
            self._hvac_mode = HVAC_MODE_OFF

        if self._target_temp is None:
            # If we have a previously saved temperature
            if old_state is not None and ATTR_TEMPERATURE in old_state.attributes:
                self._target_temp = float(
                    old_state.attributes[ATTR_TEMPERATURE])
            else:
                preset_target = self._get_preset_target_temp()
                if preset_target is not None:
                    self._target_temp = preset_target
                else:
                    self._target_temp = float((self.min_temp + self.max_temp)/2)
                    _LOGGER.warning("climate.%s - Undefined target temperature,"
                                    "falling back to %s", self._name , self._target_temp)
        

    async def async_will_remove_from_hass(self):
        if self._async_unsub_sensor_state_changed is not None:
            self._async_unsub_sensor_state_changed()
            self._async_unsub_sensor_state_changed = None
        if self._async_unsub_slave_state_changed is not None:
            self._async_unsub_slave_state_changed()
            self._async_unsub_slave_state_changed = None
        if self._async_unsub_motion_sensors is not None:
            self._async_unsub_motion_sensors()
            self._async_unsub_motion_sensors = None

    @property
    def should_poll(self):
        return False

    @property
    def name(self):
        return self._name

    @property
    def precision(self):
        """Return the precision of the system."""
        if self._precision is not None:
            return self._precision
        return super().precision

    @property
    def temperature_unit(self):
        """Return the unit of measurement."""
        return self._unit

    @property
    def current_temperature(self):
        """Return the sensor temperature."""
        return self._calc_temp

    @property
    def target_temperature(self):
        return self._target_temp
    
    @property
    def min_temp(self):
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self):
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    @property
    def fan_mode(self):
        return self._fan_mode

    @property
    def hvac_mode(self):
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_modes(self):
        return [HVAC_MODE_OFF, HVAC_MODE_HEAT, HVAC_MODE_COOL]

    @property
    def hvac_action(self):
        """Return the current running hvac operation if supported.
        Need to be one of CURRENT_HVAC_*.
        """
        return self._action

    @property
    def preset_mode(self):
        return self._preset_mode

    @property
    def preset_modes(self):
        return list(self._presets.keys())

    @property
    def supported_features(self):
        return SUPPORT_TARGET_TEMPERATURE | SUPPORT_PRESET_MODE

    @property
    def device_state_attributes(self):
        return self._attrs

    async def async_update(self):
        await self._async_set_slaves_mode()

        thermostat = self._thermostat()
        if thermostat is None:
            return
        entity = self.hass.states.get(thermostat)
        if entity is None:
            _LOGGER.error("unable to find entity: %s", thermostat)
            return

        await self._async_slave_changed(thermostat, None, entity)

    async def async_set_hvac_mode(self, hvac_mode):
        self._hvac_mode = hvac_mode
        self._slave_override = False

        await self._async_set_slaves_mode()

        new_target = self._get_preset_target_temp()
        if new_target is not None:
            self._target_temp = new_target
            await self._async_control_slave()

        if hvac_mode == HVAC_MODE_OFF:
            self._action = CURRENT_HVAC_OFF
        
        self.async_write_ha_state()
    
    async def _async_set_slaves_mode(self):
        states = {}
        states[self._heat_slave] = HVAC_MODE_OFF
        states[self._cool_slave] = HVAC_MODE_OFF
        new_thermostat = self._thermostat()
        states[new_thermostat] = self._hvac_mode

        for therm, mode in states.items():
            if therm is None:
                continue
            data = {
                ATTR_ENTITY_ID: therm,
                ATTR_HVAC_MODE: mode,
            }
            await self.hass.services.async_call(
                DOMAIN, SERVICE_SET_HVAC_MODE, data
            )

    async def async_set_preset_mode(self, preset_mode):
        if preset_mode not in self.preset_modes:
            _LOGGER.error("Unrecognized preset: %s", preset_mode)
            return
        self._preset_mode = preset_mode
        new_target = self._get_preset_target_temp()
        if new_target is not None:
            self._target_temp = new_target

        self._calc_temp = self._get_calculated_temp()
        self._slave_override = False
        await self._async_control_slave()
        self.async_write_ha_state()

    async def async_set_fan_mode(self, fan_mode):
        self._fan_mode = fan_mode
        thermostat = self._thermostat()
        if thermostat is None:
            return
        data = {
            ATTR_ENTITY_ID: thermostat,
            ATTR_FAN_MODE: fan_mode,
        }
        await self.hass.services.async_call(
            DOMAIN, SERVICE_SET_FAN_MODE, data
        )

    async def async_set_temperature(self, **kwargs):
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return
        self._target_temp = temperature
        self._slave_override = False
        await self._async_control_slave()
        self.async_write_ha_state()

    async def _async_sensor_changed(self, entity_id, old_state, new_state):
        """Handle temperature changes."""
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(entity_id, new_state)
        await self._async_control_slave()
        self.async_write_ha_state()

    async def _async_motion_sensor_changed(self, entity_id, old_state, new_state):
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        if entity_id not in self._motion_history:
            self._motion_history[entity_id] = []
        
        if new_state.state == STATE_ON:
            self._motion_history[entity_id].append(datetime.utcnow())

        self._calc_temp = self._get_calculated_temp()
        await self._async_control_slave()
        self.async_write_ha_state()

    async def _async_slave_changed(self, entity_id, old_state, new_state):
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return
        
        current_thermostat = self._thermostat()
        if entity_id != current_thermostat:
            return

        self._slave_cur_temp = new_state.attributes.get(ATTR_CURRENT_TEMPERATURE)
        self._action = new_state.attributes.get(ATTR_HVAC_ACTION)
        slave_target = new_state.attributes.get(ATTR_TEMPERATURE)

        if None not in (slave_target, self._slave_target_temp):
            diff = slave_target - self._slave_target_temp
            if abs(diff) >= self.precision:
                _LOGGER.debug("Slave set target to %s, ours is %s. Noting override", slave_target, self._slave_target_temp)
                self._slave_override = True

        await self._async_control_slave()
        self.async_write_ha_state()


    @callback
    def _async_update_temp(self, entity_id, state):
        val = None
        try:
            val = float(state.state)
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor %s: %s", entity_id, ex)
            return

        self._sensor_temps[entity_id] = val
        self._calc_temp = self._get_calculated_temp()

    async def _async_control_slave(self):
        _LOGGER.debug(
            "Custom thermostat. Calc: %s. Slave temp: %s. Target: %s",
            self._calc_temp,
            self._slave_cur_temp,
            self._target_temp,
        )
        async with self._temp_lock:
            if not self._active and None not in (self._calc_temp, self._slave_cur_temp, self._target_temp):
                self._active = True
                _LOGGER.info(
                    "Obtained current, thermostat, and target temperature. "
                    "Custom thermostat active. %s, %s, %s",
                    self._calc_temp,
                    self._slave_cur_temp,
                    self._target_temp,
                )

            if not self._active or self._hvac_mode == HVAC_MODE_OFF:
                return

            if self._slave_override:
                return

            err = self._target_temp - self._calc_temp
            buffer = 1 if err > 0 else -1
            if abs(err) < self._deadband:
                buffer = 0
            slave_target = self._slave_cur_temp + err + buffer
            slave_target = min(slave_target, self.max_temp)
            slave_target = max(slave_target, self.min_temp)

            self._slave_target_temp = slave_target
            await self._async_set_slave_target(slave_target)

            self._trim_motion_history()

    async def _async_set_slave_target(self, target):
        thermostat = self._thermostat()
        if thermostat is None:
            return
        data = {
            ATTR_ENTITY_ID: thermostat,
            ATTR_TEMPERATURE: target,
        }
        await self.hass.services.async_call(
            DOMAIN, SERVICE_SET_TEMPERATURE, data
        )

    def _get_calculated_temp(self):
        now = datetime.utcnow()
        preset = self._presets[self._preset_mode]

        valid_sensors = list(filter(lambda s: self._sensor_temps.get(s.temp_sensor) is not None, preset.sensors))

        if not valid_sensors:
            return None
        
        first_slot = quantize_minute(now - timedelta(minutes=preset.minutes))
        motion_instances = {}
        raw_weights = {}
        for sensor in valid_sensors:
            if sensor.motion_sensor:
                recent_events = filter(lambda dt: dt > first_slot, self._motion_history.get(sensor.motion_sensor, []))
                quantized = set(map(quantize_minute, recent_events))
                instances = len(quantized)
                motion_instances[sensor.temp_sensor] = instances
                if instances > 0 and sensor.method == METHOD_WEIGHTED_MOTION:
                    raw_weights[sensor.temp_sensor] = instances

        base_raw_weight = preset.minutes
        total_raw_weight = sum(raw_weights.values())
        if total_raw_weight > 0:
            base_raw_weight = total_raw_weight / len(raw_weights)

        for sensor in valid_sensors:
            if sensor.method == METHOD_ALWAYS:
                raw_weights[sensor.temp_sensor] = base_raw_weight
            elif sensor.method == METHOD_MOTION:
                raw_weights[sensor.temp_sensor] = base_raw_weight if motion_instances.get(sensor.temp_sensor) else 0
            elif sensor.method == METHOD_WEIGHTED_MOTION:
                if sensor.temp_sensor not in motion_instances:
                    raw_weights[sensor.temp_sensor] = 0
            
        total_raw_weight = sum(raw_weights.values())

        if total_raw_weight == 0:
            # No sensors reporting motion, and no "always" sensors to set the baseline
            # so instead, we average all of the motion sensors
            for sensor in valid_sensors:
                if sensor.method in (METHOD_MOTION, METHOD_WEIGHTED_MOTION):
                    raw_weights[sensor.temp_sensor] = base_raw_weight

        total_raw_weight = sum(raw_weights.values())

        _LOGGER.debug("raw_weights: %s", raw_weights)

        final_weights = {}
        for k, v in raw_weights.items():
            final_weights[k] = v / total_raw_weight

        weights_attr = self._attrs.get("sensor_weights", {})
        for k in weights_attr.keys():
            weights_attr[k] = 0

        for k, v in final_weights.items():
            weights_attr[k] = v
        self._attrs["sensor_weights"] = weights_attr

        accum = 0
        for k, v in final_weights.items():
            accum += self._sensor_temps.get(k) * v
        return accum

    def _trim_motion_history(self):
        now = datetime.utcnow()
        first_slot = quantize_minute(now - timedelta(minutes=self._largest_lookback_period))
        new_history = {}
        for k, history in self._motion_history.items():
            new_history[k] = filter(lambda d: d > first_slot, history)
        self._motion_history = new_history

    def _sensor_entity_ids(self):
        ids = set()
        for preset in self._presets.values():
            for sensor in preset.sensors:
                ids.add(sensor.temp_sensor)

        return list(ids)

    def _motion_sensor_entity_ids(self):
        ids = set()
        for preset in self._presets.values():
            for sensor in preset.sensors:
                if sensor.motion_sensor:
                    ids.add(sensor.motion_sensor)
        
        return list(ids)

    def _thermostat(self, mode=None):
        hvac_mode = mode or self._hvac_mode
        if hvac_mode == HVAC_MODE_HEAT:
            return self._heat_slave
        elif hvac_mode == HVAC_MODE_COOL:
            return self._cool_slave
        else:
            return None

    def _get_preset_target_temp(self):
        preset = self._presets[self._preset_mode]
        if self.hvac_mode == HVAC_MODE_HEAT and preset.heat_target is not None:
            return preset.heat_target
        elif self.hvac_mode == HVAC_MODE_COOL and preset.cool_target is not None:
            return preset.cool_target
        return None

def quantize_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)