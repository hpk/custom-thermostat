from homeassistant.helpers.entity import Entity

class CalculatedTemperature(Entity):
    def __init__(self, thermostat):
        self._state = None
        self._thermostat = thermostat

    @property
    def name(self):
        """Return the name of the sensor."""
        return "{0} Temperature".format(self._thermostat.name)

    @property
    def state(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        return self._thermostat.temperature_unit

    def update(self):
        """Fetch new state data for the sensor.
        This is the only method that should fetch new data for Home Assistant.
        """
        self._state = self._thermostat.current_temperature
