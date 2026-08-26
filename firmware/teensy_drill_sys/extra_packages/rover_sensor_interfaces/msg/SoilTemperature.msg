# Custom telemetry for a 1-Wire DS18B20 Soil Temp Sensor

# The 8-byte 1-Wire ROM address
uint8[8] device_address

# The measured temperature in degrees Celsius
float32 temperature_c

# Hardware status flag (True if read was successful)
bool is_valid
