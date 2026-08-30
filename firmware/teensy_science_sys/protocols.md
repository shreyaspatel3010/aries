# 🛰️ ERC Science Module Command & Calibration Guide

## 📡 Monitoring Telemetry
Before sending commands, open a dedicated WSL terminal tab to watch the telemetry array update in real-time. This is the exact same process used to monitor the load cell weights during the containment calibration[cite: 2].

```bash
ros2 topic echo /science/telemetry

```

---

## 🎛️ Master Command Table

Use this base command in your terminal, replacing `XX` with the 2-digit codes below:
`ros2 topic pub /science/sensor_cmd std_msgs/msg/UInt8 "{data: XX}" --once`

| Sensor | Array Index | Init Command (XX) | Read Command (XX) |
| --- | --- | --- | --- |
| **pH Sensor** | 0 | `01` | `02` |
| **Soil Moisture** | 1 | `11` | `12` |
| **Soil EC / TDS** | 2 | `21` | `22` |
| **ORP Meter** | 3 | `31` | `32` |
| **DS18B20 Temp** | 4 | `41` | `42` |
| **BME688** | 5, 6, 7, 8 | `51` | `52` |
| **SCD41 CO2** | 9 | `91` | `92` |

---

## 🔬 Calibration & Testing Procedures

### 1. pH Sensor

* **Test:** Send `01`, then `02`.
* **Calibrate:** Wash the probe with distilled water, then dip it into a standard pH 7.0 buffer solution. If the array outputs 6.8, you have a -0.2 offset. Update the `m_calibration_offset` in your code (or via a future ROS command) to map the raw voltage to exactly 7.0. Repeat with a pH 4.0 buffer to verify the slope.

### 2. Soil EC / TDS (SEN0244)

* **Test:** Send `21`, then `22`. The current code returns the raw ADC voltage.
* **Calibrate:** Clean the prongs with distilled water (should read near 0). Submerge the prongs in a known calibration fluid (e.g., 1413 µS/cm or standard TDS ppm solution) and record the raw reading.
* **Scaling:** You will calculate the scale factor by dividing the raw reading by the known fluid concentration. Replace your placeholder macro in `main.cpp` with this final calculated number.



### 3. ORP Meter (SEN0165)

* **Test:** Send `31`, then `32`.
* **Calibrate:** The SEN0165 has a physical calibration button on its circuit board. Keep the probe disconnected (or short the BNC connector with a wire), press and hold the calibration button, and send the read command (`32`). Adjust the potentiometer on the board until the ROS topic outputs exactly **0.0 mV**.

### 4. DS18B20 Soil Temp

* **Test:** Send `41`, then `42`. Because this is a slow sensor, the Teensy will safely yield back to the executor and wait 800ms before placing the value into Index 4.
* **Calibrate:** These are factory-calibrated. To test accuracy, place the metal tip in a glass of ice water; it should read very close to 0°C.
* **Warning:** After sending the read signal, you will have to WAIT FOR 5 SECONDS ATLEAST to send the next read, otherwise the sensor might throw up some errors, or maybe not.

### 5. BME688 Environmental

* **Test:** Send `51`, then `52`. This single command populates four array slots simultaneously: Index 5 (Temp), 6 (Humidity), 7 (Pressure), and 8 (Gas Resistance).
* **Calibrate:** Fully digital and factory-calibrated. Verify the readings match the ambient conditions in your room.

### 6. SCD41 CO2

* **Test:** Send `91`, then `92`.
* **Calibrate:** The SCD41 features Automatic Self-Calibration (ASC). For a manual baseline test, take the rover outside into fresh air and send the read command; it should read approximately 400 ppm.
