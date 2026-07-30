"""Realistic Industrial IoT Sensor Data Generator.

Generates multi-sensor time-series telemetry representing degrading industrial machinery
(turbofans, pumps, compressors, CNC machines) with physical sensor interactions,
random noise, error codes, and maintenance history.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_iot_sensor_data(
    num_engines: int = 100,
    min_cycles: int = 120,
    max_cycles: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic Industrial IoT sensor telemetry with degradation dynamics.

    Sensors Included:
        - Temperature (°C)
        - Pressure (PSI)
        - Vibration (mm/s RMS)
        - Voltage (V)
        - Current (A)
        - Humidity (%)
        - Power Consumption (kW)
        - RPM (Revolutions Per Minute)
        - Torque (Nm)
        - Machine Type (Categorical: Pump, Compressor, Turbofan, Turbine)
        - Operating Hours (Cumulative)
        - Error Codes (0: None, 1: Minor Warning, 2: Thermal Spike, 3: High Vibration)
        - Maintenance History (Days since last maintenance)

    Returns:
        pd.DataFrame: Simulated multi-sensor IoT telemetry dataframe.
    """
    np.random.seed(seed)
    records = []

    machine_types = ["Turbofan_X100", "Centrifugal_Pump_V2", "Industrial_Compressor_C5", "Gas_Turbine_GT9"]

    for engine_id in range(1, num_engines + 1):
        machine = np.random.choice(machine_types)
        total_life = np.random.randint(min_cycles, max_cycles + 1)
        base_operating_hours = np.random.randint(500, 5000)

        # Base operational parameters
        base_temp = np.random.normal(70.0, 2.0)
        base_pressure = np.random.normal(150.0, 5.0)
        base_rpm = np.random.normal(3000.0, 50.0)
        base_voltage = np.random.normal(400.0, 5.0)
        days_since_maint = np.random.randint(10, 180)

        for cycle in range(1, total_life + 1):
            # Degradation factor: exponential growth in late cycles
            degradation_pct = cycle / total_life
            degrad_exponent = np.power(degradation_pct, 2.5)

            # Sensor degradation physics
            temp = base_temp + (25.0 * degrad_exponent) + np.random.normal(0, 0.8)
            pressure = base_pressure - (18.0 * degrad_exponent) + np.random.normal(0, 1.2)
            vibration = 0.5 + (4.5 * degrad_exponent) + np.random.normal(0, 0.15)
            rpm = base_rpm - (200.0 * degrad_exponent) + np.random.normal(0, 15.0)
            torque = 120.0 + (35.0 * degrad_exponent) + np.random.normal(0, 2.0)
            current = 15.0 + (12.0 * degrad_exponent) + np.random.normal(0, 0.5)
            voltage = base_voltage - (8.0 * degrad_exponent) + np.random.normal(0, 1.0)
            power = (voltage * current * np.sqrt(3) * 0.85) / 1000.0  # kW formula approximation
            humidity = 45.0 + np.random.normal(0, 3.0)

            # Operational state & error codes
            error_code = 0
            if degradation_pct > 0.85:
                error_code = np.random.choice([2, 3], p=[0.5, 0.5])
            elif degradation_pct > 0.70:
                error_code = np.random.choice([0, 1], p=[0.6, 0.4])

            rul = total_life - cycle
            op_hours = base_operating_hours + (cycle * 8)
            maint_days = days_since_maint + (cycle // 3)

            records.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "machine_type": machine,
                    "temperature": round(temp, 2),
                    "pressure": round(pressure, 2),
                    "vibration": round(vibration, 4),
                    "voltage": round(voltage, 2),
                    "current": round(current, 2),
                    "humidity": round(humidity, 2),
                    "power_consumption": round(power, 2),
                    "rpm": round(rpm, 1),
                    "torque": round(torque, 2),
                    "operating_hours": op_hours,
                    "error_code": error_code,
                    "days_since_maintenance": maint_days,
                    "RUL": rul,
                }
            )

    df = pd.DataFrame(records)
    logger.info(f"Generated synthetic IoT sensor dataset: {df.shape[0]} rows across {num_engines} engines.")
    return df
