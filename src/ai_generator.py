"""
AI-Powered PCB Generator using Gemini

Generates realistic PCB layouts from natural language descriptions.
Uses Google's Gemini API to interpret descriptions and create component layouts.

Usage:
    generator = AIPCBGenerator(api_key="your-key")
    layout = generator.generate_from_description(
        "Arduino Uno with ATmega328P, USB-B connector, 5V regulator, and 6 LEDs"
    )
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re


class ComponentType(Enum):
    """Standard PCB component types with thermal properties."""
    MICROCONTROLLER = "Microcontroller/MCU"
    CPU = "CPU/Processor"
    FPGA = "FPGA"
    POWER_IC = "Power IC/Regulator"
    MOSFET = "MOSFET"
    TRANSISTOR = "Transistor"
    RESISTOR = "Resistor"
    CAPACITOR = "Capacitor"
    INDUCTOR = "Inductor"
    LED = "LED"
    CONNECTOR = "Connector"
    CRYSTAL = "Crystal/Oscillator"
    MEMORY = "Memory (RAM/Flash)"
    SENSOR = "Sensor"
    DIODE = "Diode"
    FUSE = "Fuse"
    RELAY = "Relay"
    TRANSFORMER = "Transformer"
    BATTERY = "Battery/Holder"
    SWITCH = "Switch/Button"
    ANTENNA = "Antenna"
    USB = "USB Controller"
    ETHERNET = "Ethernet PHY"
    WIFI = "WiFi Module"
    BLUETOOTH = "Bluetooth Module"


# Thermal properties database (typical values)
THERMAL_PROPERTIES = {
    ComponentType.MICROCONTROLLER: {"power_mw": (100, 500), "theta_jb": 40},
    ComponentType.CPU: {"power_mw": (500, 5000), "theta_jb": 25},
    ComponentType.FPGA: {"power_mw": (200, 3000), "theta_jb": 30},
    ComponentType.POWER_IC: {"power_mw": (200, 2000), "theta_jb": 25},
    ComponentType.MOSFET: {"power_mw": (100, 3000), "theta_jb": 30},
    ComponentType.TRANSISTOR: {"power_mw": (50, 500), "theta_jb": 100},
    ComponentType.RESISTOR: {"power_mw": (10, 500), "theta_jb": 200},
    ComponentType.CAPACITOR: {"power_mw": (1, 50), "theta_jb": 500},
    ComponentType.INDUCTOR: {"power_mw": (50, 500), "theta_jb": 50},
    ComponentType.LED: {"power_mw": (20, 200), "theta_jb": 100},
    ComponentType.CONNECTOR: {"power_mw": (1, 10), "theta_jb": 1000},
    ComponentType.CRYSTAL: {"power_mw": (1, 20), "theta_jb": 300},
    ComponentType.MEMORY: {"power_mw": (50, 500), "theta_jb": 50},
    ComponentType.SENSOR: {"power_mw": (10, 100), "theta_jb": 150},
    ComponentType.DIODE: {"power_mw": (10, 200), "theta_jb": 150},
    ComponentType.WIFI: {"power_mw": (100, 500), "theta_jb": 60},
    ComponentType.BLUETOOTH: {"power_mw": (50, 200), "theta_jb": 80},
    ComponentType.USB: {"power_mw": (50, 200), "theta_jb": 70},
    ComponentType.ETHERNET: {"power_mw": (100, 500), "theta_jb": 50},
}


@dataclass
class AIComponent:
    """A component parsed from AI description."""
    name: str
    type: ComponentType
    count: int = 1
    power_mw: float = 0  # Will be estimated if not specified
    package: str = ""  # e.g., "QFP-48", "0805", "TO-220"
    notes: str = ""
    
    def estimate_size(self) -> Tuple[int, int]:
        """Estimate component size in grid units based on type and package."""
        base_sizes = {
            ComponentType.MICROCONTROLLER: (12, 12),
            ComponentType.CPU: (20, 20),
            ComponentType.FPGA: (25, 25),
            ComponentType.POWER_IC: (8, 8),
            ComponentType.MOSFET: (6, 8),
            ComponentType.TRANSISTOR: (4, 4),
            ComponentType.RESISTOR: (2, 4),
            ComponentType.CAPACITOR: (2, 3),
            ComponentType.INDUCTOR: (8, 8),
            ComponentType.LED: (3, 3),
            ComponentType.CONNECTOR: (8, 15),
            ComponentType.CRYSTAL: (4, 6),
            ComponentType.MEMORY: (10, 6),
            ComponentType.SENSOR: (5, 5),
            ComponentType.WIFI: (15, 10),
            ComponentType.BLUETOOTH: (10, 8),
            ComponentType.USB: (6, 6),
            ComponentType.ETHERNET: (8, 8),
        }
        return base_sizes.get(self.type, (5, 5))


@dataclass  
class AIBoardSpec:
    """Complete board specification from AI parsing."""
    name: str = "Custom PCB"
    description: str = ""
    board_type: str = "general"  # arduino, raspberry_pi, power_supply, iot, etc.
    size_mm: Tuple[float, float] = (100, 100)
    layers: int = 2
    components: List[AIComponent] = field(default_factory=list)
    total_power_w: float = 0
    copper_weight_oz: float = 1.0
    has_ground_plane: bool = True
    estimated_complexity: str = "medium"
    thermal_notes: List[str] = field(default_factory=list)


class AIPCBGenerator:
    """Generate PCB layouts from natural language using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini API key."""
        # Try multiple sources for API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Also try Streamlit secrets if available
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass
        
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                try:
                    # Fallback to old API if new one not available
                    import google.generativeai as genai_old
                    genai_old.configure(api_key=self.api_key)
                    self.client = genai_old.GenerativeModel('gemini-2.0-flash')
                except Exception as e:
                    print(f"Warning: Could not initialize Gemini: {e}")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
    
    def parse_description(self, description: str) -> AIBoardSpec:
        """Use Gemini to parse a natural language PCB description."""
        if not self.client:
            return self._fallback_parse(description)
        
        prompt = f"""You are a PCB design expert. Parse this PCB description and extract component information.

Description: "{description}"

Return a JSON object with this exact structure (no markdown, just JSON):
{{
    "name": "Board name",
    "board_type": "one of: arduino, raspberry_pi, power_supply, motor_driver, iot, sensor, audio, display, communication, general",
    "size_mm": [width, height],
    "layers": 2 or 4,
    "copper_weight_oz": 1.0 or 2.0,
    "has_ground_plane": true/false,
    "total_power_estimate_w": number,
    "complexity": "simple/medium/complex",
    "components": [
        {{
            "name": "Component name (e.g., ATmega328P)",
            "type": "one of: MICROCONTROLLER, CPU, FPGA, POWER_IC, MOSFET, TRANSISTOR, RESISTOR, CAPACITOR, INDUCTOR, LED, CONNECTOR, CRYSTAL, MEMORY, SENSOR, DIODE, WIFI, BLUETOOTH, USB, ETHERNET",
            "count": number,
            "power_mw": estimated power in milliwatts (be realistic),
            "package": "package type if known (QFP-32, 0805, TO-220, etc.)",
            "notes": "any thermal concerns"
        }}
    ],
    "thermal_notes": ["List of thermal design considerations"]
}}

Be realistic with power estimates:
- ATmega328P at 16MHz: ~50-100mW
- ESP32: ~200-500mW  
- Voltage regulators: depends on dropout and current
- LEDs: ~20-60mW each
- Power MOSFETs: based on RDS(on) and current

Return ONLY valid JSON, no explanation."""

        try:
            # Try new google.genai API first
            from google import genai
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            text = response.text.strip()
            
            # Clean up response (remove markdown if present)
            if text.startswith("```"):
                text = re.sub(r'^```json?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            
            data = json.loads(text)
            return self._json_to_spec(data)
            
        except Exception as e:
            print(f"AI parsing failed: {e}, using fallback")
            return self._fallback_parse(description)
    
    def _json_to_spec(self, data: dict) -> AIBoardSpec:
        """Convert JSON response to AIBoardSpec."""
        components = []
        
        for comp_data in data.get("components", []):
            try:
                comp_type = ComponentType[comp_data["type"]]
            except (KeyError, ValueError):
                comp_type = ComponentType.RESISTOR
            
            components.append(AIComponent(
                name=comp_data.get("name", "Unknown"),
                type=comp_type,
                count=comp_data.get("count", 1),
                power_mw=comp_data.get("power_mw", 50),
                package=comp_data.get("package", ""),
                notes=comp_data.get("notes", "")
            ))
        
        size = data.get("size_mm", [100, 100])
        
        return AIBoardSpec(
            name=data.get("name", "Custom PCB"),
            description=data.get("description", ""),
            board_type=data.get("board_type", "general"),
            size_mm=(size[0], size[1]),
            layers=data.get("layers", 2),
            components=components,
            total_power_w=data.get("total_power_estimate_w", 1.0),
            copper_weight_oz=data.get("copper_weight_oz", 1.0),
            has_ground_plane=data.get("has_ground_plane", True),
            estimated_complexity=data.get("complexity", "medium"),
            thermal_notes=data.get("thermal_notes", [])
        )
    
    def _fallback_parse(self, description: str) -> AIBoardSpec:
        """Fallback parsing without AI - keyword based."""
        desc_lower = description.lower()
        components = []
        
        # Keywords ordered by specificity - more specific patterns first
        # Some groups are mutually exclusive, others can all match
        
        # ===== MUTUALLY EXCLUSIVE GROUPS =====
        # Microcontrollers - most specific first (only one matches)
        mcu_found = False
        for keyword, comp_type, name, power in [
            ("atmega328", ComponentType.MICROCONTROLLER, "ATmega328P", 100),
            ("atmega", ComponentType.MICROCONTROLLER, "ATmega", 100),
            ("esp32", ComponentType.MICROCONTROLLER, "ESP32", 400),
            ("esp8266", ComponentType.MICROCONTROLLER, "ESP8266", 300),
            ("stm32", ComponentType.MICROCONTROLLER, "STM32", 200),
            ("arduino", ComponentType.MICROCONTROLLER, "Arduino MCU", 150),
        ]:
            if keyword in desc_lower and not mcu_found:
                components.append(AIComponent(name=name, type=comp_type, count=1, power_mw=power))
                mcu_found = True
                break
        
        # CPUs
        if "raspberry" in desc_lower:
            components.append(AIComponent("Raspberry Pi", ComponentType.CPU, 1, 2500))
        
        # Voltage regulators - specific part numbers first (only one matches)
        reg_found = False
        for keyword, name, power in [
            ("7805", "7805 Regulator", 1000),
            ("78xx", "78xx Regulator", 1000),
            ("ldo", "LDO Regulator", 300),
            ("voltage regulator", "Voltage Regulator", 500),
            ("regulator", "Voltage Regulator", 400),
        ]:
            if keyword in desc_lower and not reg_found:
                components.append(AIComponent(name=name, type=ComponentType.POWER_IC, count=1, power_mw=power))
                reg_found = True
                break
        
        # ===== MOTOR DRIVER SPECIFIC =====
        if "motor driver" in desc_lower or "motor" in desc_lower:
            # Check for H-bridge
            if "h-bridge" in desc_lower or "h bridge" in desc_lower:
                components.append(AIComponent("H-Bridge Controller", ComponentType.POWER_IC, 1, 800))
            elif "motor driver" in desc_lower:
                components.append(AIComponent("Motor Driver IC", ComponentType.POWER_IC, 1, 600))
        
        # ===== COUNTABLE COMPONENTS (can all match with quantities) =====
        countable_patterns = [
            ("mosfet", ComponentType.MOSFET, "Power MOSFET", 500),
            ("transistor", ComponentType.TRANSISTOR, "Transistor", 100),
            ("led", ComponentType.LED, "LED", 40),
            ("capacitor", ComponentType.CAPACITOR, "Capacitor", 5),
            ("resistor", ComponentType.RESISTOR, "Resistor", 10),
            ("inductor", ComponentType.INDUCTOR, "Inductor", 50),
            ("sensor", ComponentType.SENSOR, "Sensor", 50),
            ("button", ComponentType.CONNECTOR, "Button", 1),
        ]
        
        for keyword, comp_type, name, power in countable_patterns:
            if keyword in desc_lower:
                count = 1
                # Look for patterns with word boundary: "4 mosfets", "6 LEDs", "3 temperature sensors"
                # Use word boundary \b to avoid matching "32" from "ESP32"
                patterns = [
                    rf'\b(\d{{1,2}})\s*x?\s*{keyword}s?\b',  # "4 mosfets" or "4x mosfet"
                    rf'\b(\d{{1,2}})\s+\w+\s+{keyword}s?\b',  # "3 temperature sensors"
                ]
                for pattern in patterns:
                    match = re.search(pattern, desc_lower)
                    if match:
                        count = min(int(match.group(1)), 20)
                        break
                components.append(AIComponent(name=name, type=comp_type, count=count, power_mw=power * count))
        
        # ===== CONNECTIVITY =====
        if "wifi" in desc_lower:
            components.append(AIComponent("WiFi Module", ComponentType.WIFI, 1, 400))
        if "bluetooth" in desc_lower:
            components.append(AIComponent("Bluetooth Module", ComponentType.BLUETOOTH, 1, 150))
        if "ethernet" in desc_lower:
            components.append(AIComponent("Ethernet PHY", ComponentType.ETHERNET, 1, 300))
        if "usb" in desc_lower:
            components.append(AIComponent("USB Connector", ComponentType.CONNECTOR, 1, 10))
        if "can" in desc_lower and "transceiver" in desc_lower:
            components.append(AIComponent("CAN Transceiver", ComponentType.ETHERNET, 1, 100))
        if "mcp2515" in desc_lower:
            components.append(AIComponent("MCP2515 CAN Controller", ComponentType.MICROCONTROLLER, 1, 50))
        if "tja1050" in desc_lower:
            components.append(AIComponent("TJA1050 CAN Transceiver", ComponentType.ETHERNET, 1, 80))
        if "lora" in desc_lower:
            components.append(AIComponent("LoRa Module", ComponentType.WIFI, 1, 150))
        if "oled" in desc_lower or "display" in desc_lower:
            components.append(AIComponent("OLED Display", ComponentType.MEMORY, 1, 30))
        if "sd card" in desc_lower or "sdcard" in desc_lower:
            components.append(AIComponent("SD Card Slot", ComponentType.CONNECTOR, 1, 5))
        if "rtc" in desc_lower or "real time clock" in desc_lower:
            components.append(AIComponent("RTC Module", ComponentType.CRYSTAL, 1, 5))
        if "relay" in desc_lower:
            # Check for count
            match = re.search(r'\b(\d{1,2})\s*relay', desc_lower)
            count = int(match.group(1)) if match else 1
            components.append(AIComponent("Relay", ComponentType.POWER_IC, count, 100 * count))
        if "esd" in desc_lower or "protection" in desc_lower:
            components.append(AIComponent("ESD Protection", ComponentType.DIODE, 4, 4))
        if "isolated" in desc_lower and "power" in desc_lower:
            components.append(AIComponent("Isolated DC-DC", ComponentType.POWER_IC, 1, 200))
        if "adc" in desc_lower:
            components.append(AIComponent("ADC", ComponentType.MICROCONTROLLER, 1, 50))
        if "dac" in desc_lower:
            components.append(AIComponent("DAC", ComponentType.MICROCONTROLLER, 1, 50))
        if "amplifier" in desc_lower or "amp" in desc_lower:
            components.append(AIComponent("Amplifier IC", ComponentType.POWER_IC, 1, 500))
        if "gpio" in desc_lower or "expander" in desc_lower:
            components.append(AIComponent("GPIO Expander", ComponentType.MICROCONTROLLER, 1, 30))
        if "optocoupler" in desc_lower or "opto" in desc_lower:
            components.append(AIComponent("Optocoupler", ComponentType.DIODE, 1, 20))
        if "connector" in desc_lower:
            match = re.search(r'\b(\d{1,2})\s*(?:\w+\s+)?connector', desc_lower)
            count = int(match.group(1)) if match else 1
            components.append(AIComponent("Connector", ComponentType.CONNECTOR, count, 5 * count))
        if "ddr" in desc_lower or "memory" in desc_lower:
            components.append(AIComponent("DDR Memory", ComponentType.MEMORY, 1, 500))
        if "fpga" in desc_lower or "spartan" in desc_lower:
            components.append(AIComponent("FPGA", ComponentType.FPGA, 1, 1500))
        if "jtag" in desc_lower:
            components.append(AIComponent("JTAG Connector", ComponentType.CONNECTOR, 1, 5))
        if "buck" in desc_lower:
            components.append(AIComponent("Buck Converter", ComponentType.POWER_IC, 1, 400))
        if "boost" in desc_lower:
            components.append(AIComponent("Boost Converter", ComponentType.POWER_IC, 1, 400))
        if "lipo" in desc_lower or "charger" in desc_lower:
            components.append(AIComponent("LiPo Charger IC", ComponentType.POWER_IC, 1, 200))
        if "bme280" in desc_lower:
            components.append(AIComponent("BME280 Sensor", ComponentType.SENSOR, 1, 10))
        if "mpu" in desc_lower or "imu" in desc_lower or "accelerometer" in desc_lower:
            components.append(AIComponent("IMU/Accelerometer", ComponentType.SENSOR, 1, 20))
        
        # ===== MISC =====
        if "crystal" in desc_lower or "oscillator" in desc_lower:
            components.append(AIComponent("Crystal Oscillator", ComponentType.CRYSTAL, 1, 5))
        if "heatsink" in desc_lower or "heat sink" in desc_lower:
            # Add some thermal vias indicator
            pass
        
        # If nothing detected, add default components
        if not components:
            components = [
                AIComponent("Generic MCU", ComponentType.MICROCONTROLLER, 1, 200),
                AIComponent("Power Supply", ComponentType.POWER_IC, 1, 300),
                AIComponent("Capacitor", ComponentType.CAPACITOR, 4, 20),
                AIComponent("Resistor", ComponentType.RESISTOR, 6, 60),
            ]
        
        # Determine board type
        board_type = "general"
        if any(x in desc_lower for x in ["arduino", "atmega"]):
            board_type = "arduino"
        elif "power supply" in desc_lower or "power board" in desc_lower or "dc-dc" in desc_lower or "charger" in desc_lower:
            board_type = "power_supply"
        elif any(x in desc_lower for x in ["motor", "driver"]):
            board_type = "motor_driver"
        elif any(x in desc_lower for x in ["wifi", "iot", "esp", "lora"]):
            board_type = "iot"
        elif any(x in desc_lower for x in ["can", "automotive"]):
            board_type = "automotive"
        elif any(x in desc_lower for x in ["fpga", "spartan", "ddr"]):
            board_type = "fpga"
        
        # power_mw already includes count multiplication
        total_power = sum(c.power_mw for c in components) / 1000
        
        return AIBoardSpec(
            name="Parsed PCB",
            description=description,
            board_type=board_type,
            components=components,
            total_power_w=total_power,
            estimated_complexity="medium"
        )
    
    def generate_layout(self, spec: AIBoardSpec, grid_size: int = 128):
        """Generate actual PCB layout arrays from specification."""
        
        copper_density = np.zeros((grid_size, grid_size), dtype=np.float32)
        via_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        component_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        power_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Ground plane base
        if spec.has_ground_plane:
            copper_density += 0.3
        
        # Place components intelligently
        placed_components = []
        margin = 6  # Reduced margin for tighter packing
        
        # Sort by power (place high-power components first for thermal spacing)
        sorted_comps = sorted(
            [(c, i) for i, c in enumerate(spec.components) for _ in range(c.count)],
            key=lambda x: -x[0].power_mw
        )
        
        # Use smart placement algorithm
        positions = self._calculate_positions_smart(sorted_comps, grid_size, margin)
        
        for idx, (comp, _) in enumerate(sorted_comps):
            if idx >= len(positions):
                break
                
            x, y = positions[idx]
            w, h = comp.estimate_size()
            
            # Ensure within bounds
            x = min(x, grid_size - w - margin)
            y = min(y, grid_size - h - margin)
            x = max(x, margin)
            y = max(y, margin)
            
            # Place component
            component_map[y:y+h, x:x+w] = 1.0
            
            # Power distribution (concentrated in center for ICs, distributed for passives)
            # Note: comp.power_mw is TOTAL for all instances, so divide by count
            instance_power_mw = comp.power_mw / comp.count
            power_per_pixel = instance_power_mw / (w * h)
            if comp.type in [ComponentType.MICROCONTROLLER, ComponentType.CPU, ComponentType.FPGA]:
                # Concentrate power in center
                cy, cx = y + h//2, x + w//2
                r = min(w, h) // 3
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dy*dy + dx*dx <= r*r:
                            py, px = cy + dy, cx + dx
                            if 0 <= py < grid_size and 0 <= px < grid_size:
                                power_map[py, px] += power_per_pixel * 2
            else:
                power_map[y:y+h, x:x+w] += power_per_pixel
            
            # Add copper traces to component
            copper_density[y:y+h, x:x+w] = np.maximum(
                copper_density[y:y+h, x:x+w], 
                0.7 if comp.type in [ComponentType.POWER_IC, ComponentType.MOSFET] else 0.5
            )
            
            # Add thermal vias for high-power components
            if instance_power_mw > 200:
                via_count = min(int(instance_power_mw / 100), 9)
                via_spacing = max(w, h) // (int(np.sqrt(via_count)) + 1)
                for vy in range(y + 2, y + h - 2, via_spacing):
                    for vx in range(x + 2, x + w - 2, via_spacing):
                        if 0 <= vy < grid_size and 0 <= vx < grid_size:
                            via_map[vy, vx] = 1.0
            
            placed_components.append({
                'component': comp,
                'position': (x, y),
                'size': (w, h)
            })
        
        # Add trace routing (simplified)
        self._add_traces(copper_density, placed_components, grid_size)
        
        # Add perimeter traces (power rails)
        copper_density[margin:margin+2, margin:grid_size-margin] = 0.8  # Top rail
        copper_density[grid_size-margin-2:grid_size-margin, margin:grid_size-margin] = 0.8  # Bottom
        copper_density[margin:grid_size-margin, margin:margin+2] = 0.8  # Left
        copper_density[margin:grid_size-margin, grid_size-margin-2:grid_size-margin] = 0.8  # Right
        
        # Create layout object
        class GeneratedLayout:
            pass
        
        layout = GeneratedLayout()
        layout.copper_density = np.clip(copper_density, 0, 1)
        layout.via_map = via_map
        layout.component_map = component_map
        layout.power_map = power_map
        layout.placed_components = placed_components
        layout.spec = spec
        
        return layout
    
    def _calculate_positions_smart(self, components: List[Tuple], grid_size: int, margin: int) -> List[Tuple[int, int]]:
        """
        Smart component placement using bin-packing algorithm.
        Places components efficiently to minimize wasted space.
        Components is list of (component, original_index) tuples sorted by power.
        """
        positions = []
        usable = grid_size - 2 * margin
        
        # Calculate total component area to determine optimal packing
        total_area = 0
        comp_sizes = []
        for comp, _ in components:
            w, h = comp.estimate_size()
            # Add minimum spacing between components
            spacing = 4 if comp.power_mw < 100 else 6  # More space for hot components
            comp_sizes.append((w + spacing, h + spacing, w, h))
            total_area += (w + spacing) * (h + spacing)
        
        # Calculate fill ratio - if components fill <50% of board, pack tighter
        board_area = usable * usable
        fill_ratio = total_area / board_area if board_area > 0 else 1
        
        # Use shelf-packing algorithm for efficient placement
        # Track shelves: each shelf has a y position and remaining width
        shelves = []  # [(y_start, current_x, shelf_height)]
        
        for idx, (padded_w, padded_h, actual_w, actual_h) in enumerate(comp_sizes):
            comp, _ = components[idx]
            placed = False
            
            # Adjust placement based on fill ratio
            if fill_ratio < 0.3:
                # Very sparse - use center-focused placement
                center_x = grid_size // 2
                center_y = grid_size // 2
                
                # Spiral outward from center
                spiral_positions = self._generate_spiral_positions(
                    center_x, center_y, grid_size, margin, len(positions)
                )
                if idx < len(spiral_positions):
                    x, y = spiral_positions[idx]
                    x = min(max(x - actual_w // 2, margin), grid_size - actual_w - margin)
                    y = min(max(y - actual_h // 2, margin), grid_size - actual_h - margin)
                    positions.append((x, y))
                    placed = True
            
            if not placed:
                # Try to fit on existing shelf
                for shelf_idx, (shelf_y, shelf_x, shelf_h) in enumerate(shelves):
                    if shelf_x + padded_w <= usable + margin and padded_h <= shelf_h:
                        # Fits on this shelf
                        x = shelf_x
                        y = shelf_y
                        shelves[shelf_idx] = (shelf_y, shelf_x + padded_w, shelf_h)
                        positions.append((x, y))
                        placed = True
                        break
                
                if not placed:
                    # Create new shelf
                    if shelves:
                        new_y = shelves[-1][0] + shelves[-1][2]
                    else:
                        new_y = margin
                    
                    if new_y + padded_h <= usable + margin:
                        shelves.append((new_y, margin + padded_w, padded_h))
                        positions.append((margin, new_y))
                        placed = True
            
            if not placed:
                # Fallback: find any open spot
                x = margin + (idx * 15) % (usable - actual_w)
                y = margin + (idx * 20) % (usable - actual_h)
                positions.append((x, y))
        
        return positions
    
    def _generate_spiral_positions(self, cx: int, cy: int, grid_size: int, margin: int, offset: int) -> List[Tuple[int, int]]:
        """Generate positions spiraling outward from center."""
        positions = [(cx, cy)]
        x, y = cx, cy
        dx, dy = 1, 0
        steps_in_direction = 1
        steps_taken = 0
        direction_changes = 0
        
        for _ in range(200):  # Generate plenty of positions
            x += dx * 15  # Step size
            y += dy * 15
            steps_taken += 1
            
            # Keep within bounds
            bounded_x = min(max(x, margin), grid_size - margin)
            bounded_y = min(max(y, margin), grid_size - margin)
            positions.append((bounded_x, bounded_y))
            
            if steps_taken >= steps_in_direction:
                steps_taken = 0
                direction_changes += 1
                # Rotate direction 90 degrees
                dx, dy = -dy, dx
                if direction_changes % 2 == 0:
                    steps_in_direction += 1
        
        return positions[offset:] if offset < len(positions) else positions
    
    def _calculate_positions(self, num_components: int, grid_size: int, margin: int) -> List[Tuple[int, int]]:
        """Legacy method - kept for compatibility. Use _calculate_positions_smart instead."""
        positions = []
        usable = grid_size - 2 * margin
        
        cols = int(np.ceil(np.sqrt(num_components)))
        rows = int(np.ceil(num_components / cols))
        
        cell_w = usable // cols
        cell_h = usable // rows
        
        for i in range(num_components):
            row = i // cols
            col = i % cols
            x = margin + col * cell_w + cell_w // 4
            y = margin + row * cell_h + cell_h // 4
            positions.append((x, y))
        
        return positions
    
    def _add_traces(self, copper: np.ndarray, components: list, grid_size: int):
        """Add simplified trace routing between components."""
        if len(components) < 2:
            return
        
        # Connect components with Manhattan routing
        for i in range(len(components) - 1):
            c1 = components[i]
            c2 = components[i + 1]
            
            x1, y1 = c1['position']
            w1, h1 = c1['size']
            x2, y2 = c2['position']
            
            # Center points
            cx1, cy1 = x1 + w1//2, y1 + h1//2
            cx2, cy2 = x2 + c2['size'][0]//2, y2 + c2['size'][1]//2
            
            # Horizontal then vertical
            trace_width = 2
            
            # Horizontal segment
            y_trace = cy1
            x_start, x_end = min(cx1, cx2), max(cx1, cx2)
            copper[y_trace:y_trace+trace_width, x_start:x_end] = np.maximum(
                copper[y_trace:y_trace+trace_width, x_start:x_end], 0.6
            )
            
            # Vertical segment
            x_trace = cx2
            y_start, y_end = min(cy1, cy2), max(cy1, cy2)
            copper[y_start:y_end, x_trace:x_trace+trace_width] = np.maximum(
                copper[y_start:y_end, x_trace:x_trace+trace_width], 0.6
            )


def generate_from_description(description: str, api_key: Optional[str] = None):
    """Convenience function to generate PCB from description."""
    generator = AIPCBGenerator(api_key)
    spec = generator.parse_description(description)
    layout = generator.generate_layout(spec)
    return layout, spec


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    descriptions = [
        "Arduino Uno clone with ATmega328P, 16MHz crystal, USB-B connector, 5V LDO regulator, and 4 status LEDs",
        "Motor driver board with 4 MOSFETs, H-bridge driver IC, and large inductor for PWM filtering",
        "IoT sensor node with ESP32, BME280 sensor, 3.3V regulator, and WiFi antenna",
    ]
    
    generator = AIPCBGenerator()
    
    for desc in descriptions:
        print(f"\n{'='*60}")
        print(f"Description: {desc}")
        print('='*60)
        
        spec = generator.parse_description(desc)
        
        print(f"Board: {spec.name}")
        print(f"Type: {spec.board_type}")
        print(f"Total Power: {spec.total_power_w:.2f}W")
        print(f"Components:")
        for c in spec.components:
            print(f"  - {c.count}x {c.name} ({c.type.value}): {c.power_mw}mW")
        
        if spec.thermal_notes:
            print(f"Thermal Notes:")
            for note in spec.thermal_notes:
                print(f"  - {note}")
