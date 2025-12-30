"""
PCB Thermal AI - Interactive Web Demo v2

Major Features:
- Upload PCB images (PNG/JPG) or Gerber files
- Realistic PCB visualization with specific component types
- Professional thermal analysis with detailed reports
- Export functionality

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generation import PCBGenerator, HeatEquationSolver
from src.inference import ThermalPredictor

# Try to import AI generator (optional)
try:
    from src.ai_generator import AIPCBGenerator
    AI_GENERATOR_AVAILABLE = True
except ImportError:
    AI_GENERATOR_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="PCB Thermal AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# COMPONENT DEFINITIONS
# ============================================================================

class ComponentType(Enum):
    IC = "IC/Microcontroller"
    POWER_IC = "Power IC/Regulator"
    MOSFET = "MOSFET/Transistor"
    RESISTOR = "Resistor"
    CAPACITOR = "Capacitor"
    INDUCTOR = "Inductor"
    LED = "LED"
    CONNECTOR = "Connector"
    CRYSTAL = "Crystal/Oscillator"


@dataclass
class Component:
    """Represents a specific PCB component."""
    type: ComponentType
    x: int
    y: int
    width: int
    height: int
    power_mw: float  # Power dissipation in mW
    name: str = ""
    
    @property
    def thermal_resistance(self) -> float:
        """Typical junction-to-board thermal resistance (°C/W)."""
        resistances = {
            ComponentType.IC: 40,
            ComponentType.POWER_IC: 25,
            ComponentType.MOSFET: 30,
            ComponentType.RESISTOR: 200,
            ComponentType.CAPACITOR: 500,
            ComponentType.INDUCTOR: 50,
            ComponentType.LED: 100,
            ComponentType.CONNECTOR: 1000,
            ComponentType.CRYSTAL: 300,
        }
        return resistances.get(self.type, 100)
    
    @property
    def color(self) -> str:
        """Component color for visualization."""
        colors = {
            ComponentType.IC: "#1a1a1a",
            ComponentType.POWER_IC: "#2d2d2d",
            ComponentType.MOSFET: "#3d3d3d",
            ComponentType.RESISTOR: "#4a3728",
            ComponentType.CAPACITOR: "#8B4513",
            ComponentType.INDUCTOR: "#2F4F4F",
            ComponentType.LED: "#00ff00",
            ComponentType.CONNECTOR: "#C0C0C0",
            ComponentType.CRYSTAL: "#708090",
        }
        return colors.get(self.type, "#333333")


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .component-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
        background: #1a1a2e;
        border-radius: 8px;
        margin-top: 10px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 0.8rem;
    }
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL & GENERATOR LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    checkpoint_path = Path("checkpoints/best.pth")
    if checkpoint_path.exists():
        return ThermalPredictor.load(str(checkpoint_path))
    return None


@st.cache_resource
def get_generator():
    """Get PCB generator (cached)."""
    return PCBGenerator(grid_size=(128, 128))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_realistic_pcb_image(layout, components: List[Component], size=(512, 512)) -> np.ndarray:
    """Create a realistic PCB visualization with labeled components."""
    h, w = size
    
    # Create base image (FR4 green)
    img = Image.new('RGB', (w, h), color=(20, 80, 20))
    draw = ImageDraw.Draw(img)
    
    # Scale factors
    scale_x = w / 128
    scale_y = h / 128
    
    # Draw copper traces
    copper = layout.copper_density
    for y in range(128):
        for x in range(128):
            if copper[y, x] > 0.3:
                intensity = min(255, int(180 + 75 * copper[y, x]))
                color = (intensity, int(intensity * 0.65), 30)
                x1, y1 = int(x * scale_x), int(y * scale_y)
                x2, y2 = int((x + 1) * scale_x), int((y + 1) * scale_y)
                draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Draw vias
    via_map = layout.via_map
    for y in range(128):
        for x in range(128):
            if via_map[y, x] > 0:
                cx = int((x + 0.5) * scale_x)
                cy = int((y + 0.5) * scale_y)
                r = int(scale_x * 1.5)
                # Via hole (dark center, silver ring)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(180, 180, 190), outline=(100, 100, 100))
                draw.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill=(40, 40, 40))
    
    # Draw components with labels
    for i, comp in enumerate(components):
        x1 = int(comp.x * scale_x)
        y1 = int(comp.y * scale_y)
        x2 = int((comp.x + comp.width) * scale_x)
        y2 = int((comp.y + comp.height) * scale_y)
        
        # Component body
        color = tuple(int(comp.color[i:i+2], 16) for i in (1, 3, 5))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(100, 100, 100))
        
        # Component markings based on type
        if comp.type == ComponentType.IC:
            # IC dot (pin 1 marker)
            draw.ellipse([x1+3, y1+3, x1+8, y1+8], fill=(200, 200, 200))
            # IC text
            draw.rectangle([x1+2, y1+10, x2-2, y2-10], fill=(40, 40, 40))
        elif comp.type == ComponentType.RESISTOR:
            # Resistor bands
            band_w = (x2 - x1) // 6
            for j, c in enumerate([(150, 50, 50), (100, 100, 100), (200, 150, 50)]):
                bx = x1 + band_w * (j + 1)
                draw.rectangle([bx, y1, bx + band_w//2, y2], fill=c)
        elif comp.type == ComponentType.CAPACITOR:
            # Capacitor marking
            draw.line([(x1 + (x2-x1)//3, y1), (x1 + (x2-x1)//3, y2)], fill=(200, 200, 200), width=2)
        elif comp.type == ComponentType.LED:
            # LED glow effect
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = min(x2-x1, y2-y1) // 3
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 255, 0))
        
        # Power indicator (red tint for high power)
        if comp.power_mw > 500:
            overlay = Image.new('RGBA', (x2-x1, y2-y1), (255, 0, 0, 50))
            img.paste(Image.alpha_composite(
                img.crop([x1, y1, x2, y2]).convert('RGBA'), overlay
            ).convert('RGB'), (x1, y1))
    
    return np.array(img)


def create_thermal_visualization(temp_field, hotspots=None, title="Temperature Distribution"):
    """Create a professional thermal visualization."""
    fig = go.Figure()
    
    # Thermal heatmap with FLIR-style colorscale
    fig.add_trace(go.Heatmap(
        z=temp_field,
        colorscale=[
            [0.0, '#000033'],   # Dark blue (coldest)
            [0.2, '#0000ff'],   # Blue
            [0.35, '#00ffff'],  # Cyan
            [0.5, '#00ff00'],   # Green
            [0.65, '#ffff00'],  # Yellow
            [0.8, '#ff8800'],   # Orange
            [0.9, '#ff0000'],   # Red
            [1.0, '#ffffff'],   # White (hottest)
        ],
        colorbar=dict(
            title=dict(text='Temperature (°C)', side='right'),
            tickformat='.0f',
            thickness=15,
            len=0.9
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Temp: %{z:.1f}°C<extra></extra>',
        zsmooth='best'
    ))
    
    # Add hotspot markers with crosshairs
    if hotspots:
        for i, hs in enumerate(hotspots[:5]):
            y, x = hs['location']
            
            # Crosshair lines
            fig.add_shape(type="line", x0=x-8, x1=x+8, y0=y, y1=y,
                         line=dict(color="white", width=1, dash="dot"))
            fig.add_shape(type="line", x0=x, x1=x, y0=y-8, y1=y+8,
                         line=dict(color="white", width=1, dash="dot"))
            
            # Circle marker
            fig.add_shape(type="circle", x0=x-5, x1=x+5, y0=y-5, y1=y+5,
                         line=dict(color="white", width=2))
            
            # Label
            fig.add_annotation(
                x=x, y=y-12,
                text=f"#{i+1}: {hs['temperature']:.1f}°C",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.7)",
                borderpad=2
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color='white')),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x'),
        margin=dict(l=10, r=60, t=50, b=10),
        height=450,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )
    
    return fig


def create_component_power_chart(components: List[Component]):
    """Create a bar chart showing power dissipation by component."""
    if not components:
        return None
    
    # Sort by power
    sorted_comps = sorted(components, key=lambda c: c.power_mw, reverse=True)[:10]
    
    names = [f"{c.name or c.type.value[:10]}" for c in sorted_comps]
    powers = [c.power_mw for c in sorted_comps]
    colors = [c.color for c in sorted_comps]
    
    fig = go.Figure(go.Bar(
        x=powers,
        y=names,
        orientation='h',
        marker_color=['#ff6b6b' if p > 500 else '#ffa500' if p > 200 else '#4CAF50' for p in powers],
        text=[f"{p:.0f} mW" for p in powers],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="Power Dissipation by Component", x=0.5),
        xaxis_title="Power (mW)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white')
    )
    
    return fig


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def identify_hotspots(temp_field, threshold_percentile=95):
    """Identify thermal hotspots."""
    threshold = np.percentile(temp_field, threshold_percentile)
    hotspots = []
    
    from scipy import ndimage
    mask = temp_field > threshold
    labeled, num_features = ndimage.label(mask)
    
    for i in range(1, num_features + 1):
        region = labeled == i
        coords = np.where(region)
        if len(coords[0]) > 0:
            max_idx = temp_field[region].argmax()
            y, x = coords[0][max_idx], coords[1][max_idx]
            hotspots.append({
                'location': (int(y), int(x)),
                'temperature': float(temp_field[y, x]),
                'area': int(region.sum()),
                'severity': 'Critical' if temp_field[y, x] > np.percentile(temp_field, 99) else 'Warning'
            })
    
    hotspots.sort(key=lambda x: x['temperature'], reverse=True)
    return hotspots[:5]


def generate_smart_recommendations(temp_field, layout, hotspots, components: List[Component]):
    """Generate intelligent design recommendations based on analysis."""
    recommendations = []
    max_temp = temp_field.max()
    mean_temp = temp_field.mean()
    
    # Critical temperature warnings
    if max_temp > 125:
        recommendations.append({
            'priority': 'Critical',
            'category': '🚨 Junction Temp',
            'issue': f'Peak temperature ({max_temp:.1f}°C) exceeds typical IC junction limit (125°C)',
            'solution': 'Add heatsink, improve airflow, or reduce power dissipation',
            'icon': '🔴'
        })
    elif max_temp > 100:
        recommendations.append({
            'priority': 'High',
            'category': '⚠️ Overheating',
            'issue': f'Peak temperature ({max_temp:.1f}°C) is dangerously high',
            'solution': 'Consider adding thermal vias or copper pours near hotspots',
            'icon': '🟠'
        })
    
    # High-power component recommendations
    high_power_comps = [c for c in components if c.power_mw > 500]
    for comp in high_power_comps[:2]:
        recommendations.append({
            'priority': 'Medium',
            'category': f'🔌 {comp.type.value}',
            'issue': f'{comp.name or comp.type.value} dissipates {comp.power_mw:.0f}mW',
            'solution': f'Add thermal pad or via array under this component',
            'icon': '🟡'
        })
    
    # Thermal via recommendations
    via_count = layout.via_map.sum()
    if via_count < 15:
        recommendations.append({
            'priority': 'Medium',
            'category': '🔘 Thermal Vias',
            'issue': f'Only {via_count:.0f} vias detected - limited vertical heat transfer',
            'solution': 'Add via arrays (5x5 or 7x7) under high-power components',
            'icon': '🟡'
        })
    
    # Copper pour analysis
    copper_coverage = layout.copper_density.mean()
    if copper_coverage < 0.4:
        recommendations.append({
            'priority': 'Low',
            'category': '🟫 Copper Pour',
            'issue': f'Copper coverage ({copper_coverage*100:.0f}%) is below optimal (>50%)',
            'solution': 'Add copper pours on unused areas connected to ground',
            'icon': '🟢'
        })
    
    # Good design recognition
    if max_temp < 70 and len(recommendations) == 0:
        recommendations.append({
            'priority': 'Good',
            'category': '✅ Thermal Design',
            'issue': 'No critical thermal issues detected',
            'solution': 'Design meets thermal requirements. Consider margin for production variation.',
            'icon': '✅'
        })
    
    return recommendations


def generate_components_from_layout(layout, total_power: float) -> List[Component]:
    """Generate component list from layout analysis."""
    components = []
    comp_map = layout.component_map
    power_map = layout.power_map
    
    from scipy import ndimage
    labeled, num_features = ndimage.label(comp_map > 0)
    
    component_types = [
        ComponentType.IC, ComponentType.POWER_IC, ComponentType.MOSFET,
        ComponentType.RESISTOR, ComponentType.CAPACITOR, ComponentType.INDUCTOR
    ]
    
    for i in range(1, num_features + 1):
        region = labeled == i
        coords = np.where(region)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            area = region.sum()
            power = power_map[region].sum()
            
            # Determine component type based on size and power
            if area > 200 and power > 300:
                comp_type = ComponentType.POWER_IC
            elif area > 100:
                comp_type = ComponentType.IC
            elif width > height * 2:
                comp_type = ComponentType.RESISTOR
            elif height > width * 2:
                comp_type = ComponentType.CAPACITOR
            elif power > 100:
                comp_type = ComponentType.MOSFET
            else:
                comp_type = component_types[i % len(component_types)]
            
            name = f"{comp_type.value.split('/')[0]}_{i}"
            
            components.append(Component(
                type=comp_type,
                x=int(x_min),
                y=int(y_min),
                width=int(width),
                height=int(height),
                power_mw=float(power),
                name=name
            ))
    
    return components


def process_uploaded_image(uploaded_file, power_estimate: float):
    """Process uploaded PCB image with better feature extraction."""
    img = Image.open(uploaded_file).convert('RGB')
    original_size = img.size
    img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    
    # Better feature extraction
    gray = np.mean(img_array, axis=2)
    
    # Copper detection (look for copper-colored regions)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    copper_mask = (r > 0.5) & (g > 0.3) & (g < 0.7) & (b < 0.4)  # Copper color
    green_mask = (g > r) & (g > b)  # FR4 green
    
    copper_density = np.where(copper_mask, 0.8, 0.2)
    copper_density = np.where(green_mask, copper_density * 0.5, copper_density)
    
    # Component detection (dark regions)
    dark_mask = gray < 0.3
    from scipy import ndimage
    
    # Clean up the mask
    dark_mask = ndimage.binary_opening(dark_mask, iterations=2)
    dark_mask = ndimage.binary_closing(dark_mask, iterations=2)
    
    labeled, num_features = ndimage.label(dark_mask)
    
    component_map = np.zeros_like(gray)
    power_map = np.zeros_like(gray)
    
    # Count valid components first
    valid_components = 0
    component_sizes = []
    for i in range(1, min(num_features + 1, 15)):
        region = labeled == i
        if region.sum() > 30:
            valid_components += 1
            component_sizes.append((i, region.sum()))
    
    # Distribute power based on component size (larger = more power)
    total_area = sum(s for _, s in component_sizes)
    
    for comp_id, area in component_sizes:
        region = labeled == comp_id
        component_map[region] = 1
        # Power is distributed per-pixel, proportional to component size
        # Total power for this component = (area / total_area) * total_power
        comp_power = (area / max(total_area, 1)) * power_estimate * 1000
        # Distribute evenly across pixels
        power_map[region] = comp_power / area
    
    # Via detection (small bright spots)
    bright = gray > 0.85
    via_map = ndimage.binary_erosion(bright, iterations=1).astype(float)
    
    # Create layout object
    class UploadedLayout:
        pass
    
    layout = UploadedLayout()
    layout.copper_density = copper_density.astype(np.float32)
    layout.via_map = via_map.astype(np.float32)
    layout.component_map = component_map.astype(np.float32)
    layout.power_map = power_map.astype(np.float32)
    
    return layout, np.array(img.resize((512, 512), Image.Resampling.LANCZOS))


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 PCB Thermal AI Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Thermal Analysis • Seconds Not Hours</p>', unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    generator = get_generator()
    
    if predictor is None:
        st.error("❌ No trained model found! Run `python scripts/train.py` first.")
        st.info("Quick start: `python scripts/train.py --epochs 10` for a quick test model")
        return
    
    # ========== SIDEBAR ==========
    st.sidebar.markdown("## 📥 Input Source")
    
    input_options = ["🎲 Generate Test PCB", "📤 Upload Your PCB"]
    if AI_GENERATOR_AVAILABLE:
        input_options.insert(0, "🤖 AI: Describe Your PCB")
    
    input_method = st.sidebar.radio(
        "Select input method:",
        input_options,
        help="Use AI to describe your PCB, generate a test, or upload your design"
    )
    
    st.sidebar.markdown("---")
    
    # Input handling
    if input_method == "🤖 AI: Describe Your PCB" and AI_GENERATOR_AVAILABLE:
        st.sidebar.markdown("### 🤖 Describe Your PCB")
        st.sidebar.markdown("*Use natural language to describe your board*")
        
        # Example descriptions
        example_descs = [
            "Arduino board with ATmega328P, 7805 voltage regulator, 6 LEDs, and USB connector",
            "ESP32 IoT sensor with WiFi, 3 temperature sensors, LDO regulator",
            "Motor driver with 4 MOSFETs, H-bridge controller, and large heatsink area",
            "Raspberry Pi HAT with power supply, GPIO expander, and I2C sensors"
        ]
        
        # Quick template buttons
        st.sidebar.markdown("**Quick templates:**")
        template_cols = st.sidebar.columns(2)
        
        selected_template = None
        with template_cols[0]:
            if st.button("🔧 Arduino", use_container_width=True, key="tmpl_arduino"):
                selected_template = example_descs[0]
            if st.button("📡 IoT Sensor", use_container_width=True, key="tmpl_iot"):
                selected_template = example_descs[1]
        with template_cols[1]:
            if st.button("⚡ Motor Driver", use_container_width=True, key="tmpl_motor"):
                selected_template = example_descs[2]
            if st.button("🍓 Pi HAT", use_container_width=True, key="tmpl_pi"):
                selected_template = example_descs[3]
        
        # Handle template selection
        if selected_template:
            st.session_state.ai_description = selected_template
        
        # Main text input
        description = st.sidebar.text_area(
            "📝 PCB Description",
            value=st.session_state.get('ai_description', ''),
            height=120,
            placeholder="Example: Arduino board with ATmega328P microcontroller, 7805 voltage regulator, 6 red LEDs, USB connector, and 16MHz crystal",
            help="Describe your PCB in plain English. Include component names, quantities, and types."
        )
        
        # Advanced options
        with st.sidebar.expander("⚙️ Advanced Options"):
            board_size = st.slider("Board Size (mm)", 30, 150, 80)
            has_ground_plane = st.checkbox("Ground Plane", value=True)
            layers = st.selectbox("Layers", [2, 4], index=0)
        
        # Generate button
        generate_ai = st.sidebar.button(
            "🚀 Generate from Description", 
            use_container_width=True, 
            type="primary",
            disabled=len(description.strip()) < 10
        )
        
        if generate_ai and description.strip():
            with st.spinner("🤖 AI analyzing your description..."):
                ai_gen = AIPCBGenerator()
                spec = ai_gen.parse_description(description)
                
                # Store spec for display
                st.session_state.ai_spec = spec
                
                # Generate actual layout (returns a layout object)
                layout = ai_gen.generate_layout(spec, grid_size=128)
                
                st.session_state.layout = layout
                st.session_state.input_type = 'ai_generated'
                st.session_state.ai_components = spec.components
                st.session_state.ai_total_power = spec.total_power_w
                
            st.sidebar.success(f"✅ Generated: {spec.name}")
            
        elif 'ai_spec' in st.session_state:
            # Show parsed spec
            spec = st.session_state.ai_spec
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**📋 Parsed Board:** {spec.name}")
            st.sidebar.markdown(f"**🔌 Type:** {spec.board_type}")
            st.sidebar.markdown(f"**⚡ Power:** {spec.total_power_w:.2f}W")
            
            # Component list
            with st.sidebar.expander(f"🧩 Components ({len(spec.components)})"):
                for comp in spec.components:
                    st.markdown(f"• {comp.count}x **{comp.name}** - {comp.power_mw:.0f}mW")
        
        if 'layout' not in st.session_state or st.session_state.get('input_type') not in ['ai_generated']:
            # Show help text
            st.markdown("### 🤖 AI-Powered PCB Generation")
            st.markdown("""
            **Describe your PCB in natural language and our AI will:**
            1. 🔍 Parse your description to identify components
            2. ⚡ Estimate realistic power consumption
            3. 📐 Generate an intelligent component layout
            4. 🌡️ Run thermal analysis instantly
            
            **Example descriptions:**
            - *"Arduino Uno clone with ATmega328P, 7805 regulator, 6 LEDs, USB-B connector"*
            - *"ESP32 weather station with BME280 sensor, OLED display, LiPo charger"*
            - *"12V motor driver with 4 IRF540 MOSFETs and bootstrap capacitors"*
            
            **Tips for best results:**
            - Include specific component names (ATmega328P, ESP32, LM7805, etc.)
            - Mention quantities ("6 LEDs", "4 MOSFETs")
            - Describe high-power components for accurate thermal analysis
            """)
            return
    
    elif input_method == "📤 Upload Your PCB":
        st.sidebar.markdown("### 📤 Upload PCB Image")
        
        uploaded_file = st.sidebar.file_uploader(
            "Drag & drop or click to upload",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a top-down photo or render of your PCB"
        )
        
        if uploaded_file:
            st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
            
            power_estimate = st.sidebar.number_input(
                "Total Power Dissipation (W)",
                min_value=0.1, max_value=50.0, value=3.0, step=0.5,
                help="Estimated total power dissipation of all components"
            )
            
            with st.spinner("Processing uploaded image..."):
                layout, original_img = process_uploaded_image(uploaded_file, power_estimate)
                st.session_state.layout = layout
                st.session_state.original_img = original_img
                st.session_state.input_type = 'uploaded'
        else:
            st.sidebar.info("👆 Upload a PCB image to analyze")
            
            # Show example
            st.markdown("### 📸 How to Use")
            st.markdown("""
            **Upload a PCB image to get thermal analysis:**
            1. Take a top-down photo of your PCB
            2. Or export a render from your EDA tool (KiCad, Altium, Eagle)
            3. Specify total power dissipation
            4. Get instant thermal predictions!
            
            **Supported formats:** PNG, JPG, BMP, TIFF
            
            **Best results with:**
            - Clear, well-lit images
            - Top copper layer visible
            - Components clearly visible
            """)
            
            if 'layout' not in st.session_state or st.session_state.get('input_type') == 'uploaded':
                return
    
    else:
        # Synthetic generation
        st.sidebar.markdown("### ⚙️ PCB Configuration")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            total_power = st.number_input("Power (W)", 0.5, 15.0, 3.0, 0.5)
        with col2:
            num_components = st.number_input("Components", 1, 20, 6, 1)
        
        complexity = st.sidebar.select_slider(
            "Design Complexity",
            options=["Simple", "Medium", "Complex"],
            value="Medium"
        )
        
        copper_fill = st.sidebar.slider("Copper Fill %", 20, 80, 50)
        
        if st.sidebar.button("🎲 Generate New PCB", use_container_width=True, type="primary"):
            st.session_state.regenerate = True
        
        # Generate
        if 'layout' not in st.session_state or st.session_state.get('regenerate', True) or st.session_state.get('input_type') in ['uploaded', 'ai_generated']:
            with st.spinner("Generating PCB..."):
                layout = generator.generate(
                    complexity=complexity.lower(),
                    num_components=int(num_components),
                    total_power=total_power
                )
                scale = copper_fill / 100 / max(layout.copper_density.mean(), 0.01)
                layout.copper_density = np.clip(layout.copper_density * scale, 0, 1)
                
                st.session_state.layout = layout
                st.session_state.regenerate = False
                st.session_state.input_type = 'synthetic'
    
    layout = st.session_state.layout
    
    # Analysis options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analysis Options")
    show_comparison = st.sidebar.checkbox("Compare with FEA", value=False, help="Run ground truth simulation (slower)")
    show_layers = st.sidebar.checkbox("Show Layer Breakdown", value=False)
    
    # ========== MAIN ANALYSIS ==========
    
    # Generate components
    components = generate_components_from_layout(layout, layout.power_map.sum() / 1000)
    
    # Run prediction
    with st.spinner("🧠 Running neural network inference..."):
        t_start = time.time()
        result = predictor.predict(
            copper=layout.copper_density,
            vias=layout.via_map,
            components=layout.component_map,
            power=layout.power_map / 1000,
            return_dict=True
        )
        inference_time = time.time() - t_start
    
    # Analyze results
    hotspots = identify_hotspots(result['temperature'])
    temp_field = result['temperature']
    
    # ========== METRICS ROW ==========
    st.markdown("### 📈 Thermal Summary")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        delta_color = "inverse" if temp_field.max() > 100 else "normal"
        st.metric("🌡️ Peak Temp", f"{temp_field.max():.1f}°C", 
                 f"+{temp_field.max()-25:.0f}°C", delta_color=delta_color)
    with m2:
        st.metric("📊 Average", f"{temp_field.mean():.1f}°C")
    with m3:
        st.metric("📉 Min Temp", f"{temp_field.min():.1f}°C")
    with m4:
        st.metric("⚡ Inference", f"{inference_time*1000:.1f}ms")
    with m5:
        status = "🔴 Critical" if temp_field.max() > 100 else "🟡 Warning" if temp_field.max() > 85 else "🟢 OK"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # ========== MAIN VISUALIZATIONS ==========
    col_pcb, col_thermal = st.columns(2)
    
    with col_pcb:
        st.markdown("#### 🖥️ PCB Layout")
        
        if st.session_state.get('input_type') == 'uploaded' and 'original_img' in st.session_state:
            fig = go.Figure()
            fig.add_trace(go.Image(z=st.session_state.original_img))
            fig.update_layout(
                title=dict(text="Your Uploaded PCB", x=0.5),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.get('input_type') == 'ai_generated' and 'ai_spec' in st.session_state:
            pcb_img = create_realistic_pcb_image(layout, components)
            spec = st.session_state.ai_spec
            fig = go.Figure()
            fig.add_trace(go.Image(z=pcb_img))
            fig.update_layout(
                title=dict(text=f"🤖 AI Generated: {spec.name} ({spec.board_type})", x=0.5, font=dict(size=14)),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            pcb_img = create_realistic_pcb_image(layout, components)
            fig = go.Figure()
            fig.add_trace(go.Image(z=pcb_img))
            fig.update_layout(
                title=dict(text="PCB Layout Visualization", x=0.5),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Component legend
        st.markdown("""
        <div style="background:#1a1a2e; padding:10px; border-radius:8px; font-size:0.85rem;">
        <b>Components:</b>
        <span style="color:#FFD700">█</span> Copper
        <span style="color:#C0C0C0">●</span> Vias
        <span style="color:#1a1a1a">█</span> ICs
        <span style="color:#4a3728">█</span> Resistors
        <span style="color:#8B4513">█</span> Capacitors
        <span style="color:#00ff00">●</span> LEDs
        </div>
        """, unsafe_allow_html=True)
    
    with col_thermal:
        st.markdown("#### 🌡️ Thermal Analysis")
        fig = create_thermal_visualization(temp_field, hotspots, f"Temperature Distribution ({inference_time*1000:.0f}ms)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background:#1a1a2e; padding:10px; border-radius:8px; font-size:0.85rem;">
        <b>Scale:</b>
        <span style="color:#000066">█</span> Cold →
        <span style="color:#0000ff">█</span>
        <span style="color:#00ffff">█</span>
        <span style="color:#00ff00">█</span>
        <span style="color:#ffff00">█</span>
        <span style="color:#ff8800">█</span>
        <span style="color:#ff0000">█</span>
        <span style="color:#ffffff">█</span> Hot
        &nbsp;|&nbsp; <b>⭕</b> Hotspot markers
        </div>
        """, unsafe_allow_html=True)
    
    # ========== HOTSPOTS & COMPONENTS ==========
    st.markdown("---")
    
    col_hot, col_comp = st.columns(2)
    
    with col_hot:
        st.markdown("#### 🎯 Detected Hotspots")
        if hotspots:
            for i, hs in enumerate(hotspots):
                severity_color = "#ff4444" if hs['severity'] == 'Critical' else "#ffaa00"
                st.markdown(f"""
                <div style="background:#1a1a2e; border-left:4px solid {severity_color}; padding:10px; margin:5px 0; border-radius:0 8px 8px 0;">
                <b>#{i+1}</b> • <b>{hs['temperature']:.1f}°C</b> at ({hs['location'][1]}, {hs['location'][0]})
                <span style="float:right; color:{severity_color}">{hs['severity']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col_comp:
        st.markdown("#### 🔌 Component Power Analysis")
        if components:
            fig = create_component_power_chart(components)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== AI ANALYSIS (if AI generated) ==========
    if st.session_state.get('input_type') == 'ai_generated' and 'ai_spec' in st.session_state:
        st.markdown("---")
        st.markdown("#### 🤖 AI-Parsed Component Breakdown")
        
        spec = st.session_state.ai_spec
        ai_cols = st.columns([2, 1, 1])
        
        with ai_cols[0]:
            # Component table
            comp_data = []
            for comp in spec.components:
                comp_data.append({
                    "Component": comp.name,
                    "Type": comp.type.name.replace("_", " ").title(),
                    "Qty": comp.count,
                    "Power (mW)": f"{comp.power_mw:.0f}",
                    "Package": comp.package or "—"
                })
            
            if comp_data:
                st.dataframe(comp_data, use_container_width=True, hide_index=True)
        
        with ai_cols[1]:
            st.markdown("**Board Specs**")
            st.markdown(f"- **Name:** {spec.name}")
            st.markdown(f"- **Type:** {spec.board_type.replace('_', ' ').title()}")
            st.markdown(f"- **Complexity:** {spec.estimated_complexity}")
            st.markdown(f"- **Layers:** {spec.layers}")
            st.markdown(f"- **Size:** {spec.size_mm[0]}×{spec.size_mm[1]} mm")
        
        with ai_cols[2]:
            st.markdown("**Power Summary**")
            total_from_components = sum(c.power_mw for c in spec.components)
            st.metric("Total Power", f"{total_from_components/1000:.2f} W")
            st.metric("Components", len(spec.components))
            
            if spec.thermal_notes:
                st.markdown("**⚠️ AI Notes:**")
                for note in spec.thermal_notes[:3]:
                    st.markdown(f"- {note}")
    
    # ========== RECOMMENDATIONS ==========
    st.markdown("---")
    st.markdown("#### 💡 Design Recommendations")
    
    recommendations = generate_smart_recommendations(temp_field, layout, hotspots, components)
    
    rec_cols = st.columns(min(len(recommendations), 3))
    for i, rec in enumerate(recommendations):
        with rec_cols[i % 3]:
            bg_color = "#3d1a1a" if rec['priority'] == 'Critical' else "#3d2a1a" if rec['priority'] == 'High' else "#1a3d1a" if rec['priority'] == 'Good' else "#1a1a3d"
            border_color = "#ff4444" if rec['priority'] == 'Critical' else "#ff8800" if rec['priority'] == 'High' else "#44ff44" if rec['priority'] == 'Good' else "#4488ff"
            
            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border_color}; padding:15px; border-radius:10px; min-height:160px;">
            <h4 style="color:white; margin:0 0 10px 0;">{rec['category']}</h4>
            <p style="color:#ccc; font-size:0.9rem; margin:0 0 10px 0;"><b>Issue:</b> {rec['issue']}</p>
            <p style="color:#88ff88; font-size:0.9rem; margin:0;"><b>Fix:</b> {rec['solution']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== FEA COMPARISON ==========
    if show_comparison:
        st.markdown("---")
        st.markdown("#### ⚖️ ML vs FEA Comparison")
        
        with st.spinner("Running FEA simulation (what we're replacing)..."):
            t_fea = time.time()
            solver = HeatEquationSolver(grid_size=(128, 128))
            gt_temp = solver.solve(layout)
            fea_time = time.time() - t_fea
        
        comp_cols = st.columns(3)
        
        with comp_cols[0]:
            fig = create_thermal_visualization(temp_field, title=f"ML ({inference_time*1000:.0f}ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        with comp_cols[1]:
            fig = create_thermal_visualization(gt_temp, title=f"FEA ({fea_time*1000:.0f}ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        with comp_cols[2]:
            error = np.abs(temp_field - gt_temp)
            fig = go.Figure(go.Heatmap(z=error, colorscale='Reds', colorbar=dict(title='°C')))
            fig.update_layout(
                title=dict(text=f"Error Map (MAE: {error.mean():.2f}°C)", x=0.5),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False, scaleanchor='x'),
                height=400, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        speedup = fea_time / inference_time
        st.success(f"**ML is {speedup:.0f}x faster!** FEA: {fea_time*1000:.0f}ms → ML: {inference_time*1000:.0f}ms | MAE: {error.mean():.2f}°C")
    
    # ========== LAYER BREAKDOWN ==========
    if show_layers:
        st.markdown("---")
        st.markdown("#### 📚 Layer Analysis")
        
        l1, l2, l3, l4 = st.columns(4)
        
        for col, (data, title, cmap) in zip(
            [l1, l2, l3, l4],
            [(layout.copper_density, "Copper Density", "YlOrBr"),
             (layout.component_map, "Components", "Greys"),
             (layout.power_map, "Power Map (mW)", "Reds"),
             (layout.via_map, "Via Locations", "Blues")]
        ):
            with col:
                fig = go.Figure(go.Heatmap(z=data, colorscale=cmap))
                fig.update_layout(title=dict(text=title, x=0.5), height=250,
                                 xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'),
                                 margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
    <p style="text-align:center; color:#666; font-size:0.8rem;">
    PCB Thermal AI v2.0 • Built with PyTorch & Streamlit • 
    <a href="https://github.com/tusharpathaknyu/PCB-Thermal-AI">GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
