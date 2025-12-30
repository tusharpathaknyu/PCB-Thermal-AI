"""
PCB Thermal AI - Interactive Web Demo

A Streamlit app for real-time PCB thermal prediction.
Shows ML predictions, hotspot detection, and design recommendations.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generation import PCBGenerator, ThermalSolver
from src.inference import ThermalPredictor

# Page config
st.set_page_config(
    page_title="PCB Thermal AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


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


def create_heatmap(data, title, colorscale='Jet', showscale=True):
    """Create a Plotly heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        showscale=showscale,
        hoverongaps=False
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor='x'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    return fig


def identify_hotspots(temp_field, threshold_percentile=95):
    """Identify thermal hotspots in the temperature field."""
    threshold = np.percentile(temp_field, threshold_percentile)
    hotspots = []
    
    # Find connected regions above threshold
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
    
    # Sort by temperature
    hotspots.sort(key=lambda x: x['temperature'], reverse=True)
    return hotspots[:5]  # Top 5 hotspots


def generate_recommendations(temp_field, layout, hotspots):
    """Generate thermal design recommendations."""
    recommendations = []
    
    max_temp = temp_field.max()
    mean_temp = temp_field.mean()
    
    # Temperature-based recommendations
    if max_temp > 100:
        recommendations.append({
            'priority': 'High',
            'category': 'Cooling',
            'suggestion': f'Peak temperature ({max_temp:.1f}°C) exceeds 100°C. Consider adding a heatsink or improving airflow.',
            'icon': '🔴'
        })
    
    if max_temp > 85:
        recommendations.append({
            'priority': 'Medium',
            'category': 'Thermal Vias',
            'suggestion': 'Add thermal vias near hotspots to conduct heat to inner/bottom layers.',
            'icon': '🟡'
        })
    
    # Analyze hotspot locations
    for i, hs in enumerate(hotspots[:3]):
        y, x = hs['location']
        
        # Check if copper is sparse near hotspot
        region = layout.copper_density[max(0,y-10):min(128,y+10), max(0,x-10):min(128,x+10)]
        if region.mean() < 0.5:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Copper Pour',
                'suggestion': f'Hotspot #{i+1}: Increase copper pour near ({x}, {y}) to improve heat spreading.',
                'icon': '🟠'
            })
    
    # General recommendations
    if layout.via_map.sum() < 20:
        recommendations.append({
            'priority': 'Low',
            'category': 'Via Placement',
            'suggestion': 'Consider adding more thermal vias (current: {:.0f}) for better vertical heat transfer.'.format(layout.via_map.sum()),
            'icon': '🟢'
        })
    
    if len(recommendations) == 0:
        recommendations.append({
            'priority': 'Info',
            'category': 'Status',
            'suggestion': 'Thermal design looks good! No critical issues detected.',
            'icon': '✅'
        })
    
    return recommendations


def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 PCB Thermal AI Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">ML-powered thermal analysis in seconds, not hours</p>', unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    generator = get_generator()
    
    if predictor is None:
        st.error("❌ No trained model found! Run `python scripts/train.py` first.")
        return
    
    # Sidebar controls
    st.sidebar.header("⚙️ PCB Parameters")
    
    total_power = st.sidebar.slider(
        "Total Power (W)",
        min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="Total power dissipation across all components"
    )
    
    num_components = st.sidebar.slider(
        "Number of Components",
        min_value=1, max_value=15, value=5,
        help="Number of heat-generating components"
    )
    
    complexity = st.sidebar.selectbox(
        "Layout Complexity",
        options=["simple", "medium", "complex"],
        index=1,
        help="Affects copper density and via patterns"
    )
    
    copper_fill = st.sidebar.slider(
        "Copper Fill (%)",
        min_value=20, max_value=80, value=50,
        help="Target copper coverage percentage"
    )
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🎲 Generate New PCB", use_container_width=True):
        st.session_state.regenerate = True
    
    show_ground_truth = st.sidebar.checkbox("Show Ground Truth (slower)", value=False)
    show_recommendations = st.sidebar.checkbox("Show Recommendations", value=True)
    
    # Generate PCB
    if 'layout' not in st.session_state or st.session_state.get('regenerate', True):
        with st.spinner("Generating PCB layout..."):
            layout = generator.generate(
                complexity=complexity,
                num_components=num_components,
                total_power=total_power
            )
            # Adjust copper fill
            layout.copper_density = layout.copper_density * (copper_fill/100 / max(layout.copper_density.mean(), 0.01))
            layout.copper_density = np.clip(layout.copper_density, 0, 1)
            
            st.session_state.layout = layout
            st.session_state.regenerate = False
    
    layout = st.session_state.layout
    
    # Run prediction
    with st.spinner("Running ML prediction..."):
        t_start = time.time()
        result = predictor.predict(
            copper=layout.copper_density,
            vias=layout.via_map,
            components=layout.component_map,
            power=layout.power_map / 1000,
            return_dict=True
        )
        ml_time = time.time() - t_start
    
    # Identify hotspots
    hotspots = identify_hotspots(result['temperature'])
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Max Temperature", f"{result['max_temp']:.1f}°C")
    with col2:
        st.metric("📊 Mean Temperature", f"{result['mean_temp']:.1f}°C")
    with col3:
        st.metric("⚡ Inference Time", f"{ml_time*1000:.0f}ms")
    with col4:
        st.metric("🔥 Hotspots Found", len(hotspots))
    
    st.markdown("---")
    
    # Main visualization
    tab1, tab2, tab3 = st.tabs(["🌡️ Thermal Analysis", "📋 PCB Layout", "📊 Analysis Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Field (ML Prediction)")
            fig = create_heatmap(result['temperature'], "Temperature (°C)", 'Jet')
            
            # Add hotspot markers
            for i, hs in enumerate(hotspots):
                y, x = hs['location']
                fig.add_annotation(
                    x=x, y=y,
                    text=f"#{i+1}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='white',
                    font=dict(color='white', size=10),
                    bgcolor='red' if hs['severity'] == 'Critical' else 'orange',
                    bordercolor='white'
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if show_ground_truth:
                st.subheader("Ground Truth (FEA Simulation)")
                with st.spinner("Running FEA simulation..."):
                    t_start = time.time()
                    solver = ThermalSolver(grid_size=(128, 128))
                    gt_temp = solver.solve(layout)
                    fea_time = time.time() - t_start
                
                fig = create_heatmap(gt_temp, f"FEA Result ({fea_time*1000:.0f}ms)", 'Jet')
                st.plotly_chart(fig, use_container_width=True)
                
                # Error analysis
                mae = np.abs(result['temperature'] - gt_temp).mean()
                st.info(f"📏 **Prediction Error:** MAE = {mae:.2f}°C | Speedup: {fea_time/ml_time:.0f}x faster")
            else:
                st.subheader("Power Dissipation Map")
                fig = create_heatmap(layout.power_map, "Power (W/m²)", 'Hot')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = create_heatmap(layout.copper_density, "Copper Layer", 'YlOrBr')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_heatmap(layout.component_map, "Components", 'Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = create_heatmap(layout.via_map, "Thermal Vias", 'Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Hotspot Analysis")
            if hotspots:
                for i, hs in enumerate(hotspots):
                    severity_color = "🔴" if hs['severity'] == 'Critical' else "🟡"
                    st.markdown(f"""
                    **{severity_color} Hotspot #{i+1}**
                    - Location: ({hs['location'][1]}, {hs['location'][0]})
                    - Temperature: {hs['temperature']:.1f}°C
                    - Area: {hs['area']} pixels
                    - Severity: {hs['severity']}
                    """)
            else:
                st.success("No significant hotspots detected!")
        
        with col2:
            st.subheader("📈 Temperature Distribution")
            fig = px.histogram(
                result['temperature'].flatten(),
                nbins=50,
                labels={'value': 'Temperature (°C)', 'count': 'Pixels'},
                title='Temperature Histogram'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    if show_recommendations:
        st.markdown("---")
        st.subheader("💡 Design Recommendations")
        
        recommendations = generate_recommendations(result['temperature'], layout, hotspots)
        
        cols = st.columns(min(len(recommendations), 3))
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;">
                    <h4>{rec['icon']} {rec['category']}</h4>
                    <p style="font-size: 0.9rem;">{rec['suggestion']}</p>
                    <span style="background: {'#ff6b6b' if rec['priority']=='High' else '#ffa500' if rec['priority']=='Medium' else '#4CAF50'}; 
                                 color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                        {rec['priority']} Priority
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: gray; font-size: 0.8rem;">
        PCB Thermal AI | Built with PyTorch & Streamlit | 
        <a href="https://github.com/tusharpathaknyu/PCB-Thermal-AI">GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
