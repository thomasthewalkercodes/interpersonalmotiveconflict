import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import math
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Interactive Motive Circumplex", page_icon="🎯", layout="wide"
)


def initialize_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.motives = [
            "Warm",
            "Warm Dominant",
            "Dominant",
            "Cold Dominant",
            "Cold",
            "Cold Submissive",
            "Submissive",
            "Warm Submissive",
        ]

        st.session_state.connections = {}
        # Initialize all possible connections to 0
        for i in range(8):  # Fixed number instead of accessing session state
            for j in range(i + 1, 8):
                key = f"{i}-{j}"
                st.session_state.connections[key] = 0.0

        st.session_state.prototypes = {}
        st.session_state.initialized = True


def calculate_positions(motives):
    """Calculate positions of motives on the circumplex"""
    positions = []
    n_motives = len(motives)

    for i in range(n_motives):
        angle = i * (2 * math.pi / n_motives)  # Start from right side (0 degrees)
        x = math.cos(angle)
        y = math.sin(angle)
        positions.append((x, y))

    return positions


def create_circumplex_plot(motives, connections):
    """Create the interactive circumplex visualization"""
    positions = calculate_positions(motives)

    fig = go.Figure()

    # Add connection lines
    for key, strength in connections.items():
        i, j = map(int, key.split("-"))
        x1, y1 = positions[i]
        x2, y2 = positions[j]

        # Determine line properties based on strength
        abs_strength = abs(strength)
        line_width = max(0.5, abs_strength * 8)
        opacity = 0.3 + abs_strength * 0.7

        if strength > 0:
            color = f"rgba(34, 197, 94, {opacity})"  # Green for positive
        elif strength < 0:
            color = f"rgba(239, 68, 68, {opacity})"  # Red for negative
        else:
            color = "rgba(128, 128, 128, 0.2)"  # Gray for zero

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color=color, width=line_width),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add motive points
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                size=40, color="rgb(102, 126, 234)", line=dict(color="white", width=3)
            ),
            text=motives,
            textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    # Add circle background
    circle_x = [math.cos(t) for t in np.linspace(0, 2 * math.pi, 100)]
    circle_y = [math.sin(t) for t in np.linspace(0, 2 * math.pi, 100)]

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="lightgray", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Interactive Motive Circumplex",
            x=0.5,
            font=dict(size=20, color="rgb(51, 51, 51)"),
        ),
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[-1.5, 1.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1.5, 1.5],
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=600,
        height=600,
        margin=dict(l=50, r=50, t=70, b=50),
    )

    return fig


def gaussian_random(mean=0, std=0.3):
    """Generate random number from Gaussian distribution"""
    return np.clip(np.random.normal(mean, std), -1, 1)


def randomize_connections():
    """Randomize all connections using Gaussian distribution"""
    for key in st.session_state.connections:
        st.session_state.connections[key] = gaussian_random()


def reset_connections():
    """Reset all connections to 0"""
    for key in st.session_state.connections:
        st.session_state.connections[key] = 0.0


def save_prototype(name):
    """Save current state as a prototype"""
    if name.strip():
        st.session_state.prototypes[name] = {
            "motives": st.session_state.motives.copy(),
            "connections": st.session_state.connections.copy(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return True
    return False


def load_prototype(name):
    """Load a prototype"""
    if name in st.session_state.prototypes:
        prototype = st.session_state.prototypes[name]
        st.session_state.motives = prototype["motives"].copy()
        st.session_state.connections = prototype["connections"].copy()
        return True
    return False


def export_to_csv(motives, connections):
    """Export connections to CSV format"""
    data = []
    for key, strength in connections.items():
        i, j = map(int, key.split("-"))
        data.append(
            {
                "Motive1": motives[i],
                "Motive2": motives[j],
                "Connection_Strength": strength,
            }
        )

    return pd.DataFrame(data)


def main():
    """Main application function"""
    # Initialize session state first
    initialize_session_state()

    st.title("🎯 Interactive Motive Circumplex")
    st.markdown(
        "Visualize and adjust relationships between different motives in a circumplex model"
    )

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Circumplex Visualization")

        # Display the plot
        fig = create_circumplex_plot(
            st.session_state.motives, st.session_state.connections
        )
        st.plotly_chart(fig, use_container_width=False)

    with col2:
        st.subheader("Controls")

        # Motive name editor
        st.markdown("#### Edit Motive Names")
        with st.expander("Click to edit motive names", expanded=False):
            for i, motive in enumerate(st.session_state.motives):
                new_name = st.text_input(
                    f"Motive {i+1}:", value=motive, key=f"motive_{i}"
                )
                if new_name != motive:
                    st.session_state.motives[i] = new_name

        st.markdown("#### Connection Strengths")

        # Connection controls
        for key, current_strength in st.session_state.connections.items():
            i, j = map(int, key.split("-"))
            label = f"{st.session_state.motives[i]} ↔ {st.session_state.motives[j]}"

            new_strength = st.slider(
                label,
                min_value=-1.0,
                max_value=1.0,
                value=current_strength,
                step=0.1,
                key=f"slider_{key}",
                format="%.1f",
            )

            if new_strength != current_strength:
                st.session_state.connections[key] = new_strength

        st.markdown("#### Actions")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "🎲 Randomize", help="Apply Gaussian distribution (μ=0, σ=0.3)"
            ):
                randomize_connections()
                st.rerun()

        with col_b:
            if st.button("🔄 Reset All", help="Reset all connections to 0"):
                reset_connections()
                st.rerun()

        st.markdown("#### Prototypes")

        # Save prototype
        prototype_name = st.text_input("Prototype name:", placeholder="Enter name...")
        if st.button("💾 Save Current"):
            if save_prototype(prototype_name):
                st.success(f"Prototype '{prototype_name}' saved!")
                st.rerun()
            else:
                st.error("Please enter a valid name")

        # Load prototype
        if st.session_state.prototypes:
            st.markdown("**Saved Prototypes:**")
            for name, prototype in st.session_state.prototypes.items():
                col_load, col_delete = st.columns([3, 1])

                with col_load:
                    if st.button(f"📂 {name}", key=f"load_{name}"):
                        load_prototype(name)
                        st.success(f"Loaded '{name}'!")
                        st.rerun()

                with col_delete:
                    if st.button("🗑️", key=f"delete_{name}", help="Delete prototype"):
                        del st.session_state.prototypes[name]
                        st.rerun()

        st.markdown("#### Export Data")

        if st.button("📊 Generate CSV"):
            df = export_to_csv(st.session_state.motives, st.session_state.connections)

            # Display the data
            st.dataframe(df, use_container_width=True)

            # Create download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"motive_connections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Statistics
        st.markdown("#### Statistics")
        total_connections = len(st.session_state.connections)
        positive_connections = sum(
            1 for v in st.session_state.connections.values() if v > 0
        )
        negative_connections = sum(
            1 for v in st.session_state.connections.values() if v < 0
        )
        zero_connections = (
            total_connections - positive_connections - negative_connections
        )

        st.metric("Total Connections", total_connections)

        col_stats = st.columns(3)
        with col_stats[0]:
            st.metric(
                "Positive",
                positive_connections,
                delta=f"{positive_connections/total_connections*100:.1f}%",
            )
        with col_stats[1]:
            st.metric(
                "Negative",
                negative_connections,
                delta=f"{negative_connections/total_connections*100:.1f}%",
            )
        with col_stats[2]:
            st.metric(
                "Zero",
                zero_connections,
                delta=f"{zero_connections/total_connections*100:.1f}%",
            )


# Run the app
if __name__ == "__main__":
    main()
