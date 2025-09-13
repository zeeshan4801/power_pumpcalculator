"""
Pump Power Calculator — Gradio app (clean, fixed)
Calculates hydraulic, shaft and motor power for pumps. Supports multiple units,
optionally rounds motor to common standard kW sizes, shows a bar chart and lets
you download results as CSV.
"""

import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# Constants
G = 9.80665  # m/s^2
HP_CONV = 745.699872  # W per mechanical horsepower


def flow_to_m3s(value: float, unit: str) -> float:
    """Convert various flow units to m^3/s"""
    if unit == "m³/s":
        return float(value)
    if unit == "m³/h":
        return float(value) / 3600.0
    if unit == "L/s":
        return float(value) / 1000.0
    if unit == "L/min":
        return float(value) / 60000.0
    if unit == "m³/min":
        return float(value) / 60.0
    if unit == "ft³/s (cfs)":
        return float(value) * 0.028316846592
    if unit == "US gpm":
        return float(value) * 0.003785411784
    raise ValueError(f"Unsupported flow unit: {unit}")


def head_to_m(value: float, unit: str) -> float:
    """Convert head to meters"""
    if unit == "m":
        return float(value)
    if unit == "ft":
        return float(value) * 0.3048
    raise ValueError(f"Unsupported head unit: {unit}")


def compute_pump_power(
    flow_value,
    flow_unit,
    head_value,
    head_unit,
    density,
    gravity,
    pump_eff_pct,
    motor_eff_pct,
    service_factor,
    round_motor_to_standard,
):
    """
    Returns: (summary_text, results_dataframe, matplotlib_figure, csv_filepath)
    """

    # Convert inputs
    Q = flow_to_m3s(flow_value, flow_unit)  # m^3/s
    H = head_to_m(head_value, head_unit)    # m

    rho = float(density)
    g = float(gravity)

    # Efficiencies as decimals; protect against zero
    pump_eff = max(float(pump_eff_pct) / 100.0, 1e-6)
    motor_eff = max(float(motor_eff_pct) / 100.0, 1e-6)
    sf = float(service_factor)

    # Hydraulic power (W)
    P_hyd_W = rho * g * Q * H

    # Shaft power (W)
    P_shaft_W = P_hyd_W / pump_eff

    # Motor recommended power (W)
    P_motor_W = (P_shaft_W * sf) / motor_eff

    # Convert to kW and HP
    P_hyd_kW = P_hyd_W / 1000.0
    P_shaft_kW = P_shaft_W / 1000.0
    P_motor_kW = P_motor_W / 1000.0

    P_hyd_HP = P_hyd_W / HP_CONV
    P_shaft_HP = P_shaft_W / HP_CONV
    P_motor_HP = P_motor_W / HP_CONV

    # Optionally round motor to next standard motor rating (kW)
    rounded_kW = None
    if round_motor_to_standard:
        standard_kW = [
            0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 3.7,
            4.0, 4.5, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55,
            75, 90, 110,
        ]
        required = P_motor_kW
        # choose first standard >= required; if none, take last
        rounded_kW = next((s for s in standard_kW if s >= required), standard_kW[-1])

    # Build DataFrame for table output
    rows = [
        {
            "Power type": "Hydraulic",
            "W": P_hyd_W,
            "kW": P_hyd_kW,
            "HP": P_hyd_HP,
        },
        {
            "Power type": "Shaft",
            "W": P_shaft_W,
            "kW": P_shaft_kW,
            "HP": P_shaft_HP,
        },
        {
            "Power type": "Motor (recommended)",
            "W": P_motor_W,
            "kW": P_motor_kW,
            "HP": P_motor_HP,
        },
    ]

    if rounded_kW is not None:
        rows.append(
            {
                "Power type": "Motor (rounded standard)",
                "W": rounded_kW * 1000.0,
                "kW": rounded_kW,
                "HP": (rounded_kW * 1000.0) / HP_CONV,
            }
        )

    df = pd.DataFrame(rows)

    # Create a bar chart (kW)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["Power type"], df["kW"].astype(float))
    ax.set_ylabel("Power (kW)")
    ax.set_title("Pump Power Summary")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Summary text
    summary_lines = [
        f"Flow = {flow_value} {flow_unit} -> {Q:.6e} m³/s",
        f"Head = {head_value} {head_unit} -> {H:.6f} m",
        f"Density = {rho:.2f} kg/m³, g = {g:.5f} m/s²",
        f"Pump eff = {pump_eff_pct}%  |  Motor eff = {motor_eff_pct}%  |  Service factor = {sf}",
        "",
        f"Hydraulic power: {P_hyd_kW:.4f} kW ({P_hyd_HP:.4f} HP)",
        f"Shaft power:      {P_shaft_kW:.4f} kW ({P_shaft_HP:.4f} HP)",
        f"Motor recommended:{P_motor_kW:.4f} kW ({P_motor_HP:.4f} HP)",
    ]
    if rounded_kW is not None:
        summary_lines.append(f"Rounded motor (standard): {rounded_kW:.2f} kW")

    summary = "\n".join(summary_lines)

    # Write CSV to a temporary file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp_path = tmp.name
    tmp.close()

    return summary, df, fig, tmp_path


# --- Build Gradio UI ---
with gr.Blocks(title="Pump Power Calculator") as demo:
    gr.Markdown("## 🔧 Pump Power Calculator (Gradio)\nCalculate hydraulic, shaft and motor power for pumps.")
    with gr.Row():
        with gr.Column(scale=2):
            flow_value = gr.Number(value=50.0, label="Flow value", info="Input flow magnitude")
            flow_unit = gr.Dropdown(
                choices=["m³/s", "m³/h", "L/s", "L/min", "m³/min", "ft³/s (cfs)", "US gpm"],
                value="L/s",
                label="Flow unit",
            )
            head_value = gr.Number(value=20.0, label="Head value")
            head_unit = gr.Dropdown(choices=["m", "ft"], value="m", label="Head unit")
            density = gr.Number(value=1000.0, label="Fluid density (kg/m³)")
            gravity = gr.Number(value=G, label="Gravity (m/s²)")
        with gr.Column(scale=1):
            pump_eff = gr.Slider(1, 100, value=70, step=1, label="Pump efficiency (%)")
            motor_eff = gr.Slider(1, 100, value=90, step=1, label="Motor efficiency (%)")
            service_factor = gr.Number(value=1.1, label="Service factor")
            round_standard = gr.Checkbox(value=True, label="Round motor to standard rating (common kW sizes)")
            compute_btn = gr.Button("Compute")

    summary_out = gr.Textbox(label="Summary", interactive=False)
    table_out = gr.Dataframe(headers=["Power type", "W", "kW", "HP"], label="Power table")
    plot_out = gr.Plot(label="Power chart (kW)")
    download_file = gr.File(label="Download CSV (results table)")

    compute_btn.click(
        fn=compute_pump_power,
        inputs=[flow_value, flow_unit, head_value, head_unit, density, gravity, pump_eff, motor_eff, service_factor, round_standard],
        outputs=[summary_out, table_out, plot_out, download_file],
    )

    gr.Markdown(
        "### Notes\n"
        "- Hydraulic power is theoretical (no losses).\n"
        "- Shaft power includes pump efficiency losses.\n"
        "- Motor recommendation includes service factor and motor efficiency.\n"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
