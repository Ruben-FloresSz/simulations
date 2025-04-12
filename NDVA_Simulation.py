import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# 1. Load and prepare data
df = pd.read_csv(r"C:\Users\rubof\Downloads\Kaxanuk\S_Data-Curator\Output\NVDA.csv")
df['m_date'] = pd.to_datetime(df['m_date'])
df.sort_values('m_date', inplace=True)
df = df[df['m_adjusted_close'] > 0]

# 2. Compute log returns
df['log_return'] = np.log(df['m_adjusted_close'] / df['m_adjusted_close'].shift(1))
df.dropna(inplace=True)

# 3. Estimate GBM parameters
mu = df['log_return'].mean() * 252
sigma = df['log_return'].std() * np.sqrt(252)
S0 = df['m_adjusted_close'].iloc[-1]

# 4. GBM simulation
n_simulations = 50000
n_days = 252
dt = 1 / n_days
price_paths = np.zeros((n_days, n_simulations))
price_paths[0] = S0

for t in range(1, n_days):
    Z = np.random.standard_normal(n_simulations)
    price_paths[t] = price_paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# 5. Historical data (last 6 months)
cutoff_date = df['m_date'].max() - pd.DateOffset(months=6)
recent_data = df[df['m_date'] >= cutoff_date]
dates_real = recent_data['m_date'].tolist()
dates_sim = pd.bdate_range(start=dates_real[-1], periods=n_days + 1)[1:]
all_dates = dates_real + list(dates_sim)

# 6. Combine real and simulated prices
full_paths = np.full((len(all_dates), n_simulations), np.nan)
full_paths[:len(dates_real), :] = np.tile(recent_data['m_adjusted_close'].values.reshape(-1, 1), n_simulations)
full_paths[len(dates_real):, :] = price_paths

# 7. Analyze final prices
final_prices = price_paths[-1]
expected_price = final_prices.mean()
median_price = np.percentile(final_prices, 50)
p90_price = np.percentile(final_prices, 90)
p10_price = np.percentile(final_prices, 10)

print(f"Target Price (12-month expected): ${expected_price:.2f}")
print(f"Median (P50): ${median_price:.2f}")
print(f"Optimistic Scenario (P90): ${p90_price:.2f}")
print(f"Pessimistic Scenario (P10): ${p10_price:.2f}")

# === INTERACTIVE PLOTLY TRAJECTORIES WITH LEGEND ===
fig = go.Figure()

# Plot 300 sample trajectories
for i in range(300):
    fig.add_trace(go.Scatter(
        x=all_dates,
        y=full_paths[:, i],
        mode='lines',
        line=dict(width=0.7),
        opacity=0.3,
        showlegend=False,
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

# Horizontal lines + simulated legend
fig.add_hline(y=expected_price, line_dash='dash', line_color='blue')
fig.add_hline(y=median_price, line_dash='dash', line_color='green')
fig.add_hline(y=p90_price, line_dash='dash', line_color='orange')
fig.add_hline(y=p10_price, line_dash='dash', line_color='red')

fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', dash='dash'), name=f"Target: ${expected_price:.2f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name=f"Median: ${median_price:.2f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='orange', dash='dash'), name=f"P90: ${p90_price:.2f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name=f"P10: ${p10_price:.2f}"))

fig.update_layout(
    title="NVDA Price Trajectories: Last 6 Months + 12-Month GBM Simulation",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    height=600,
    legend=dict(x=0.75, y=0.95, bordercolor="black", borderwidth=1)
)
fig.write_html("nvda_trajectories.html")
fig.show()

# === INTERACTIVE HISTOGRAM WITH LEGEND ===
hist_fig = px.histogram(
    x=final_prices,
    nbins=100,
    title="Distribution of Simulated NVDA Prices After 12 Months (GBM)",
    labels={'x': 'Estimated Price', 'y': 'Frequency'},
    opacity=0.75,
    color_discrete_sequence=['lightgreen']
)

# Vertical lines + fake traces for legend
hist_fig.add_vline(x=expected_price, line_dash='dash', line_color='blue')
hist_fig.add_vline(x=median_price, line_dash='dash', line_color='green')
hist_fig.add_vline(x=p90_price, line_dash='dash', line_color='orange')
hist_fig.add_vline(x=p10_price, line_dash='dash', line_color='red')

hist_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', dash='dash'), name=f"Target: ${expected_price:.2f}"))
hist_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name=f"Median: ${median_price:.2f}"))
hist_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='orange', dash='dash'), name=f"P90: ${p90_price:.2f}"))
hist_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name=f"P10: ${p10_price:.2f}"))

hist_fig.update_layout(
    template="plotly_white",
    legend=dict(x=0.75, y=0.95, bordercolor="black", borderwidth=1),
)
hist_fig.write_html("nvda_histogram.html")
hist_fig.show()

# === STATIC MATPLOTLIB GRAPHS ===

# Static price trajectories
plt.figure(figsize=(14, 6))
for i in range(300):
    plt.plot(all_dates, full_paths[:, i], lw=0.5, alpha=0.3)
plt.axhline(expected_price, linestyle='--', color='blue', label=f'Target: ${expected_price:.2f}')
plt.axhline(median_price, linestyle='--', color='green', label=f'Median: ${median_price:.2f}')
plt.axhline(p90_price, linestyle='--', color='orange', label=f'P90: ${p90_price:.2f}')
plt.axhline(p10_price, linestyle='--', color='red', label=f'P10: ${p10_price:.2f}')
plt.title('NVDA Price Trajectories (Historical + 12-Month Simulation)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Static histogram
plt.figure(figsize=(10, 5))
plt.hist(final_prices, bins=100, density=True, alpha=0.75, color='lightgreen')
plt.axvline(expected_price, color='blue', linestyle='--', label=f'Target: ${expected_price:.2f}')
plt.axvline(median_price, color='green', linestyle='--', label=f'Median: ${median_price:.2f}')
plt.axvline(p90_price, color='orange', linestyle='--', label=f'P90: ${p90_price:.2f}')
plt.axvline(p10_price, color='red', linestyle='--', label=f'P10: ${p10_price:.2f}')
plt.title('Simulated Price Distribution of NVDA in 12 Months (GBM)')
plt.xlabel('Estimated Price')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
