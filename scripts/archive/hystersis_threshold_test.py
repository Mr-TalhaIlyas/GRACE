import numpy as np
import matplotlib.pyplot as plt

# Generate the same synthetic probability sequence
np.random.seed(42)
time_steps = np.arange(0, 100)
prob_sequence = 0.5 + 0.3 * np.sin(0.2 * time_steps) + 0.05 * np.random.randn(len(time_steps))
prob_sequence = np.clip(prob_sequence, 0, 1)

# Thresholds
high_threshold = 0.7
low_threshold = 0.3
standard_threshold = 0.5

# Standard classification with a single threshold at 0.5
standard_preds = (prob_sequence >= standard_threshold).astype(int)

# Hysteresis-based classification
hyst_preds = np.zeros(len(prob_sequence), dtype=int)
current_state = 0  # Start with negative class
for i, prob in enumerate(prob_sequence):
    if current_state == 0:
        if prob >= high_threshold:
            current_state = 1
    else:
        if prob < low_threshold:
            current_state = 0
    hyst_preds[i] = current_state

# Plot everything on one figure using twin axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot probability and thresholds on ax1
line_prob, = ax1.plot(time_steps, prob_sequence, label='Probability', color='tab:orange', linewidth=1.5)
ax1.axhline(high_threshold, color='grey', linestyle='--', label='High Threshold (0.7)')
ax1.axhline(low_threshold, color='grey', linestyle='-.', label='Low Threshold (0.3)')
ax1.axhline(standard_threshold, color='black', linestyle=':', label='Standard Threshold (0.5)')

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Probability')
ax1.set_ylim(-0.1, 1.05)

# Create a second y-axis for prediction sequences
ax2 = ax1.twinx()

# Plot step plots for standard and hysteresis predictions on ax2
line_std, = ax2.step(time_steps, standard_preds, where='mid', label='Standard (0.5) Predictions', linewidth=1.5, color='tab:blue')
line_hyst, = ax2.step(time_steps, hyst_preds, where='mid', label='Hysteresis Predictions', linewidth=1.5, linestyle='--', color='tab:red')

ax2.set_ylabel('Prediction (0=Neg, 1=Pos)')
ax2.set_ylim(-0.1, 1.05)

# Combine legends from both axes
lines = [line_prob, line_std, line_hyst]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Combined View: Probabilities with Standard vs. Hysteresis Predictions')
plt.tight_layout()
plt.show()
