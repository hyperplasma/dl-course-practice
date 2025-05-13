import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df_summary = pd.read_csv('checkpoints/resnet50/training_summary.csv')
# df_summary = pd.read_csv('checkpoints/deeplabv3/training_summary.csv')

# Extract relevant columns
steps = df_summary['step']
train_dice_losses = df_summary['train dice loss']
eval_dscs = df_summary['eval dsc']

# Convert step to integers for plotting
steps = steps.apply(lambda x: int(x.strip('Step[]')))

# Plotting
plt.figure(figsize=(10, 5))

# Plot train dice loss
plt.subplot(1, 2, 1)
plt.plot(steps, train_dice_losses, label='Train Dice Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Train Dice Loss')
plt.title('Train Dice Loss Over Steps')
plt.legend()

# Plot eval dsc
plt.subplot(1, 2, 2)
plt.plot(steps, eval_dscs, label='Eval DSC', color='red')
plt.xlabel('Step')
plt.ylabel('Eval DSC')
plt.title('Eval DSC Over Steps')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
