# ==============================================================================
# Exploratory Deep Learning: PyTorch LSTM Network Training & Evaluation
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Two-layer stacked LSTM architecture:
# - Input window: 24 past sequential hours
# - Hidden Layer 1: 64 LSTM units + Dropout (0.2)
# - Hidden Layer 2: 32 LSTM units + Dropout (0.2)
# - Fully Connected Dense Layer: 16 units (ReLU) -> 1 linear output unit

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Define LSTM Model Architecture
class TrafficLSTM(nn.Module):
    """
    Two-layer stacked LSTM network for sequential hourly traffic flow prediction.
    - Hidden dimension = 64 units per recurrent cell to balance representation capacity against parameter budget.
    - Layer stacking: 2 recurrent layers capture multi-hour diurnal and inter-day transitions.
    - Regularization: Dropout (p=0.2) applied between recurrent layers to mitigate over-parameterization.
    - Linear projection: Fully connected projection (16 units with ReLU activation) maps hidden state to scalar output.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take output from last timestep only
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        return out


def main():
    # 1. Load Prepared Sequences
    data_path = "lstm_sequences.npz"
    if not os.path.exists(data_path):
        print("[ERROR] Cannot find lstm_sequences.npz!")
        print("  -> Please run 06_lstm_data_preparation.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 3.1 — SCRIPT 7: LSTM MODEL TRAINING (PyTorch)")
    print("=" * 60)

    data = np.load(data_path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scale_min = float(data['scale_min'])
    scale_max = float(data['scale_max'])

    print(f"[INFO] Data loaded from: {data_path}")
    print(f"  -> Training sequences: {X_train.shape[0]:,} (shape: {X_train.shape})")
    print(f"  -> Testing sequences:  {X_test.shape[0]:,}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # 2. Build LSTM Model
    print("\n[INFO] Building LSTM Neural Network Architecture...")
    device = torch.device('cpu')
    model = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> Total trainable parameters: {total_params:,}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. Train the Model
    EPOCHS = 50
    print(f"\n[INFO] Training LSTM for {EPOCHS} epochs (this may take a few minutes)...")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase (using test set for monitoring)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t.to(device))
            val_loss = criterion(val_outputs, y_test_t.to(device)).item()
            val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:>3}/{EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  -> Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 4. Save Model
    model_path = "lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] LSTM model saved to: {model_path}")

    # 5. Predict on Test Set
    print("[INFO] Predicting on test holdout set...")
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_t.to(device)).cpu().numpy().flatten()

    # Inverse transform: convert scaled (0-1) values back to real traffic counts
    y_test_real = y_test * (scale_max - scale_min) + scale_min
    predictions_real = predictions_scaled * (scale_max - scale_min) + scale_min

    # 6. Calculate Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
    mae = mean_absolute_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)

    non_zero = y_test_real != 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((y_test_real[non_zero] - predictions_real[non_zero]) / y_test_real[non_zero])) * 100
    else:
        mape = float('nan')

    # Baselines for comparison
    arima_rmse = 2730.6931
    xgb_rmse = 804.7589

    print("\n" + "=" * 60)
    print("LSTM EVALUATION METRICS")
    print("=" * 60)
    print(f"  LSTM RMSE:      {rmse:,.4f}")
    print(f"  LSTM MAE:       {mae:,.4f}")
    print(f"  LSTM MAPE:      {mape:.2f}%")
    print(f"  LSTM R2 Score:  {r2:.4f}")
    print("-" * 60)
    print(f"  vs ARIMA RMSE ({arima_rmse:,.2f}): {((arima_rmse - rmse) / arima_rmse) * 100:.1f}% improvement")
    xgb_comparison = ((xgb_rmse - rmse) / xgb_rmse) * 100
    if rmse < xgb_rmse:
        print(f"  vs XGBoost RMSE ({xgb_rmse:,.2f}):  {xgb_comparison:.1f}% improvement")
    else:
        print(f"  vs XGBoost RMSE ({xgb_rmse:,.2f}):  XGBoost is {abs(xgb_comparison):.1f}% better (expected for tabular data)")
    print("=" * 60)

    # 7. Generate Forecast Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('LSTM Deep Learning Forecast vs Actual Traffic (20% Holdout)', fontsize=18, fontweight='bold')

    plot_len = min(200, len(y_test_real))
    ax.plot(range(plot_len), y_test_real[:plot_len], color='#3498db', label='Actual Traffic', linewidth=2.0)
    ax.plot(range(plot_len), predictions_real[:plot_len], color='#e67e22', linestyle='--', label='LSTM Prediction', linewidth=2.0)

    ax.set_xlabel('Holdout Observation Index (Hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Traffic Volume (all_motor_vehicles)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='upper left', fontsize=12)

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#fdebd0', alpha=0.9, edgecolor='gray')
    metrics_str = f"LSTM RMSE = {rmse:,.2f}\nLSTM MAE  = {mae:,.2f}\nR2 Score  = {r2:.4f}"
    ax.text(0.98, 0.95, metrics_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

    plt.tight_layout()
    plt.savefig("lstm_forecast_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Forecast plot saved to: lstm_forecast_plot.png")

    # 8. Training Loss Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Training Loss', color='#3498db')
    ax.plot(val_losses, label='Validation Loss', color='#e74c3c')
    ax.set_title('LSTM Training & Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error (Loss)')
    ax.legend()
    plt.tight_layout()
    plt.savefig("lstm_training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Training loss plot saved to: lstm_training_loss.png")

    # 9. Save Results
    with open("lstm_results.txt", "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("PHASE 3.1: LSTM DEEP LEARNING MODEL — RESULTS REPORT\n")
        f.write("======================================================================\n\n")
        f.write(f"LSTM RMSE:  {rmse:,.4f}\n")
        f.write(f"LSTM MAE:   {mae:,.4f}\n")
        f.write(f"LSTM MAPE:  {mape:.2f}%\n")
        f.write(f"LSTM R2:    {r2:.4f}\n\n")
        f.write(f"Epochs trained: {len(train_losses)}\n")
        f.write(f"Architecture: LSTM(64, 2 layers) -> Dropout(0.2) -> Dense(16) -> Dense(1)\n")
        f.write(f"Framework: PyTorch {torch.__version__}\n")

    print("[SUCCESS] Results saved to: lstm_results.txt")

if __name__ == "__main__":
    main()
