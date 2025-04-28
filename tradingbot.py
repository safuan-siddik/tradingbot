"""
Trading Bot with Custom Neural Network Implementation
=====================================================
A complete trading bot that uses custom neural network implementation,
LSTM, and XGBoost for time series prediction.

Available models:
- 'nn': Simple neural network
- 'lstm': Long Short-Term Memory network
- 'xgb': XGBoost regressor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import time
import logging
import os
import json
import argparse
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import random
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class CustomNeuralNetwork:
    """
    A custom implementation of a neural network with one hidden layer
    that can be used for time series prediction.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """Initialize the neural network with random weights."""
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def tanh(self, x):
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh function."""
        return 1 - np.power(x, 2)
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """Forward pass through the network."""
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.tanh(self.hidden_input)
        
        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # Linear activation for regression
        
        return self.output
    
    def backward(self, X, y, output):
        """Backward pass to update weights."""
        # Calculate error
        output_error = y - output
        output_delta = output_error
        
        # Hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        
        # Return error for monitoring
        return np.mean(np.square(output_error))
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train the neural network."""
        history = {'loss': []}
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Create mini-batches
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = math.ceil(n_samples / batch_size)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                batch_loss = self.backward(X_batch, y_batch, output)
                epoch_loss += batch_loss
            
            # Average loss for the epoch
            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions using the trained network."""
        return self.forward(X)
    
    def save_weights(self, filepath):
        """Save the model weights to a file."""
        weights = {
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_output': self.bias_output.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, filepath):
        """Load model weights from a file."""
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        self.weights_input_hidden = np.array(weights['weights_input_hidden'])
        self.bias_hidden = np.array(weights['bias_hidden'])
        self.weights_hidden_output = np.array(weights['weights_hidden_output'])
        self.bias_output = np.array(weights['bias_output'])


class CustomLSTMCell:
    """
    A custom implementation of a Long Short-Term Memory (LSTM) cell
    for sequence modeling without external dependencies.
    """
    
    def __init__(self, input_size, hidden_size):
        """Initialize LSTM cell parameters."""
        # Xavier/Glorot initialization for weights
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Input gate
        self.Wi = np.random.randn(input_size, hidden_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros((1, hidden_size))
        
        # Forget gate
        self.Wf = np.random.randn(input_size, hidden_size) * scale
        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.bf = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wo = np.random.randn(input_size, hidden_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros((1, hidden_size))
        
        # Cell state
        self.Wc = np.random.randn(input_size, hidden_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.bc = np.zeros((1, hidden_size))
        
        # Gradients
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
        
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
        
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)
        
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)
    
    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h_prev, c_prev):
        """Forward pass through the LSTM cell."""
        # Input gate
        i = self.sigmoid(np.dot(x, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(x, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
        
        # Cell state candidate
        c_candidate = np.tanh(np.dot(x, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
        
        # Cell state
        c_next = f * c_prev + i * c_candidate
        
        # Hidden state
        h_next = o * np.tanh(c_next)
        
        # Cache for backward pass
        cache = (x, h_prev, c_prev, i, f, o, c_candidate, c_next)
        
        return h_next, c_next, cache
    
    def backward(self, dh_next, dc_next, cache, learning_rate=0.01):
        """Backward pass through the LSTM cell."""
        (x, h_prev, c_prev, i, f, o, c_candidate, c_next) = cache
        
        # Output gate
        do = dh_next * np.tanh(c_next)
        do = do * o * (1 - o)
        
        # Cell state
        dc_next = dc_next + dh_next * o * (1 - np.tanh(c_next)**2)
        
        # Forget gate
        df = dc_next * c_prev
        df = df * f * (1 - f)
        
        # Input gate
        di = dc_next * c_candidate
        di = di * i * (1 - i)
        
        # Cell candidate
        dc_candidate = dc_next * i
        dc_candidate = dc_candidate * (1 - c_candidate**2)
        
        # Gradients for previous hidden state and cell state
        dh_prev = (np.dot(do, self.Uo.T) + 
                   np.dot(df, self.Uf.T) + 
                   np.dot(di, self.Ui.T) + 
                   np.dot(dc_candidate, self.Uc.T))
        dc_prev = dc_next * f
        
        # Compute gradients
        self.dWo += np.dot(x.T, do)
        self.dUo += np.dot(h_prev.T, do)
        self.dbo += np.sum(do, axis=0, keepdims=True)
        
        self.dWf += np.dot(x.T, df)
        self.dUf += np.dot(h_prev.T, df)
        self.dbf += np.sum(df, axis=0, keepdims=True)
        
        self.dWi += np.dot(x.T, di)
        self.dUi += np.dot(h_prev.T, di)
        self.dbi += np.sum(di, axis=0, keepdims=True)
        
        self.dWc += np.dot(x.T, dc_candidate)
        self.dUc += np.dot(h_prev.T, dc_candidate)
        self.dbc += np.sum(dc_candidate, axis=0, keepdims=True)
        
        # Gradient for input
        dx = (np.dot(do, self.Wo.T) + 
              np.dot(df, self.Wf.T) + 
              np.dot(di, self.Wi.T) + 
              np.dot(dc_candidate, self.Wc.T))
        
        # Update weights with gradient descent
        self.Wi -= learning_rate * self.dWi
        self.Ui -= learning_rate * self.dUi
        self.bi -= learning_rate * self.dbi
        
        self.Wf -= learning_rate * self.dWf
        self.Uf -= learning_rate * self.dUf
        self.bf -= learning_rate * self.dbf
        
        self.Wo -= learning_rate * self.dWo
        self.Uo -= learning_rate * self.dUo
        self.bo -= learning_rate * self.dbo
        
        self.Wc -= learning_rate * self.dWc
        self.Uc -= learning_rate * self.dUc
        self.bc -= learning_rate * self.dbc
        
        # Reset gradients
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
        
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
        
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)
        
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)
        
        return dx, dh_prev, dc_prev


class CustomLSTM:
    """
    A custom implementation of an LSTM network for sequence prediction.
    """
    
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        """Initialize the LSTM network."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # LSTM cell
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)
        
        # Output layer
        scale = np.sqrt(2.0 / (hidden_size + output_size))
        self.Wy = np.random.randn(hidden_size, output_size) * scale
        self.by = np.zeros((1, output_size))
        
        # Gradients
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)
    
    def forward(self, X):
        """
        Forward pass through the LSTM network.
        X shape: (batch_size, sequence_length, input_size)
        """
        batch_size = X.shape[0]
        
        # Initialize hidden state and cell state
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        # Forward through time steps
        caches = []
        for t in range(self.sequence_length):
            x_t = X[:, t, :]
            h, c, cache = self.lstm_cell.forward(x_t, h, c)
            caches.append(cache)
        
        # Output layer
        y_pred = np.dot(h, self.Wy) + self.by
        
        # Cache for backward pass
        cache = (X, caches, h)
        
        return y_pred, cache
    
    def backward(self, y_pred, y_true, cache, learning_rate=0.01):
        """
        Backward pass through the LSTM network.
        y_pred, y_true shape: (batch_size, output_size)
        """
        X, caches, h = cache
        batch_size = X.shape[0]
        
        # Output layer gradients
        dy = y_pred - y_true
        dh = np.dot(dy, self.Wy.T)
        
        self.dWy = np.dot(h.T, dy)
        self.dby = np.sum(dy, axis=0, keepdims=True)
        
        # Update output layer weights
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby
        
        # Initialize gradients for hidden and cell states
        dh_next = dh
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Backward through time
        for t in reversed(range(self.sequence_length)):
            cache_t = caches[t]
            x_t = X[:, t, :]
            
            dx_t, dh_next, dc_next = self.lstm_cell.backward(
                dh_next, dc_next, cache_t, learning_rate
            )
        
        # Compute total loss
        loss = np.mean(np.square(y_pred - y_true))
        
        return loss
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=True):
        """
        Train the LSTM network.
        X shape: (n_samples, sequence_length, input_size)
        y shape: (n_samples, output_size)
        """
        n_samples = X.shape[0]
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = math.ceil(n_samples / batch_size)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred, cache = self.forward(X_batch)
                
                # Backward pass
                loss = self.backward(y_pred, y_batch, cache, learning_rate)
                epoch_loss += loss
            
            # Average loss for the epoch
            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the LSTM network.
        X shape: (n_samples, sequence_length, input_size)
        """
        y_pred, _ = self.forward(X)
        return y_pred
    
    def save_weights(self, filepath):
        """Save the model weights to a file."""
        # Flatten all weights into lists for JSON serialization
        weights = {
            'Wi': self.lstm_cell.Wi.tolist(),
            'Ui': self.lstm_cell.Ui.tolist(),
            'bi': self.lstm_cell.bi.tolist(),
            'Wf': self.lstm_cell.Wf.tolist(),
            'Uf': self.lstm_cell.Uf.tolist(),
            'bf': self.lstm_cell.bf.tolist(),
            'Wo': self.lstm_cell.Wo.tolist(),
            'Uo': self.lstm_cell.Uo.tolist(),
            'bo': self.lstm_cell.bo.tolist(),
            'Wc': self.lstm_cell.Wc.tolist(),
            'Uc': self.lstm_cell.Uc.tolist(),
            'bc': self.lstm_cell.bc.tolist(),
            'Wy': self.Wy.tolist(),
            'by': self.by.tolist(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, filepath):
        """Load model weights from a file."""
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        # Load weights into model
        self.lstm_cell.Wi = np.array(weights['Wi'])
        self.lstm_cell.Ui = np.array(weights['Ui'])
        self.lstm_cell.bi = np.array(weights['bi'])
        
        self.lstm_cell.Wf = np.array(weights['Wf'])
        self.lstm_cell.Uf = np.array(weights['Uf'])
        self.lstm_cell.bf = np.array(weights['bf'])
        
        self.lstm_cell.Wo = np.array(weights['Wo'])
        self.lstm_cell.Uo = np.array(weights['Uo'])
        self.lstm_cell.bo = np.array(weights['bo'])
        
        self.lstm_cell.Wc = np.array(weights['Wc'])
        self.lstm_cell.Uc = np.array(weights['Uc'])
        self.lstm_cell.bc = np.array(weights['bc'])
        
        self.Wy = np.array(weights['Wy'])
        self.by = np.array(weights['by'])


class XGBoostModel:
    """
    XGBoost wrapper class that follows the same interface as our custom models.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """Initialize XGBoost model with parameters."""
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=learning_rate,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        self.input_size = input_size
        self.sequence_length = None  # Will be set during training
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train the XGBoost model."""
        # For XGBoost, we need to reshape the input if it's sequential data
        if len(X.shape) == 3:  # (samples, sequence_length, features)
            self.sequence_length = X.shape[1]
            X = X.reshape(X.shape[0], -1)  # Flatten to 2D
        
        # Create eval set for monitoring
        eval_size = int(len(X) * 0.2)
        X_train, X_eval = X[:-eval_size], X[-eval_size:]
        y_train, y_eval = y[:-eval_size], y[-eval_size:]
        eval_set = [(X_eval, y_eval)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose,
            early_stopping_rounds=10
        )
        
        # Return dummy history for compatibility
        return {'loss': []}
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if len(X.shape) == 3 and self.sequence_length:  # Sequential data
            X = X.reshape(X.shape[0], -1)  # Flatten to 2D
        return self.model.predict(X).reshape(-1, 1)
    
    def save_weights(self, filepath):
        """Save the model to a file."""
        # Change extension for XGBoost model
        filepath = filepath.replace('.json', '.xgb')
        self.model.save_model(filepath)
    
    def load_weights(self, filepath):
        """Load the model from a file."""
        # Change extension for XGBoost model
        filepath = filepath.replace('.json', '.xgb')
        self.model.load_model(filepath)


class TradingBot:
    """
    Trading Bot that uses custom neural networks for prediction
    and includes complete trading strategy with risk management.
    """
    
    def __init__(self, config_file=None):
        """Initialize the trading bot with configuration parameters."""
        # Default configuration
        self.config = {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "lookback_days": 365,
            "test_size": 0.2,
            "feature_columns": ["Open", "High", "Low", "Close", "Volume"],
            "target_column": "Close",
            "sequence_length": 20,
            "training_epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "hidden_size": 32,
            "model_type": "lstm",  # Options: "lstm", "nn", or "xgb"
            "initial_capital": 10000,
            "position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "models_dir": "models",
            "data_dir": "data"
        }
        
        # Load configuration from file if provided
        if config_file:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Create directories if they don't exist
        for directory in [self.config["models_dir"], self.config["data_dir"]]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize class variables
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.portfolio = {
            "cash": self.config["initial_capital"],
            "positions": {},
            "history": []
        }
        
        logger.info(f"Trading bot initialized with configuration: {self.config}")
    
    def fetch_data(self, symbol, start_date=None, end_date=None):
        """Fetch historical market data for a given symbol."""
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=self.config["lookback_days"])).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            data_file = os.path.join(self.config["data_dir"], f"{symbol}_{start_date}_{end_date}.csv")
            data.to_csv(data_file)
            logger.info(f"Data saved to {data_file}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def prepare_data(self, data, symbol):
        """
        Prepare and preprocess data for training the model.
        Includes feature engineering, scaling, and sequence creation.
        """
        logger.info(f"Preparing data for {symbol}")
        
        # Create additional features
        df = data.copy()
        
        # Technical indicators
        # SMA - Simple Moving Average
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # EMA - Exponential Moving Average
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # RSI - Relative Strength Index
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Volatility (using standard deviation of returns)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Drop NaN values
        df.dropna(inplace=True)
        if df.empty:
            logger.warning(f"After feature engineering and dropna, no data left for {symbol}. Skipping.")
            return None  # Safe exit

        
        # Select features for model training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20',
            'BB_upper', 'BB_lower', 'RSI',
            'MACD', 'MACD_signal', 'Daily_Return', 'Volatility'
        ]
        
        # Create feature and target datasets
        features = df[feature_columns].values
        targets = df[self.config['target_column']].values.reshape(-1, 1)
        
        # Scale features and targets
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        features_scaled = feature_scaler.fit_transform(features)
        targets_scaled = target_scaler.fit_transform(targets)
        
        # Store scalers for later use
        self.scalers[symbol] = {
            'features': feature_scaler,
            'target': target_scaler
        }
        
        # Create sequences for model training
        X, y = [], []
        seq_length = self.config['sequence_length']
        
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i:i+seq_length])
            y.append(targets_scaled[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data preparation completed for {symbol}. Features shape: {X.shape}, Targets shape: {y.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'df': df
        }
    
    def build_model(self, symbol, data):
        """Build and initialize the prediction model."""
        model_type = self.config['model_type']
        input_size = data['X_train'].shape[2]  # Number of features
        hidden_size = self.config['hidden_size']
        output_size = 1  # Price prediction
        sequence_length = self.config['sequence_length']
        
        if model_type == 'lstm':
            model = CustomLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                sequence_length=sequence_length
            )
        elif model_type == 'xgb':
            model = XGBoostModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                learning_rate=self.config['learning_rate']
            )
        else:  # Default to simple neural network
            # Flatten input for simple neural network
            model = CustomNeuralNetwork(
                input_size=input_size * sequence_length,
                hidden_size=hidden_size,
                output_size=output_size,
                learning_rate=self.config['learning_rate']
            )
        
        logger.info(f"Model built for {symbol} with {model_type} architecture")
        return model
    def train_all_models(self):
        """Train models for all configured symbols."""
        for symbol in self.config['symbols']:
            # Fetch data
            data = self.fetch_data(symbol)
            if data is None or data.empty:
                logger.error(f"Could not fetch data for {symbol}, skipping")
                continue
            
            # Prepare data
            prepared_data = self.prepare_data(data, symbol)
            
            # Train model
            self.train_model(symbol, prepared_data)
            
            # Store evaluation metrics for later use
            if not hasattr(self, 'evaluation_metrics'):
                self.evaluation_metrics = {}
            self.evaluation_metrics[symbol] = self.evaluate_model(symbol, prepared_data)
        
        logger.info("All models trained successfully")

    def load_all_models(self):
        """Load all saved models."""
        for symbol in self.config['symbols']:
            model_path = os.path.join(self.config["models_dir"], f"{symbol}_model.json")
            
            if not os.path.exists(model_path):
                logger.warning(f"No saved model found for {symbol}")
                continue
            
            # Create model instance
            # Need to first get data to determine input dimensions
            data = self.fetch_data(symbol)
            if data is None or data.empty:
                logger.error(f"Could not fetch data for {symbol}, skipping model loading")
                continue
            
            prepared_data = self.prepare_data(data, symbol)
            
            # Build model with correct dimensions
            model = self.build_model(symbol, prepared_data)
            
            # Load weights
            try:
                model.load_weights(model_path)
                self.models[symbol] = model
                logger.info(f"Model loaded for {symbol}")
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")
        
        logger.info("All available models loaded")

    def run_backtest(self, start_date, end_date):
        """Run a backtest of the trading strategy over a historical period."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Reset portfolio to initial state for backtest
        backtest_portfolio = {
            "cash": self.config["initial_capital"],
            "positions": {},
            "history": []
        }
        
        # Store original portfolio
        original_portfolio = self.portfolio
        self.portfolio = backtest_portfolio
        
        # Fetch historical data for all symbols
        historical_data = {}
        for symbol in self.config['symbols']:
            data = self.fetch_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                historical_data[symbol] = data
        
        # Process each day in the backtest period
        dates = sorted(list(set.union(*[set(data.index) for data in historical_data.values()])))
        portfolio_values = []
        
        for i, date in enumerate(dates):
            if i < self.config['sequence_length']:
                continue  # Skip initial dates as we need sequence_length data points
            
            logger.info(f"Backtest processing date: {date.strftime('%Y-%m-%d')}")
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol, data in historical_data.items():
                if date in data.index:
                    current_prices[symbol] = data.loc[date, 'Close']
            
            # Check stop loss and take profit
            self.check_stop_loss_take_profit(current_prices)
            
            # Make predictions and generate signals
            for symbol, data in historical_data.items():
                # Get data up to current date for prediction
                data_until_date = data[data.index <= date]
                if len(data_until_date) < self.config['sequence_length'] + 1:
                    logger.warning(f"Skipping {symbol} on {date.strftime('%Y-%m-%d')} due to insufficient data.")
                    continue

                if symbol not in self.models:
                    logger.info(f"No model found for {symbol}, training on available data...")
                    prepared_data = self.prepare_data(data_until_date, symbol)
                    if prepared_data is None:
                        logger.warning(f"Skipping {symbol} on {date.strftime('%Y-%m-%d')} due to no prepared data after feature engineering.")
                        continue  # Skip to next symbol
                    if len(prepared_data['X_train'].shape) != 3:
                        logger.warning(f"Skipping {symbol} on {date.strftime('%Y-%m-%d')} because X_train shape is invalid: {prepared_data['X_train'].shape}")
                        continue
                    model = self.train_model(symbol, prepared_data)
                    if model is None:
                        logger.warning(f"Failed to train model for {symbol} on {date.strftime('%Y-%m-%d')}, skipping prediction.")
                        continue
                    self.models[symbol] = model


                prediction = self.predict_next_day(symbol, data_until_date)
                if prediction:
                    prediction['current_price'] = current_prices.get(symbol, 0)
                    signal = self.generate_trading_signals(prediction)
                    self.execute_trade(signal)
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(current_prices)
            portfolio_values.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': portfolio_value['total_value']
            })
        
        # Calculate backtest metrics
        initial_value = self.config["initial_capital"]
        final_value = portfolio_values[-1]['value'] if portfolio_values else initial_value
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate annualized return
        days = len(dates)
        if days > 0:
            annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate drawdown
        values = [pv['value'] for pv in portfolio_values]
        max_drawdown = 0
        if values:
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        logger.info("Backtest Results:")
        logger.info(f"Initial Portfolio Value: ${initial_value:.2f}")
        logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Annualized Return: {annualized_return:.2f}%")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        dates = [pv['date'] for pv in portfolio_values]
        values = [pv['value'] for pv in portfolio_values]
        plt.plot(dates, values)
        plt.title('Portfolio Value During Backtest')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["data_dir"], "backtest_results.png"))
        plt.close()
        
        # Restore original portfolio
        self.portfolio = original_portfolio
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values
        }

    def run_trading_session(self):
        """Run a complete trading session including prediction and execution."""
        logger.info("Starting trading session")
        
        # Load portfolio state
        self.load_portfolio()
        
        # Load or train models if not already done
        if not self.models:
            try:
                self.load_all_models()
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                logger.info("Training new models")
                self.train_all_models()
        
        # Get current market prices
        current_prices = {}
        for symbol in self.config['symbols']:
            # Fetch recent data
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
            recent_data = self.fetch_data(symbol, start_date, end_date)
            
            if recent_data is not None and not recent_data.empty:
                current_prices[symbol] = recent_data['Close'].iloc[-1]
        
        # Check stop loss and take profit
        self.check_stop_loss_take_profit(current_prices)
        
        # Make predictions and generate signals
        signals = []
        for symbol in self.config['symbols']:
            prediction = self.predict_next_day(symbol)
            if prediction:
                signal = self.generate_trading_signals(prediction)
                signals.append(signal)
                
                # Execute trade based on signal
                self.execute_trade(signal)
        
        # Update portfolio value
        portfolio_value = self.update_portfolio_value(current_prices)
        
        # Save current portfolio state
        self.save_portfolio()
        
        logger.info("Trading session completed")
        return {
            'signals': signals,
            'portfolio_value': portfolio_value
        }

    def visualize_portfolio_performance(self, days=30):
        """Generate visualization of portfolio performance over time."""
        if not self.portfolio['history']:
            logger.info("No trading history available for visualization")
            return
        
        # Extract dates and portfolio values from history
        history = sorted(self.portfolio['history'], key=lambda x: x['timestamp'])
        trades_df = pd.DataFrame(history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Group by date and visualize
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Trade actions
        plt.subplot(2, 1, 1)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        for symbol in self.config['symbols']:
            symbol_data = self.fetch_data(symbol, 
                                        (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d'),
                                        datetime.datetime.now().strftime('%Y-%m-%d'))
            if symbol_data is not None and not symbol_data.empty:
                plt.plot(symbol_data.index, symbol_data['Close'], label=symbol)
                
                # Plot buy/sell points
                symbol_buys = buy_trades[buy_trades['symbol'] == symbol]
                symbol_sells = sell_trades[sell_trades['symbol'] == symbol]
                
                if not symbol_buys.empty:
                    plt.scatter(symbol_buys['timestamp'], symbol_buys['price'], 
                            marker='^', color='g', s=100, label=f'{symbol} Buy' if symbol == self.config['symbols'][0] else "")
                
                if not symbol_sells.empty:
                    plt.scatter(symbol_sells['timestamp'], symbol_sells['price'], 
                            marker='v', color='r', s=100, label=f'{symbol} Sell' if symbol == self.config['symbols'][0] else "")
        
        plt.title('Trading Actions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Portfolio value over time
        plt.subplot(2, 1, 2)
        
        # Calculate portfolio value at each trade point
        portfolio_values = []
        cash = self.config['initial_capital']
        positions = {}
        
        for trade in history:
            date = pd.to_datetime(trade['timestamp'])
            symbol = trade['symbol']
            action = trade['action']
            price = trade['price']
            shares = trade['shares']
            
            if action == 'BUY':
                cash -= price * shares
                if symbol in positions:
                    positions[symbol] += shares
                else:
                    positions[symbol] = shares
            elif action == 'SELL':
                cash += price * shares
                positions[symbol] -= shares
                if positions[symbol] <= 0:
                    del positions[symbol]
            
            # Calculate positions value
            positions_value = 0
            for sym, qty in positions.items():
                positions_value += self.fetch_latest_price(sym) * qty
            
            total_value = cash + positions_value
            portfolio_values.append({
                'date': date,
                'value': total_value
            })
        
        # Plot portfolio value
        if portfolio_values:
            portfolio_df = pd.DataFrame(portfolio_values)
            plt.plot(portfolio_df['date'], portfolio_df['value'], 'b-', label='Portfolio Value')
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["data_dir"], "portfolio_performance.png"))
        plt.close()
        
        logger.info(f"Portfolio performance visualization saved to {os.path.join(self.config['data_dir'], 'portfolio_performance.png')}")

    def fetch_latest_price(self, symbol):
        """Fetch the latest available price for a symbol."""
        try:
            data = yf.download(symbol, period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
        
        # If we can't get current price, use the last known price from portfolio
        if symbol in self.portfolio['positions']:
            return self.portfolio['positions'][symbol]['avg_price']
        
        return 0
        
    def train_model(self, symbol, data):
        """Train the prediction model on prepared data."""
        logger.info(f"Training model for {symbol}")
        
        # Build model
        model = self.build_model(symbol, data)
        
        # Training parameters
        epochs = self.config['training_epochs']
        batch_size = self.config['batch_size']
        learning_rate = self.config['learning_rate']
        
        # Prepare data for training
        X_train, y_train = data['X_train'], data['y_train']
        
        if len(X_train) < self.config['batch_size']:
            logger.warning(f"Not enough training samples for {symbol} to train the model. Needed: {self.config['batch_size']}, Found: {len(X_train)}. Skipping.")
            return None
        # Train the model based on its type
        if self.config['model_type'] == 'lstm':
            history = model.train(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        elif self.config['model_type'] == 'xgb':
            history = model.train(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True
            )
        else:  # Simple neural network
            # Flatten input for simple neural network
            X_train_flattened = X_train.reshape(X_train.shape[0], -1)
            history = model.train(
                X_train_flattened, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True
            )
        
        # Save the trained model
        model_path = os.path.join(self.config["models_dir"], f"{symbol}_model.json")
        if self.config['model_type'] == 'lstm':
            model.save_weights(model_path)
        elif self.config['model_type'] == 'xgb':
            model.save_weights(model_path)
        else:
            model.save_weights(model_path)
        
        self.models[symbol] = model
        logger.info(f"Model for {symbol} trained and saved to {model_path}")
        
        # Evaluate model on test data
        self.evaluate_model(symbol, data)
        
        return model

    def evaluate_model(self, symbol, data):
        """Evaluate the trained model on test data."""
        model = self.models[symbol]
        
        # Prepare test data
        X_test, y_test = data['X_test'], data['y_test']
        
        # Make predictions based on model type
        if self.config['model_type'] == 'lstm':
            y_pred = model.predict(X_test)
        elif self.config['model_type'] == 'xgb':
            y_pred = model.predict(X_test)
        else:  # Simple neural network
            X_test_flattened = X_test.reshape(X_test.shape[0], -1)
            y_pred = model.predict(X_test_flattened)
        
        # Inverse transform predictions and actual values for evaluation
        target_scaler = self.scalers[symbol]['target']
        y_pred_unscaled = target_scaler.inverse_transform(y_pred)
        y_test_unscaled = target_scaler.inverse_transform(y_test)
        
        # Calculate error metrics
        mse = np.mean(np.square(y_pred_unscaled - y_test_unscaled))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_unscaled - y_test_unscaled))
        
        # Calculate directional accuracy (how often we predict the right direction)
        direction_pred = np.diff(y_pred_unscaled.flatten())
        direction_actual = np.diff(y_test_unscaled.flatten())
        direction_accuracy = np.mean((direction_pred > 0) == (direction_actual > 0))
        
        logger.info(f"Model evaluation for {symbol}:")
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Directional Accuracy: {direction_accuracy:.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_unscaled, label='Actual')
        plt.plot(y_pred_unscaled, label='Predicted')
        plt.title(f"{symbol} Price Prediction")
        plt.legend()
        plt.savefig(os.path.join(self.config["data_dir"], f"{symbol}_prediction.png"))
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy
        }

    def predict_next_day(self, symbol, latest_data=None):
        """Predict the next day's price for a given symbol."""
        if symbol not in self.models:
            logger.error(f"No trained model found for {symbol}")
            return None
        
        model = self.models[symbol]
        
        # Get the latest data for prediction
        if latest_data is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=self.config["lookback_days"])).strftime('%Y-%m-%d')
            latest_data = self.fetch_data(symbol, start_date, end_date)
        
        if latest_data is None or latest_data.empty:
            logger.error(f"No data available for prediction for {symbol}")
            return None
        
        # Prepare the data for prediction
        prepared_data = self.prepare_data(latest_data, symbol)
        feature_scaler = self.scalers[symbol]['features']
        target_scaler = self.scalers[symbol]['target']
        
        # Get the most recent sequence for prediction
        df = prepared_data['df']
        feature_columns = prepared_data['feature_columns']
        
        # Use the last sequence_length rows for prediction
        recent_features = df[feature_columns].values[-self.config['sequence_length']:]
        recent_features_scaled = feature_scaler.transform(recent_features)
        
        # Reshape for model input
        if self.config['model_type'] == 'lstm':
            model_input = recent_features_scaled.reshape(1, self.config['sequence_length'], -1)
            prediction_scaled = model.predict(model_input)
        elif self.config['model_type'] == 'xgb':
            model_input = recent_features_scaled.reshape(1, -1)
            prediction_scaled = model.predict(model_input)
        else:  # Simple neural network
            model_input = recent_features_scaled.reshape(1, -1)
            prediction_scaled = model.predict(model_input)
        
        # Inverse transform to get actual price prediction
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Calculate confidence based on model evaluation metrics (could be improved)
        # Simple approach: higher direction accuracy = higher confidence
        confidence = 0.5  # Default
        if hasattr(self, 'evaluation_metrics') and symbol in self.evaluation_metrics:
            confidence = self.evaluation_metrics[symbol].get('direction_accuracy', 0.5)
        
        logger.info(f"Prediction for {symbol} next day: {prediction:.2f} with confidence: {confidence:.2f}")
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'current_price': df['Close'].iloc[-1],
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def generate_trading_signals(self, prediction_results):
        """Generate buy/sell/hold signals based on predictions."""
        current_price = prediction_results['current_price']
        predicted_price = prediction_results['prediction']
        confidence = prediction_results['confidence']
        symbol = prediction_results['symbol']
        
        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Force it into a single float
        price_change_pct = float(price_change_pct)
        
        # Basic signal generation logic
        signal = 'HOLD'
        if price_change_pct > 0.01 and confidence > 0.55:  # 1% gain with good confidence
            signal = 'BUY'
        elif price_change_pct < -0.01 and confidence > 0.55:  # 1% loss with good confidence
            signal = 'SELL'
        
        # Consider current positions (avoid buying if already holding, etc.)
        if symbol in self.portfolio['positions'] and signal == 'BUY':
            signal = 'HOLD'  # Already holding
        elif symbol not in self.portfolio['positions'] and signal == 'SELL':
            signal = 'HOLD'  # Nothing to sell
        
        logger.info(f"Signal for {symbol}: {signal} (Change: {price_change_pct:.2%}, Confidence: {confidence:.2f})")
        
        return {
            'symbol': symbol,
            'signal': signal,
            'prediction': predicted_price,
            'current_price': current_price,
            'change_pct': price_change_pct,
            'confidence': confidence,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def execute_trade(self, signal_data):
        """Execute a trade based on the generated signal."""
        symbol = signal_data['symbol']
        signal = signal_data['signal']
        current_price = signal_data['current_price']
        
        if signal == 'HOLD':
            logger.info(f"Holding position for {symbol}")
            return
        
        # Calculate position size
        cash_available = self.portfolio['cash']
        position_value = cash_available * self.config['position_size']
        
        if signal == 'BUY':
            # Calculate number of shares to buy
            shares_to_buy = int(position_value / current_price)
            
            if shares_to_buy <= 0:
                logger.info(f"Not enough cash to buy {symbol}")
                return
            
            # Update portfolio
            cost = shares_to_buy * current_price
            self.portfolio['cash'] -= cost
            
            if symbol in self.portfolio['positions']:
                self.portfolio['positions'][symbol]['shares'] += shares_to_buy
                self.portfolio['positions'][symbol]['avg_price'] = (
                    (self.portfolio['positions'][symbol]['avg_price'] * self.portfolio['positions'][symbol]['shares'] + cost) / 
                    (self.portfolio['positions'][symbol]['shares'] + shares_to_buy)
                )
            else:
                self.portfolio['positions'][symbol] = {
                    'shares': shares_to_buy,
                    'avg_price': current_price,
                    'stop_loss': current_price * (1 - self.config['stop_loss_pct']),
                    'take_profit': current_price * (1 + self.config['take_profit_pct'])
                }
            
            logger.info(f"BOUGHT {shares_to_buy} shares of {symbol} at ${current_price:.2f}")
        
        elif signal == 'SELL':
            if symbol not in self.portfolio['positions']:
                logger.info(f"No position to sell for {symbol}")
                return
            
            # Sell all shares
            shares_to_sell = self.portfolio['positions'][symbol]['shares']
            revenue = shares_to_sell * current_price
            
            # Calculate profit/loss
            avg_buy_price = self.portfolio['positions'][symbol]['avg_price']
            profit_loss = (current_price - avg_buy_price) * shares_to_sell
            profit_loss_pct = (current_price - avg_buy_price) / avg_buy_price
            
            # Update portfolio
            self.portfolio['cash'] += revenue
            del self.portfolio['positions'][symbol]
            
            logger.info(f"SOLD {shares_to_sell} shares of {symbol} at ${current_price:.2f}")
            logger.info(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2%})")
        
        # Record trade in history
        trade_record = {
            'symbol': symbol,
            'action': signal,
            'price': current_price,
            'shares': shares_to_buy if signal == 'BUY' else shares_to_sell,
            'value': cost if signal == 'BUY' else revenue,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if signal == 'SELL':
            trade_record['profit_loss'] = profit_loss
            trade_record['profit_loss_pct'] = profit_loss_pct
        
        self.portfolio['history'].append(trade_record)
        
        # Save portfolio state
        self.save_portfolio()

    def check_stop_loss_take_profit(self, current_prices):
        """Check and execute stop loss or take profit orders."""
        for symbol, position in list(self.portfolio['positions'].items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Check stop loss
            if current_price <= stop_loss:
                logger.info(f"STOP LOSS triggered for {symbol} at ${current_price:.2f}")
                
                # Create signal data for stop loss execution
                signal_data = {
                    'symbol': symbol,
                    'signal': 'SELL',
                    'current_price': current_price
                }
                self.execute_trade(signal_data)
            
            # Check take profit
            elif current_price >= take_profit:
                logger.info(f"TAKE PROFIT triggered for {symbol} at ${current_price:.2f}")
                
                # Create signal data for take profit execution
                signal_data = {
                    'symbol': symbol,
                    'signal': 'SELL',
                    'current_price': current_price
                }
                self.execute_trade(signal_data)

    def update_portfolio_value(self, current_prices):
        """Update and calculate current portfolio value."""
        # Calculate positions value
        positions_value = 0
        for symbol, position in self.portfolio['positions'].items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]
        
        # Total portfolio value
        total_value = self.portfolio['cash'] + positions_value
        
        # Calculate return metrics
        initial_capital = self.config['initial_capital']
        absolute_return = total_value - initial_capital
        percentage_return = (total_value / initial_capital - 1) * 100
        
        logger.info(f"Portfolio Update:")
        logger.info(f"Cash: ${self.portfolio['cash']:.2f}")
        logger.info(f"Positions Value: ${positions_value:.2f}")
        logger.info(f"Total Value: ${total_value:.2f}")
        logger.info(f"Return: ${absolute_return:.2f} ({percentage_return:.2f}%)")
        
        return {
            'cash': self.portfolio['cash'],
            'positions_value': positions_value,
            'total_value': total_value,
            'absolute_return': absolute_return,
            'percentage_return': percentage_return
        }

    def save_portfolio(self):
        """Save portfolio state to a file."""
        portfolio_file = os.path.join(self.config["data_dir"], "portfolio.json")
        
        # Convert datetime objects to strings for JSON serialization
        portfolio_copy = self.portfolio.copy()
        
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio_copy, f, indent=4)
        
        logger.info(f"Portfolio saved to {portfolio_file}")

    def load_portfolio(self):
        """Load portfolio state from a file."""
        portfolio_file = os.path.join(self.config["data_dir"], "portfolio.json")
        
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                self.portfolio = json.load(f)
            logger.info(f"Portfolio loaded from {portfolio_file}")
        else:
            logger.info("No portfolio file found, using default portfolio")


def main():
    """Main function to run the trading bot."""
    parser = argparse.ArgumentParser(description='Trading Bot with Neural Networks')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'trade', 'backtest', 'visualize'],
                        help='Operation mode')
    parser.add_argument('--backtest_start', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--backtest_end', type=str, help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize trading bot
    bot = TradingBot(args.config)
    
    if args.mode == 'train':
        bot.train_all_models()
    
    elif args.mode == 'trade':
        bot.run_trading_session()
    
    elif args.mode == 'backtest':
        if not args.backtest_start or not args.backtest_end:
            logger.error("Backtest mode requires start and end dates")
            return
        
        bot.run_backtest(args.backtest_start, args.backtest_end)
    
    elif args.mode == 'visualize':
        bot.visualize_portfolio_performance()


if __name__ == "__main__":
    main()