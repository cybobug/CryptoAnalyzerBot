import ccxt
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from datetime import datetime, timedelta
import time
import os
import logging
import asyncio
from typing import Optional, Tuple, Dict, List, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import numpy as np
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
import pytz
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeFrame(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"

@dataclass
class TechnicalIndicators:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class ExchangeInfo:
    name: str
    symbols: Set[str]
    timeframes: Set[str]
    has_fetchOHLCV: bool

class EnhancedPricePrediction:
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_days: int = 7,
        lstm_units: List[int] = [100, 50, 50],
        dropout_rates: List[float] = [0.2, 0.2, 0.2]
    ):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.lstm_model = None
        self.prophet_model = None
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger('PricePrediction')

    def create_sequences(self, data: np.array) -> Tuple[np.array, np.array]:
        """Create sequences for LSTM training with multiple features."""
        X, y = [], []
        for i in range(self.sequence_length, len(data) - self.prediction_days + 1):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i:i + self.prediction_days, 0])  # Only predict closing prices
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model with configurable architecture."""
        self.lstm_model = Sequential()
        
        # First LSTM layer
        self.lstm_model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        self.lstm_model.add(Dropout(self.dropout_rates[0]))
        
        # Middle LSTM layers
        for units, dropout in zip(self.lstm_units[1:-1], self.dropout_rates[1:-1]):
            self.lstm_model.add(LSTM(units=units, return_sequences=True))
            self.lstm_model.add(Dropout(dropout))
        
        # Final LSTM layer
        self.lstm_model.add(LSTM(units=self.lstm_units[-1]))
        self.lstm_model.add(Dropout(self.dropout_rates[-1]))
        
        # Output layer
        self.lstm_model.add(Dense(self.prediction_days))
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae']
        )

    def prepare_features(self, df: pd.DataFrame) -> np.array:
        """Prepare and scale features for LSTM model."""
        feature_columns = ['close', 'volume', 'high', 'low']
        features = df[feature_columns].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def train_models(
        self,
        df: pd.DataFrame,
        lstm_epochs: int = 50,
        lstm_batch_size: int = 32,
        validation_split: float = 0.2
    ) -> dict:
        """Train both LSTM and Prophet models."""
        results = {}
        
        try:
            # Train LSTM
            scaled_features = self.prepare_features(df)
            X, y = self.create_sequences(scaled_features)
            
            if X.shape[0] > 0:
                self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                history = self.lstm_model.fit(
                    X, y,
                    epochs=lstm_epochs,
                    batch_size=lstm_batch_size,
                    validation_split=validation_split,
                    verbose=1
                )
                results['lstm_history'] = history.history
            
            # Train Prophet
            prophet_df = df.reset_index()[['timestamp', 'close']].rename(
                columns={'timestamp': 'ds', 'close': 'y'}
            )
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(prophet_df)
            results['prophet_fitted'] = True
            
        except Exception as e:
            self.logger.error(f"Error in training models: {e}")
            results['error'] = str(e)
            
        return results

    def predict(
        self,
        df: pd.DataFrame,
        combine_predictions: bool = True
    ) -> Tuple[List[str], List[float]]:
        """Generate predictions using both LSTM and Prophet models."""
        try:
            # LSTM predictions
            scaled_features = self.prepare_features(df)
            last_sequence = scaled_features[-self.sequence_length:]
            last_sequence = last_sequence.reshape(1, self.sequence_length, scaled_features.shape[1])
            lstm_predictions = self.lstm_model.predict(last_sequence)
            lstm_predictions = lstm_predictions.reshape(-1, 1)
            lstm_predictions = self.scaler.inverse_transform(
                np.hstack([lstm_predictions, np.zeros((lstm_predictions.shape[0], 3))])
            )[:, 0]
            
            # Prophet predictions
            future_dates = self.prophet_model.make_future_dataframe(
                periods=self.prediction_days,
                freq='D'
            )
            prophet_forecast = self.prophet_model.predict(future_dates)
            prophet_predictions = prophet_forecast.tail(self.prediction_days)['yhat'].values
            
            # Combine predictions if requested
            if combine_predictions:
                predictions = (lstm_predictions + prophet_predictions) / 2
            else:
                predictions = lstm_predictions
            
            # Generate dates for predictions
            last_date = pd.to_datetime(df['timestamp'].iloc[-1])
            prediction_dates = [
                (last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(self.prediction_days)
            ]
            
            return prediction_dates, predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in generating predictions: {e}")
            return [], []

# Assume TimeFrame, TechnicalIndicators, SignalType, and ExchangeInfo are already defined elsewhere.

class CryptoMarketAnalyzer:
    def __init__(
        self,
        exchange_ids: List[str] = None,
        indicators: Optional[TechnicalIndicators] = None,
        log_level: int = logging.INFO,
        cache_dir: str = "cache"
    ):
        self.logger = self._setup_logger(log_level)
        self.indicators = indicators or TechnicalIndicators()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.prediction_dates = []
        
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.exchange_info: Dict[str, ExchangeInfo] = {}
        
        if exchange_ids is None:
            exchange_ids = ['binance', 'coinbase', 'kraken']
            
        for exchange_id in exchange_ids:
            try:
                self._setup_exchange(exchange_id)
            except Exception as e:
                self.logger.warning(f"Failed to initialize {exchange_id}: {e}")

        self.rate_limit_delay = 1.0
        self._last_request_time = 0

    def _setup_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger('CryptoMarketAnalyzer')
        logger.setLevel(log_level)
        
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger

    def _setup_exchange(self, exchange_id: str) -> None:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            self.exchanges[exchange_id] = exchange
            self.logger.info(f"Successfully initialized {exchange_id} exchange")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {exchange_id} exchange: {e}")
            raise

    async def _rate_limit_request(self):
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()

    async def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: TimeFrame,
        limit: int = 500
    ) -> pd.DataFrame:
        exchange = self.exchanges[exchange_id]
        
        await self._rate_limit_request()
        ohlcv = await asyncio.to_thread(
            exchange.fetch_ohlcv,
            symbol,
            timeframe.value,
            limit=limit
        )
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        
        return df

    async def predict_prices(self, historical_data: pd.DataFrame, days: int = 7) -> Tuple[List[str], List[float]]:
        try:
            historical_data = historical_data.rename(columns={'timestamp': 'ds', 'close': 'y'})
            historical_data['ds'] = pd.to_datetime(historical_data['ds'])
            
            model = Prophet()
            model.fit(historical_data[['ds', 'y']])

            future = model.make_future_dataframe(periods=days, freq='D')
            forecast = model.predict(future)
            future_predictions = forecast.tail(days)

            prediction_dates = future_predictions['ds'].dt.strftime('%Y-%m-%d').tolist()
            predictions = future_predictions['yhat'].tolist()
            
            self.logger.info("Price predictions generated successfully.")
            return prediction_dates, predictions

        except Exception as e:
            self.logger.error(f"Failed to generate price predictions: {e}")
            return [], []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        rsi = RSIIndicator(
            close=df['close'],
            window=self.indicators.rsi_period
        )
        df['rsi'] = rsi.rsi()

        # Bollinger Bands
        bb = BollingerBands(
            close=df['close'],
            window=self.indicators.bb_period,
            window_dev=self.indicators.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # MACD
        macd = MACD(
            close=df['close'],
            window_slow=self.indicators.macd_slow,
            window_fast=self.indicators.macd_fast,
            window_sign=self.indicators.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # ATR
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.indicators.atr_period
        )
        df['atr'] = atr.average_true_range()

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        return df

    async def fetch_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: TimeFrame,
        start_date: str,
        end_date: str,
        limit: int = 365
    ) -> pd.DataFrame:
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            raise ValueError(f"Exchange '{exchange_id}' is not initialized.")
        
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_data = []
        while start_timestamp < end_timestamp:
            await self._rate_limit_request()
            try:
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv,
                    symbol,
                    timeframe.value,
                    since=start_timestamp,
                    limit=limit
                )
                if not ohlcv:
                    break

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                all_data.append(df)

                start_timestamp = int(df['timestamp'].iloc[-1]) + 1
            except Exception as e:
                self.logger.error(f"Failed to fetch data: {e}")
                break

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            result_df['timestamp'] = pd.to_datetime(
                result_df['timestamp'], unit='ms'
            ).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            return result_df

        return pd.DataFrame()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialize signals
        df['signal'] = SignalType.NEUTRAL.value

        # RSI signals
        df.loc[df['rsi'] < 30, 'signal'] = SignalType.BUY.value
        df.loc[df['rsi'] > 70, 'signal'] = SignalType.SELL.value

        # Bollinger Bands signals
        df.loc[df['close'] < df['bb_lower'], 'signal'] = SignalType.BUY.value
        df.loc[df['close'] > df['bb_upper'], 'signal'] = SignalType.SELL.value

        # MACD signals
        df['macd_cross'] = np.where(
            df['macd'] > df['macd_signal'],
            SignalType.BUY.value,
            np.where(
                df['macd'] < df['macd_signal'],
                SignalType.SELL.value,
                SignalType.NEUTRAL.value
            )
        )

        # Volume-price divergence
        df['vol_price_divergence'] = np.where(
            (df['price_change'] > 0) & (df['volume_change'] < 0),
            'BEARISH_DIV',
            np.where(
                (df['price_change'] < 0) & (df['volume_change'] > 0),
                'BULLISH_DIV',
                'NO_DIV'
            )
        )

        # ADX for trend strength
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['adx'] = adx.adx()

        # Generate strong signals based on multiple indicators
        conditions = [
            (df['signal'] == SignalType.BUY.value) & 
            (df['vol_price_divergence'] == 'BULLISH_DIV') &
            (df['adx'] > 25),
            
            (df['signal'] == SignalType.SELL.value) &
            (df['vol_price_divergence'] == 'BEARISH_DIV') &
            (df['adx'] > 25)
        ]
        choices = [SignalType.STRONG_BUY.value, SignalType.STRONG_SELL.value]
        df['signal'] = np.select(conditions, choices, default=df['signal'])

        return df
    async def load_exchange_info(self) -> Dict[str, ExchangeInfo]:
        cache_file = self.cache_dir / "exchange_info.json"
        
        if cache_file.exists():
            try:
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 86400:  # 24 hours
                    with cache_file.open() as f:
                        cached_data = json.load(f)
                        return {
                            ex: ExchangeInfo(**info) for ex, info in cached_data.items()
                        }
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        exchange_info = {}
        for exchange_id, exchange in self.exchanges.items():
            try:
                await self._rate_limit_request()
                markets = await asyncio.to_thread(exchange.load_markets)
                
                # Filter out inactive markets if that information is available
                active_symbols = set()
                for symbol, market in markets.items():
                    # Different exchanges might have different ways to mark active markets
                    is_active = (
                        market.get('active', True) and  # Some exchanges use 'active'
                        not market.get('expired', False) and  # Some use 'expired'
                        not market.get('disabled', False)  # Some use 'disabled'
                    )
                    if is_active:
                        active_symbols.add(symbol)
                
                exchange_info[exchange_id] = ExchangeInfo(
                    name=exchange_id,
                    symbols=active_symbols,
                    timeframes=set(exchange.timeframes) if hasattr(exchange, 'timeframes') else set(),
                    has_fetchOHLCV=exchange.has.get('fetchOHLCV', False)
                )
                
                self.logger.info(f"Loaded {len(active_symbols)} active symbols from {exchange_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading {exchange_id} info: {e}")

        try:
            with cache_file.open('w') as f:
                json.dump({
                    ex: info.__dict__ for ex, info in exchange_info.items()
                }, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache exchange info: {e}")

        return exchange_info

    async def get_all_symbols(self) -> Dict[str, Set[str]]:
        """Get all available symbols for each exchange."""
        info = await self.load_exchange_info()
        return {ex_id: ex_info.symbols for ex_id, ex_info in info.items()}
    async def analyze_market(
    self,
    symbol: str,
    timeframe: TimeFrame,
    limit: int = 500,
    save_plot: bool = True,
    plot_path: Optional[str] = None,
    include_predictions: bool = True
) -> pd.DataFrame:
        """
        Perform comprehensive market analysis for a given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: TimeFrame enum value
            limit: Number of candles to analyze
            save_plot: Whether to save the analysis plot
            plot_path: Path to save the plot
            include_predictions: Whether to include price predictions
            
        Returns:
            DataFrame containing analysis results
        """
        try:
            # Fetch historical data from primary exchange (using first available)
            exchange_id = next(iter(self.exchanges.keys()))
            df = await self.fetch_ohlcv(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol} on {exchange_id}")
                
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            # Generate trading signals
            df = self.generate_signals(df)
            
            # Add price predictions if requested
            if include_predictions:
                prediction_model = EnhancedPricePrediction()
                train_results = prediction_model.train_models(df)
                
                if 'error' not in train_results:
                    pred_dates, pred_prices = prediction_model.predict(df)
                    self.prediction_dates = pred_dates
                    
                    # Add predictions to dataframe
                    pred_df = pd.DataFrame({
                        'timestamp': pd.to_datetime(pred_dates),
                        'predicted_price': pred_prices
                    })
                    df = pd.concat([
                        df,
                        pred_df.set_index('timestamp')['predicted_price']
                    ], axis=1)
            
            # Generate and save plot if requested
            if save_plot:
                self.plot_analysis(
                    df=df,
                    symbol=symbol,
                    save_path=plot_path
                )
            
            # Add market statistics
            df['volatility'] = df['close'].pct_change().rolling(window=14).std()
            df['trend_strength'] = df['adx']
            df['volume_intensity'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
            
            # Calculate support and resistance levels
            price_max = df['high'].rolling(window=20).max()
            price_min = df['low'].rolling(window=20).min()
            df['resistance'] = price_max
            df['support'] = price_min
            
            # Add market summary
            latest = df.iloc[-1]
            self.logger.info(f"""
            Market Analysis Summary for {symbol}:
            Current Price: ${latest['close']:.2f}
            Signal: {latest['signal']}
            RSI: {latest['rsi']:.2f}
            Trend Strength (ADX): {latest['adx']:.2f}
            Volatility (14-period): {latest['volatility']:.4f}
            Volume Intensity: {latest['volume_intensity']:.2f}
            """)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            raise
    def plot_analysis(
        self,
        df: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None
    ) -> None:
        # Use default style instead of seaborn
        plt.style.use('default')
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
        
        # Format dates for x-axis
        date_format = '%Y-%m-%d %H:%M IST'
        
        # Price and Bollinger Bands
        ax1.plot(df['timestamp'], df['close'], label='Price', linewidth=1.5)
        ax1.plot(df['timestamp'], df['bb_upper'], 'r--', label='BB Upper', alpha=0.7)
        ax1.plot(df['timestamp'], df['bb_middle'], 'g--', label='BB Middle', alpha=0.7)
        ax1.plot(df['timestamp'], df['bb_lower'], 'r--', label='BB Lower', alpha=0.7)
        
        # Plot signals
        buy_signals = df[df['signal'].isin([SignalType.BUY.value, SignalType.STRONG_BUY.value])]
        sell_signals = df[df['signal'].isin([SignalType.SELL.value, SignalType.STRONG_SELL.value])]
        
        ax1.scatter(buy_signals['timestamp'], buy_signals['close'],
                   marker='^', color='g', label='Buy Signal', s=100)
        ax1.scatter(sell_signals['timestamp'], sell_signals['close'],
                   marker='v', color='r', label='Sell Signal', s=100)
        
        ax1.set_title(f'{symbol} Price Analysis (IST)', pad=20, fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45)
        
        # RSI
        ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df['timestamp'], 70, 30, alpha=0.1, color='gray')
        ax2.set_title('Relative Strength Index (RSI)', pad=20, fontsize=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # MACD
        ax3.plot(df['timestamp'], df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax3.plot(df['timestamp'], df['macd_signal'], label='Signal', color='orange', linewidth=1.5)
        colors = ['g' if x >= 0 else 'r' for x in df['macd_histogram']]
        ax3.bar(df['timestamp'], df['macd_histogram'], label='Histogram', color=colors, alpha=0.5)
        ax3.set_title('Moving Average Convergence Divergence (MACD)', pad=20, fontsize=10)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Add timestamp to indicate last update
        last_update = df['timestamp'].iloc[-1].strftime(date_format)
        fig.text(0.99, 0.01, f'Last Updated: {last_update}', 
                ha='right', va='bottom', fontsize=8, style='italic')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

        # Volume-price divergence
        df['vol_price_divergence'] = np.where(
            (df['price_change'] > 0) & (df['volume_change'] < 0),
            'BEARISH_DIV',
            np.where(
                (df['price_change'] < 0) & (df['volume_change'] > 0),
                'BULLISH_DIV',
                'NO_DIV'
            )
        )

        # ADX for trend strength
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['adx'] = adx.adx()

        # Generate strong signals based on multiple indicators
        conditions = [
            (df['signal'] == SignalType.BUY.value) & 
            (df['vol_price_divergence'] == 'BULLISH_DIV') &
            (df['adx'] > 25),
            
            (df['signal'] == SignalType.SELL.value) &
            (df['vol_price_divergence'] == 'BEARISH_DIV') &
            (df['adx'] > 25)
        ]
        choices = [SignalType.STRONG_BUY.value, SignalType.STRONG_SELL.value]
        df['signal'] = np.select(conditions, choices, default=df['signal'])
    async def get_common_symbols(self, min_exchanges: int = None) -> Dict[str, Set[str]]:
        """
        Get symbols that are common across exchanges.
        
        Args:
            min_exchanges: Minimum number of exchanges a symbol must be present on.
                        If None, symbol must be present on all exchanges.
        
        Returns:
            Dict mapping symbols to the set of exchanges they're available on
        """
        info = await self.load_exchange_info()
        if not info:
            return {}
        
        # Create a dictionary mapping symbols to the exchanges they're available on
        symbol_exchanges = {}
        for exchange_id, exchange_info in info.items():
            for symbol in exchange_info.symbols:
                if symbol not in symbol_exchanges:
                    symbol_exchanges[symbol] = set()
                symbol_exchanges[symbol].add(exchange_id)
        
        # Filter based on minimum exchange requirement
        if min_exchanges is None:
            min_exchanges = len(info)  # Must be on all exchanges
        
        common_symbols = {
            symbol: exchanges
            for symbol, exchanges in symbol_exchanges.items()
            if len(exchanges) >= min_exchanges
        }
        
        return common_symbols
    # Helper method to display symbol availability
    def print_symbol_availability(self, common_symbols: Dict[str, Set[str]]) -> None:
        """Print detailed information about symbol availability across exchanges."""
        if not common_symbols:
            print("\n❌ No symbols found matching the criteria.")
            return

        # Sort symbols by number of exchanges they're available on
        sorted_symbols = sorted(
            common_symbols.items(),
            key=lambda x: (len(x[1]), x[0]),
            reverse=True
        )
        
        print(f"\n📊 Found {len(common_symbols)} symbols matching criteria:")
        print("\nTop 10 most widely available symbols:")
        for symbol, exchanges in sorted_symbols[:10]:
            print(f"- {symbol:<12} Available on {len(exchanges)} exchanges: {', '.join(sorted(exchanges))}")
        
        # Group by number of exchanges
        availability_groups = {}
        for symbol, exchanges in common_symbols.items():
            count = len(exchanges)
            if count not in availability_groups:
                availability_groups[count] = 0
            availability_groups[count] += 1
        
        print("\nAvailability Summary:")
        for count in sorted(availability_groups.keys(), reverse=True):
            print(f"- {availability_groups[count]} symbols available on {count} exchanges")

    async def select_symbol(self, min_exchanges: int = None) -> Tuple[str, Set[str]]:
        """
        Interactive symbol selection with detailed availability information.
        
        Args:
            min_exchanges: Minimum number of exchanges required for a symbol
        
        Returns:
            Tuple of (selected symbol, set of exchanges supporting it)
        """
        while True:
            # First show all available symbols
            all_symbols = await self.get_all_symbols()
            print("\n📈 Symbol Availability Options:")
            print("1. Show all symbols from any exchange")
            print("2. Show symbols available on all exchanges")
            print(f"3. Show symbols available on at least 2 exchanges")
            
            choice = input("\nSelect option (1-3) [default=1]: ").strip() or "1"
            
            if choice == "1":
                min_exchanges = 1
            elif choice == "2":
                min_exchanges = len(self.exchanges)
            elif choice == "3":
                min_exchanges = 2
            else:
                print("❌ Invalid choice, defaulting to option 1")
                min_exchanges = 1
                
            common_symbols = await self.get_common_symbols(min_exchanges)
            self.print_symbol_availability(common_symbols)
            
            # Symbol selection
            symbol = input("\n📈 Enter the symbol to analyze (e.g., BTC/USDT) or 'back' to change filter: ").strip().upper()
            
            if symbol.lower() == 'back':
                continue
            if symbol.lower() == 'exit':
                sys.exit(0)
                
            if symbol in common_symbols:
                return symbol, common_symbols[symbol]
            else:
                # Check if it exists in any exchange
                all_exchanges = set().union(*[symbols for symbols in all_symbols.values()])
                if symbol in all_exchanges:
                    supporting_exchanges = {
                        exchange for exchange, symbols in all_symbols.items()
                        if symbol in symbols
                    }
                    return symbol, supporting_exchanges
                
                print(f"❌ Symbol {symbol} not found. Please try another symbol.")


async def main():
    """Main function to run the crypto market analyzer with enhanced user interaction."""
    import asyncio
    import sys
    from datetime import datetime, timedelta

    # Initialize the analyzer
    try:
        analyzer = CryptoMarketAnalyzer(
            exchange_ids=['binance', 'coinbase', 'kraken'],
            indicators=TechnicalIndicators(
                rsi_period=14,
                bb_period=20,
                macd_fast=12,
                macd_slow=26
            ),
            log_level=logging.INFO
        )
        symbol, supporting_exchanges = await analyzer.select_symbol()
        print(f"\n✅ Selected {symbol}, available on: {', '.join(sorted(supporting_exchanges))}")
        print("\n🚀 Crypto Market Analyzer initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize analyzer: {e}")
        sys.exit(1)

    while True:
        try:
            print("\n🔍 Symbol Search Options:")
            print("1. View all available symbols")
            print("2. Search by symbol name")
            print("3. Show most traded symbols")
            print("4. Exit")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '4':
                print("\n👋 Goodbye!")
                sys.exit(0)

            if choice == '1':
                # Get all symbols from each exchange
                all_symbols = await analyzer.get_all_symbols()
                print("\n📊 Available Symbols by Exchange:")
                for exchange, symbols in all_symbols.items():
                    print(f"\n{exchange.upper()}:")
                    # Sort and display symbols in columns
                    sorted_symbols = sorted(list(symbols))
                    for i in range(0, len(sorted_symbols), 5):
                        print("  ".join(f"{s:<12}" for s in sorted_symbols[i:i+5]))
                    print(f"Total: {len(symbols)} symbols")

            elif choice == '2':
                search_term = input("\nEnter search term (e.g., 'BTC' or 'ETH'): ").strip().upper()
                all_symbols = await analyzer.get_all_symbols()
                
                matching_symbols = {}
                for exchange, symbols in all_symbols.items():
                    matches = {s for s in symbols if search_term in s}
                    if matches:
                        matching_symbols[exchange] = matches

                if matching_symbols:
                    print(f"\n🔍 Found matching symbols:")
                    for exchange, symbols in matching_symbols.items():
                        print(f"\n{exchange.upper()}:")
                        for symbol in sorted(symbols):
                            print(f"  {symbol}")
                else:
                    print(f"\n❌ No symbols found matching '{search_term}'")
                    continue

            elif choice == '3':
                print("\nFetching most traded symbols...")
                common_symbols = await analyzer.get_common_symbols(min_exchanges=2)
                analyzer.print_symbol_availability(common_symbols)

            # Symbol selection
            symbol, supporting_exchanges = await analyzer.select_symbol()
            
            # Exchange selection
            if len(supporting_exchanges) > 1:
                print(f"\n📈 {symbol} is available on multiple exchanges:")
                for i, exchange in enumerate(sorted(supporting_exchanges), 1):
                    print(f"{i}. {exchange}")
                
                while True:
                    exchange_choice = input(f"\nSelect exchange (1-{len(supporting_exchanges)}) [default=1]: ").strip() or "1"
                    try:
                        selected_exchange = sorted(supporting_exchanges)[int(exchange_choice) - 1]
                        break
                    except (ValueError, IndexError):
                        print("❌ Invalid choice. Please try again.")
            else:
                selected_exchange = next(iter(supporting_exchanges))

            print(f"\n📊 Selected {symbol} on {selected_exchange}")

            # Timeframe selection
            print("\n⏰ Available timeframes:")
            timeframes = {
                '1': TimeFrame.MINUTE_1,
                '2': TimeFrame.MINUTE_5,
                '3': TimeFrame.MINUTE_15,
                '4': TimeFrame.HOUR_1,
                '5': TimeFrame.HOUR_4,
                '6': TimeFrame.DAY_1
            }
            for key, tf in timeframes.items():
                print(f"{key}. {tf.value}")

            timeframe_choice = input("\nSelect timeframe (1-6) [default=4]: ").strip() or "4"
            timeframe = timeframes.get(timeframe_choice, TimeFrame.HOUR_1)

            # Analysis options
            print("\n⚙️ Analysis Configuration:")
            include_predictions = input("Include price predictions? (y/n) [default=y]: ").strip().lower() != 'n'
            save_plot = input("Save analysis plot? (y/n) [default=y]: ").strip().lower() != 'n'
            save_csv = input("Save analysis results to CSV? (y/n) [default=y]: ").strip().lower() != 'n'

            # Execute analysis
            print(f"\n🔄 Analyzing {symbol} on {selected_exchange} with {timeframe.value} timeframe...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"analysis_{symbol.replace('/', '_')}_{timestamp}.png" if save_plot else None
            csv_path = f"analysis_{symbol.replace('/', '_')}_{timestamp}.csv" if save_csv else None

            analysis_results = await analyzer.analyze_market(
                symbol=symbol,
                timeframe=timeframe,
                limit=500,
                save_plot=save_plot,
                plot_path=plot_path,
                include_predictions=include_predictions
            )

            print("\n✅ Analysis completed!")

            # Display results
            latest = analysis_results.iloc[-1]
            print("\n📊 Current Market Status:")
            print(f"Price: ${latest['close']:.2f}")
            print(f"Signal: {latest['signal']}")
            print(f"RSI: {latest['rsi']:.2f}")
            print(f"Volume: {latest['volume']:.2f}")

            if save_csv and csv_path:
                print(f"\n💾 Analysis saved to: {csv_path}")
            if save_plot and plot_path:
                print(f"📊 Plot saved to: {plot_path}")

            # Ask if user wants to analyze another symbol
            if input("\nAnalyze another symbol? (y/n) [default=y]: ").strip().lower() == 'n':
                print("\n👋 Thanks for using Crypto Market Analyzer!")
                break

        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if input("\nTry again? (y/n) [default=y]: ").strip().lower() == 'n':
                break

if __name__ == "__main__":
    asyncio.run(main())
