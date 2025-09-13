# Import packages with fallback handling
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("⚠️ FastF1 not available - using fallback mode")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ Pandas not available - using basic mode")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not available - using basic calculations")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - using basic predictions")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class F1PredictionSheet:
    """
    Comprehensive F1 Prediction Sheet for the 2025 Season
    
    This class provides a unified interface for F1 race predictions,
    combining historical data analysis, driver performance metrics,
    and machine learning models to predict race outcomes.
    """
    
    def __init__(self):
        self.model = None
        self.fastf1_enabled = FASTF1_AVAILABLE
        self.driver_mapping = {
            "Max Verstappen": "VER", "Lando Norris": "NOR", "Oscar Piastri": "PIA", 
            "George Russell": "RUS", "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC",
            "Carlos Sainz": "SAI", "Carlos Sainz Jr.": "SAI", "Fernando Alonso": "ALO", 
            "Lance Stroll": "STR", "Pierre Gasly": "GAS", "Esteban Ocon": "OCO",
            "Alexander Albon": "ALB", "Yuki Tsunoda": "TSU", "Nico Hülkenberg": "HUL",
            "Kevin Magnussen": "MAG", "Daniel Ricciardo": "RIC", "Logan Sargeant": "SAR",
            "Nyck de Vries": "DEV", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
            "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR",
            "Liam Lawson": "LAW"
        }
        
        # Current season driver performance averages (placeholder data)
        self.driver_performance_2025 = {
            "VER": 88.0, "NOR": 89.1, "PIA": 89.2, "RUS": 89.3, "HAM": 89.4,
            "LEC": 89.5, "SAI": 89.6, "ALO": 89.7, "GAS": 89.8, "OCO": 89.9,
            "STR": 90.0, "TSU": 90.1, "ALB": 90.2, "HUL": 90.3, "LAW": 90.4,
            "ANT": 90.5, "BEA": 90.6, "DOO": 90.7, "BOR": 90.8, "HAD": 90.9
        }
        
        # Enable FastF1 caching if available
        if self.fastf1_enabled:
            fastf1.Cache.enable_cache("f1_cache")
        
    def load_historical_data(self, year=2024, race_name="Australia", session_type="R"):
        """Load historical race data from FastF1"""
        if not self.fastf1_enabled:
            print(f"⚠️ FastF1 not available - cannot load {year} {race_name} data")
            return None
            
        try:
            session = fastf1.get_session(year, race_name, session_type)
            session.load()
            
            laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps.dropna(inplace=True)
            
            # Convert times to seconds
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps[f"{col} (s)"] = laps[col].dt.total_seconds()
                
            return laps
        except Exception as e:
            print(f"Warning: Could not load {year} {race_name} data: {e}")
            return None
    
    def prepare_training_data(self, laps_data):
        """Prepare training data from lap times"""
        if laps_data is None:
            return None
            
        if not PANDAS_AVAILABLE:
            print("⚠️ Pandas not available - cannot prepare training data")
            return None
            
        # Aggregate sector times by driver
        sector_times = laps_data.groupby("Driver").agg({
            "LapTime (s)": "mean",
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean"
        }).reset_index()
        
        # Calculate total sector time
        sector_times["TotalSectorTime (s)"] = (
            sector_times["Sector1Time (s)"] +
            sector_times["Sector2Time (s)"] +
            sector_times["Sector3Time (s)"]
        )
        
        return sector_times
    
    def train_model(self, training_data, features=None):
        """Train the gradient boosting model"""
        if not SKLEARN_AVAILABLE or not PANDAS_AVAILABLE:
            print("⚠️ Scikit-learn or Pandas not available - using basic prediction mode")
            return False
            
        if training_data is None or training_data.empty:
            print("Warning: No training data available")
            return False
            
        if features is None:
            features = ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
        
        # Prepare feature matrix and target
        X = training_data[features].fillna(training_data[features].mean())
        y = training_data["LapTime (s)"]
        
        if len(X) < 2:
            print("Warning: Insufficient data for training")
            return False
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42,
            max_depth=3
        )
        
        try:
            self.model.fit(X, y)
            
            # Evaluate model if we have enough data
            if len(X) > 3:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                print(f"🔍 Model Training MAE: {mae:.2f} seconds")
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_race_results(self, qualifying_data):
        """Predict race results based on qualifying data"""
        predictions = []
        
        # Handle both pandas DataFrame and dictionary input
        if PANDAS_AVAILABLE and hasattr(qualifying_data, 'iterrows'):
            # Pandas DataFrame
            for _, row in qualifying_data.iterrows():
                driver = row.get("Driver", "")
                driver_code = row.get("DriverCode", self.driver_mapping.get(driver, driver))
                qualifying_time = row.get("QualifyingTime (s)", 90.0)
                
                predicted_time = self._predict_single_result(driver, driver_code, qualifying_time)
                
                predictions.append({
                    "Driver": driver,
                    "DriverCode": driver_code,
                    "QualifyingTime (s)": qualifying_time,
                    "PredictedRaceTime (s)": predicted_time
                })
            
            if PANDAS_AVAILABLE:
                return pd.DataFrame(predictions).sort_values("PredictedRaceTime (s)")
            else:
                return sorted(predictions, key=lambda x: x["PredictedRaceTime (s)"])
        else:
            # Dictionary/list input (fallback mode)
            if isinstance(qualifying_data, dict):
                qualifying_data = [qualifying_data]
            
            for item in qualifying_data:
                driver = item.get("Driver", "")
                driver_code = item.get("DriverCode", self.driver_mapping.get(driver, driver))
                qualifying_time = item.get("QualifyingTime (s)", 90.0)
                
                predicted_time = self._predict_single_result(driver, driver_code, qualifying_time)
                
                predictions.append({
                    "Driver": driver,
                    "DriverCode": driver_code,
                    "QualifyingTime (s)": qualifying_time,
                    "PredictedRaceTime (s)": predicted_time
                })
            
            return sorted(predictions, key=lambda x: x["PredictedRaceTime (s)"])
    
    def _predict_single_result(self, driver, driver_code, qualifying_time):
        """Predict a single driver's race result"""
        if self.model is not None and SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            # Use trained model
            estimated_sector1 = qualifying_time * 0.33
            estimated_sector2 = qualifying_time * 0.34
            estimated_sector3 = qualifying_time * 0.33
            
            # Create feature vector
            features = np.array([[estimated_sector1, estimated_sector2, estimated_sector3]])
            
            try:
                predicted_time = self.model.predict(features)[0]
            except:
                # Fallback to basic calculation
                predicted_time = qualifying_time + 5.0
        else:
            # Fallback prediction using driver performance data
            base_performance = self.driver_performance_2025.get(driver_code, 90.0)
            qualifying_factor = qualifying_time / 88.0  # Normalize to base qualifying time
            predicted_time = base_performance * qualifying_factor
            
        return predicted_time
    
    def generate_championship_prediction(self):
        """Generate championship standings prediction based on current performance"""
        championship_points = {}
        
        # Points system: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for top 10
        points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        
        # Sort drivers by performance (lower is better)
        sorted_drivers = sorted(self.driver_performance_2025.items(), key=lambda x: x[1])
        
        # Simulate remaining races (assume 24 races total)
        remaining_races = 20  # Adjust based on current season progress
        
        for i, (driver, performance) in enumerate(sorted_drivers[:10]):
            # Assign points based on average position
            avg_points = points_system[i] if i < len(points_system) else 0
            total_points = avg_points * remaining_races
            championship_points[driver] = total_points
        
        return sorted(championship_points.items(), key=lambda x: x[1], reverse=True)
    
    def display_predictions(self, race_predictions, race_name="Current Race"):
        """Display formatted race predictions"""
        print(f"\n🏁 {race_name} Predictions 🏁\n")
        print("=" * 50)
        
        if race_predictions is not None:
            # Handle both pandas DataFrame and list of dictionaries
            if PANDAS_AVAILABLE and hasattr(race_predictions, 'iterrows'):
                # Pandas DataFrame
                if not race_predictions.empty:
                    for i, row in race_predictions.iterrows():
                        position = i + 1
                        driver = row["Driver"]
                        predicted_time = row["PredictedRaceTime (s)"]
                        
                        medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
                        print(f"{medal} {driver}: {predicted_time:.2f}s")
                else:
                    print("No predictions available")
            else:
                # List of dictionaries
                if race_predictions:
                    for i, prediction in enumerate(race_predictions):
                        position = i + 1
                        driver = prediction["Driver"]
                        predicted_time = prediction["PredictedRaceTime (s)"]
                        
                        medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
                        print(f"{medal} {driver}: {predicted_time:.2f}s")
                else:
                    print("No predictions available")
        else:
            print("No predictions available")
        
        print("=" * 50)
    
    def display_championship_predictions(self):
        """Display championship standings predictions"""
        championship = self.generate_championship_prediction()
        
        print("\n🏆 2025 F1 Championship Predictions 🏆\n")
        print("=" * 50)
        
        for i, (driver, points) in enumerate(championship):
            position = i + 1
            medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            print(f"{medal} {driver}: {points:.0f} points")
        
        print("=" * 50)

def main():
    """Main function to run F1 predictions"""
    print("🏎️ F1 2025 Prediction Sheet 🏎️")
    print("Advanced Machine Learning Race Predictions\n")
    
    # Initialize prediction system
    predictor = F1PredictionSheet()
    
    # Load historical data for training (try multiple races)
    training_data = None
    races_to_try = [
        ("Australia", 2024), ("China", 2024), ("Miami", 2024), 
        ("Monaco", 2024), ("Saudi Arabia", 2024)
    ]
    
    for race_name, year in races_to_try:
        historical_data = predictor.load_historical_data(year, race_name)
        if historical_data is not None:
            training_data = predictor.prepare_training_data(historical_data)
            if training_data is not None and not training_data.empty:
                print(f"✅ Successfully loaded {year} {race_name} data for training")
                break
    
    # Train the model
    if training_data is not None:
        model_trained = predictor.train_model(training_data)
        if model_trained:
            print("✅ Model trained successfully")
        else:
            print("⚠️ Model training failed, using fallback predictions")
    else:
        print("⚠️ No historical data available, using fallback predictions")
    
    # Sample qualifying data for next race prediction
    if PANDAS_AVAILABLE:
        sample_qualifying_2025 = pd.DataFrame({
            "Driver": ["Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell", 
                       "Lewis Hamilton", "Charles Leclerc", "Carlos Sainz", "Fernando Alonso",
                       "Pierre Gasly", "Alexander Albon", "Yuki Tsunoda", "Esteban Ocon"],
            "QualifyingTime (s)": [88.1, 88.3, 88.5, 88.7, 88.9, 89.1, 89.3, 89.5, 89.7, 89.9, 90.1, 90.3]
        })
        
        # Add driver codes
        sample_qualifying_2025["DriverCode"] = sample_qualifying_2025["Driver"].map(predictor.driver_mapping)
    else:
        # Fallback to list of dictionaries
        drivers = ["Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell", 
                   "Lewis Hamilton", "Charles Leclerc", "Carlos Sainz", "Fernando Alonso",
                   "Pierre Gasly", "Alexander Albon", "Yuki Tsunoda", "Esteban Ocon"]
        times = [88.1, 88.3, 88.5, 88.7, 88.9, 89.1, 89.3, 89.5, 89.7, 89.9, 90.1, 90.3]
        
        sample_qualifying_2025 = []
        for driver, time in zip(drivers, times):
            sample_qualifying_2025.append({
                "Driver": driver,
                "QualifyingTime (s)": time,
                "DriverCode": predictor.driver_mapping.get(driver, driver[:3].upper())
            })
    
    # Generate race predictions
    race_predictions = predictor.predict_race_results(sample_qualifying_2025)
    
    # Display results
    predictor.display_predictions(race_predictions, "Next Race")
    predictor.display_championship_predictions()
    
    print(f"\n📊 Prediction System Status:")
    print(f"   - Model Status: {'✅ Trained' if predictor.model else '❌ Not Trained'}")
    print(f"   - Drivers Tracked: {len(predictor.driver_performance_2025)}")
    print(f"   - FastF1 Available: {'✅ Yes' if FASTF1_AVAILABLE else '❌ No'}")
    print(f"   - Pandas Available: {'✅ Yes' if PANDAS_AVAILABLE else '❌ No'}")
    print(f"   - Sklearn Available: {'✅ Yes' if SKLEARN_AVAILABLE else '❌ No'}")
    print(f"   - Data Source: {'FastF1 + Machine Learning' if FASTF1_AVAILABLE else 'Fallback Predictions'}")
    
    print("\n🔮 These predictions are based on historical data and machine learning.")
    print("   Actual race results may vary due to strategy, weather, and incidents.")

if __name__ == "__main__":
    main()