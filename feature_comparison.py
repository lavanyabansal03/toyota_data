import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

class FeatureComparison:
    def __init__(self):
        self.file_path = r"C:\Users\Lavanya Bansal\Downloads\FuelConsumption.csv"
        self.df = None
        self.numerical_features = [
            'ENGINESIZE', 'CYLINDERS',
            'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
            'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG',
            'CO2EMISSIONS'
        ]
        self.scaler = StandardScaler()
        self.load_data()
        self.create_gui()

    def load_data(self):
        """Load and prepare the dataset"""
        try:
            # Read the CSV file
            self.df = pd.read_csv(self.file_path)
            print("Initial columns:", self.df.columns.tolist())
            
            # Ensure MODELYEAR exists and convert to numeric
            if 'MODELYEAR' in self.df.columns:
                self.df['MODELYEAR'] = pd.to_numeric(self.df['MODELYEAR'], errors='coerce')
            else:
                print("Warning: MODELYEAR column not found")
                return
            
            # Keep only required columns
            required_columns = ['MODELYEAR'] + self.numerical_features
            self.df = self.df[required_columns]
            
            # Remove rows with missing values
            self.df = self.df.dropna()
            
            # Convert all columns to numeric
            for col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Drop any remaining rows with NaN values
            self.df = self.df.dropna()
            
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            print("Columns after processing:", self.df.columns.tolist())
            
            # Update features list
            self.features = self.numerical_features
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_feature(self, target_feature):
        """Analyze feature trends and find best fuel consumption year"""
        try:
            yearly_data = []
            years = sorted(self.df['MODELYEAR'].unique())
            
            for year in years:
                year_df = self.df[self.df['MODELYEAR'] == year]
                if len(year_df) == 0:
                    continue
                    
                # Find best fuel consumption for this year
                min_fuel_idx = year_df['FUELCONSUMPTION_COMB'].idxmin()
                best_config = year_df.loc[min_fuel_idx]
                
                yearly_data.append({
                    'Year': int(year),
                    'Feature_Value': float(best_config[target_feature]),
                    'Fuel_Consumption': float(best_config['FUELCONSUMPTION_COMB']),
                    'Cylinders': int(best_config['CYLINDERS']),
                    'Engine_Size': float(best_config['ENGINESIZE']),
                    'CO2_Emissions': float(best_config['CO2EMISSIONS'])
                })
            
            if not yearly_data:
                raise ValueError("No valid data found for analysis")
                
            analysis_df = pd.DataFrame(yearly_data)
            
            # Find overall best year
            best_year_idx = analysis_df['Fuel_Consumption'].idxmin()
            best_year_data = analysis_df.iloc[best_year_idx]
            
            # Prepare regression data
            X = analysis_df['Year'].values.reshape(-1, 1)
            y = analysis_df['Feature_Value'].values
            
            # Scale and fit model
            X_scaled = self.scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Predict 2025
            future_X = np.array([[2025]])
            future_X_scaled = self.scaler.transform(future_X)
            prediction_2025 = model.predict(future_X_scaled)[0]
            
            # Calculate metrics
            y_pred = model.predict(X_scaled)
            metrics = {
                'r2': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred)
            }
            
            return {
                'yearly_data': analysis_df,
                'best_year': best_year_data,
                'metrics': metrics,
                'prediction_2025': prediction_2025
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise

    def plot_analysis(self, target_feature):
        """Create visualization of the analysis"""
        try:
            results = self.analyze_feature(target_feature)
            df = results['yearly_data']
            best_year = results['best_year']
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot 1: Feature trend over years
            ax1.plot(df['Year'], df['Feature_Value'], 'b-', marker='o', label=target_feature)
            ax1.scatter(best_year['Year'], best_year['Feature_Value'], 
                       color='red', s=100, label='Best Year')
            
            # Add 2025 prediction
            ax1.scatter(2025, results['prediction_2025'], 
                       color='green', s=100, label='2025 Prediction')
            ax1.plot([df['Year'].iloc[-1], 2025], 
                    [df['Feature_Value'].iloc[-1], results['prediction_2025']], 
                    'g--')
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel(target_feature)
            ax1.set_title(f'{target_feature} Trend Over Years')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Feature vs Fuel Consumption
            scatter = ax2.scatter(df['Feature_Value'], df['Fuel_Consumption'], 
                                c=df['Year'], cmap='viridis', s=100)
            ax2.scatter(best_year['Feature_Value'], best_year['Fuel_Consumption'],
                       color='red', s=150, label='Best Performance', zorder=5)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax2, label='Year')
            
            # Add annotations for each point
            for _, row in df.iterrows():
                ax2.annotate(
                    f"Year: {int(row['Year'])}\n"
                    f"Cyl: {int(row['Cylinders'])}\n"
                    f"Eng: {row['Engine_Size']:.1f}",
                    (row['Feature_Value'], row['Fuel_Consumption']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    fontsize=8
                )
            
            ax2.set_xlabel(target_feature)
            ax2.set_ylabel('Fuel Consumption')
            ax2.set_title(f'Relationship between {target_feature} and Fuel Consumption')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis results
            print(f"\nAnalysis for {target_feature}")
            print("\nYearly Performance:")
            print(df.to_string(index=False))
            
            print(f"\nBest Performance (Year {int(best_year['Year'])}):")
            print(f"{target_feature}: {best_year['Feature_Value']:.2f}")
            print(f"Fuel Consumption: {best_year['Fuel_Consumption']:.2f}")
            print(f"Cylinders: {int(best_year['Cylinders'])}")
            print(f"Engine Size: {best_year['Engine_Size']:.2f}")
            print(f"CO2 Emissions: {best_year['CO2_Emissions']:.2f}")
            
            print(f"\n2025 Prediction for {target_feature}: {results['prediction_2025']:.2f}")
            
            print("\nModel Performance Metrics:")
            print(f"R² Score: {results['metrics']['r2']:.4f}")
            print(f"RMSE: {results['metrics']['rmse']:.4f}")
            print(f"MAE: {results['metrics']['mae']:.4f}")
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            raise

    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Feature Analysis Tool")
        self.root.geometry("400x200")

        tk.Label(self.root, text="Select Feature to Analyze:").pack(pady=10)
        self.feature_var = tk.StringVar()
        feature_dropdown = ttk.Combobox(self.root, textvariable=self.feature_var)
        feature_dropdown['values'] = self.features
        feature_dropdown.pack(pady=10)
        feature_dropdown.set(self.features[0])

        def analyze_wrapper():
            selected_feature = self.feature_var.get()
            if not selected_feature:
                messagebox.showerror("Error", "Please select a feature!")
                return
            try:
                self.plot_analysis(selected_feature)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                print(f"Error details: {e}")

        analyze_button = tk.Button(self.root, text="Analyze Feature",
                                 command=analyze_wrapper)
        analyze_button.pack(pady=20)

        self.root.mainloop()

if __name__ == "__main__":
    app = FeatureComparison() 