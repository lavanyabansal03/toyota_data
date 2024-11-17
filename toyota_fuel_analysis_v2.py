import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

class ToyotaAnalysisApp:
    def __init__(self):
        self.file_path = r"C:\Users\Lavanya Bansal\Downloads\Toyota Vehicle Fuel Economy Data (2021-2025)_hack.csv"
        self.df = None
        self.features = {
            'Vehicle Specifications': [
                'Vehicle Weight (lbs)',
                'Tank Size (gallons)'
            ],
            'Fuel Economy': [
                'MPG (City)',
                'MPG (Highway)'
            ],
            'Cost Metrics': [
                'Annual Fuel Cost ($)',
                'Cost to Fill Tank ($)'
            ]
        }
        self.selected_features = []
        self.selected_models = []
        self.scaler = StandardScaler()
        self.load_data()
        self.create_gui()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            
            # Add this debug print
            print("Actual columns in DataFrame:")
            for col in self.df.columns:
                print(f"'{col}'")
            
            # Convert Year to numeric
            self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
            
            # Split Make/Model into separate columns if needed
            self.df[['Make', 'Model']] = self.df['Make/Model'].str.split(' ', n=1, expand=True)
            
            print(f"Loaded {len(self.df)} Toyota vehicles from 2021-2025")
            print("\nAvailable Models:", self.df['Make/Model'].unique())
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_trends(self):
        if not self.selected_features or not self.selected_models:
            messagebox.showerror("Error", "Please select at least one feature and one model!")
            return
        
        try:
            results = {}
            for feature in self.selected_features:
                # Filter data for selected models
                model_data = self.df[self.df['Make/Model'].isin(self.selected_models)]
                
                # Create DataFrame with defined columns
                yearly_stats = pd.DataFrame(
                    columns=['mean', 'min', 'max', 'std', 'models', 'avg_mpg']
                )
                
                # Group by year and calculate statistics
                for year in sorted(model_data['Year'].unique()):
                    year_data = model_data[model_data['Year'] == year]
                    if not year_data.empty:
                        stats = {
                            'mean': year_data[feature].mean(),
                            'min': year_data[feature].min(),
                            'max': year_data[feature].max(),
                            'std': year_data[feature].std(),
                            'models': ', '.join(year_data['Make/Model'].unique()),
                            'avg_mpg': year_data['MPG (City)'].mean()
                        }
                        yearly_stats.loc[year] = stats
                
                if yearly_stats.empty:
                    continue
                    
                # Prepare data for prediction
                X = np.array(yearly_stats.index).reshape(-1, 1)
                y = yearly_stats['mean'].values
                
                # Fit model and predict 2025 if not in data
                model = LinearRegression()
                model.fit(X, y)
                
                if 2025 not in yearly_stats.index:
                    prediction_2025 = model.predict([[2025]])[0]
                else:
                    prediction_2025 = yearly_stats.loc[2025, 'mean']
                
                results[feature] = {
                    'stats': yearly_stats,
                    'prediction_2025': prediction_2025,
                    'trend_coefficient': model.coef_[0]
                }
            
            if not results:
                raise ValueError("No data available for selected models and features")
                
            return results
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise

    def analyze_performance(self, results, feature):
        """Analyze which models perform better for a given feature"""
        performance_summary = []
        
        # Get data for selected models
        model_data = self.df[self.df['Make/Model'].isin(self.selected_models)]
        
        # Different analysis based on feature type
        if feature in ['MPG (City)', 'MPG (Highway)']:
            # For MPG, higher is better
            avg_performance = model_data.groupby('Make/Model')[feature].mean().sort_values(ascending=False)
            best_model = avg_performance.index[0]
            performance_summary.append(f"Best {feature}: {best_model} "
                                     f"(Average: {avg_performance[best_model]:.1f})")
            
        elif feature == 'Vehicle Weight (lbs)':
            # For weight, context matters (no clear better/worse)
            avg_weights = model_data.groupby('Make/Model')[feature].mean().sort_values()
            lightest = avg_weights.index[0]
            heaviest = avg_weights.index[-1]
            performance_summary.append(f"Lightest: {lightest} ({avg_weights[lightest]:.1f} lbs)")
            performance_summary.append(f"Heaviest: {heaviest} ({avg_weights[heaviest]:.1f} lbs)")
            
        elif feature in ['Annual Fuel Cost ($)', 'Cost to Fill Tank ($)']:
            # For costs, lower is better
            avg_cost = model_data.groupby('Make/Model')[feature].mean().sort_values()
            best_model = avg_cost.index[0]
            performance_summary.append(f"Most Economical {feature}: {best_model} "
                                     f"(Average: ${avg_cost[best_model]:.2f})")
        
        elif feature == 'Tank Size (gallons)':
            # For tank size, context matters (no clear better/worse)
            avg_tank = model_data.groupby('Make/Model')[feature].mean().sort_values()
            smallest = avg_tank.index[0]
            largest = avg_tank.index[-1]
            performance_summary.append(f"Smallest Tank: {smallest} ({avg_tank[smallest]:.1f} gallons)")
            performance_summary.append(f"Largest Tank: {largest} ({avg_tank[largest]:.1f} gallons)")
        
        return performance_summary

    def create_results_window(self, results, feature):
        """Create a new window to display analysis results"""
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Analysis Results - {feature}")
        results_window.geometry("600x400")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add title
        text_widget.tag_configure('title', font=('Arial', 12, 'bold'))
        text_widget.tag_configure('subtitle', font=('Arial', 11, 'bold'))
        text_widget.tag_configure('normal', font=('Arial', 10))
        
        # Insert analysis results with formatting
        text_widget.insert(tk.END, f"Analysis for {feature}\n", 'title')
        text_widget.insert(tk.END, "="*50 + "\n\n", 'normal')
        
        # Yearly Statistics
        text_widget.insert(tk.END, "Yearly Statistics:\n", 'subtitle')
        text_widget.insert(tk.END, f"{results[feature]['stats'].to_string()}\n\n", 'normal')
        
        # Prediction
        if 2025 not in results[feature]['stats'].index:
            text_widget.insert(tk.END, "2025 Prediction:\n", 'subtitle')
            text_widget.insert(tk.END, f"{results[feature]['prediction_2025']:.2f}\n\n", 'normal')
        
        # Trend
        trend = "increasing" if results[feature]['trend_coefficient'] > 0 else "decreasing"
        text_widget.insert(tk.END, "Overall Trend:\n", 'subtitle')
        text_widget.insert(tk.END, f"{trend}\n\n", 'normal')
        
        # Performance Summary
        text_widget.insert(tk.END, "Performance Summary:\n", 'subtitle')
        performance_summary = self.analyze_performance(results, feature)
        for summary in performance_summary:
            text_widget.insert(tk.END, f"{summary}\n", 'normal')
        text_widget.insert(tk.END, "\n", 'normal')
        
        # Recommendations
        text_widget.insert(tk.END, "Recommendation:\n", 'subtitle')
        if feature in ['MPG (City)', 'MPG (Highway)']:
            best_model = performance_summary[0].split(': ')[1].split(' (')[0]
            text_widget.insert(tk.END, f"For best fuel efficiency, consider the {best_model}\n", 'normal')
        elif feature in ['Annual Fuel Cost ($)', 'Cost to Fill Tank ($)']:
            best_model = performance_summary[0].split(': ')[1].split(' (')[0]
            text_widget.insert(tk.END, f"For lowest costs, consider the {best_model}\n", 'normal')
        elif feature == 'Vehicle Weight (lbs)':
            text_widget.insert(tk.END, "Choose based on your needs:\n", 'normal')
            text_widget.insert(tk.END, "- Lighter vehicles typically offer better fuel economy\n", 'normal')
            text_widget.insert(tk.END, "- Heavier vehicles might offer more space/features\n", 'normal')
        elif feature == 'Tank Size (gallons)':
            text_widget.insert(tk.END, "Choose based on your needs:\n", 'normal')
            text_widget.insert(tk.END, "- Smaller tanks mean more frequent but less expensive fill-ups\n", 'normal')
            text_widget.insert(tk.END, "- Larger tanks mean fewer fill-ups but higher cost per fill\n", 'normal')
        
        text_widget.configure(state='disabled')  # Make text read-only

    def plot_analysis(self):
        try:
            print("Available columns in DataFrame:", self.df.columns.tolist())
            
            if not self.selected_features:
                messagebox.showerror("Error", "Please select at least one feature!")
                return
            
            results = self.analyze_trends()
            if not results:
                messagebox.showerror("Error", "No data available for analysis!")
                return
            
            n_features = len(self.selected_features)
            
            # Create figure with subplots - now 3 columns instead of 2
            fig = plt.figure(figsize=(20, 5*n_features))
            gs = fig.add_gridspec(n_features, 3)
            
            for idx, feature in enumerate(self.selected_features):
                if feature not in results:
                    continue
                    
                stats = results[feature]['stats']
                prediction = results[feature]['prediction_2025']
                
                # Trend plot (first column)
                ax1 = fig.add_subplot(gs[idx, 0])
                
                # Plot individual lines for each model
                for model in self.selected_models:
                    model_data = self.df[
                        (self.df['Make/Model'] == model) & 
                        (self.df['Year'].isin(stats.index))
                    ]
                    if not model_data.empty:
                        ax1.plot(model_data['Year'], 
                                model_data[feature], 
                                marker='o', 
                                label=model)
                
                # Remove the fill_between since we're showing individual models
                
                # Add 2025 predictions for each model if not in data
                if 2025 not in stats.index:
                    for model in self.selected_models:
                        model_data = self.df[self.df['Make/Model'] == model]
                        if not model_data.empty:
                            X = np.array(model_data['Year']).reshape(-1, 1)
                            y = model_data[feature].values
                            model_reg = LinearRegression()
                            model_reg.fit(X, y)
                            prediction = model_reg.predict([[2025]])[0]
                            ax1.scatter(2025, prediction, color='red', s=100)
                            ax1.plot([model_data['Year'].iloc[-1], 2025],
                                    [model_data[feature].iloc[-1], prediction],
                                    'r--')
                
                # Modify annotations to show individual values
                for year in stats.index:
                    year_data = self.df[
                        (self.df['Year'] == year) & 
                        (self.df['Make/Model'].isin(self.selected_models))
                    ]
                    for model in self.selected_models:
                        model_year_data = year_data[year_data['Make/Model'] == model]
                        if not model_year_data.empty:
                            ax1.annotate(
                                f"{model}:\n{model_year_data[feature].iloc[0]:.1f}",
                                (year, model_year_data[feature].iloc[0]),
                                xytext=(10, 5),
                                textcoords='offset points',
                                bbox=dict(
                                    boxstyle='round,pad=0.3',
                                    fc='yellow', 
                                    alpha=0.5
                                ),
                                fontsize=6,
                                verticalalignment='bottom'
                            )
                
                ax1.set_title(f'{feature} Trend Analysis', fontsize=10)
                ax1.set_xlabel('Year', fontsize=8)
                ax1.set_ylabel(feature, fontsize=8)
                ax1.tick_params(labelsize=8)
                ax1.legend(fontsize=8)
                ax1.grid(True)
                
                # Distribution plot (second column)
                ax2 = fig.add_subplot(gs[idx, 1])
                for year in stats.index:
                    year_data = self.df[
                        (self.df['Year'] == year) & 
                        (self.df['Make/Model'].isin(self.selected_models))
                    ][feature]
                    print(f"Year {year}, Data points: {len(year_data)}")  # Debug print
                    if not year_data.empty:
                        sns.kdeplot(data=year_data, label=f'Year {int(year)}', ax=ax2)
                
                ax2.set_title(f'{feature} Distribution by Year')
                ax2.set_xlabel(feature)
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True)
                
                # New correlation heatmap (third column)
                ax3 = fig.add_subplot(gs[idx, 2])
                
                # Select relevant features for correlation
                corr_features = [
                    'Vehicle Weight (lbs)',
                    'Tank Size (gallons)',
                    'MPG (City)',
                    'MPG (Highway)',
                    'Annual Fuel Cost ($)',
                    'Cost to Fill Tank ($)'
                ]
                
                # Add these debug prints before creating the correlation matrix
                print("\nTrying to access these features:")
                for feature in corr_features:
                    print(f"'{feature}' - Exists in DataFrame: {feature in self.df.columns}")
                
                model_data = self.df[
                    (self.df['Make/Model'].isin(self.selected_models))
                ][corr_features]
                
                # Calculate correlation matrix
                corr_matrix = model_data.corr()
                
                # Create heatmap
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f',
                    ax=ax3,
                    square=True,
                    annot_kws={'size': 6}
                )
                
                ax3.set_title('Feature Correlation Heatmap')
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=8)
            
            plt.tight_layout(h_pad=1.0, w_pad=1.0)
            plt.show()
            
            # Replace the print statements with window creation
            for feature in self.selected_features:
                if feature not in results:
                    continue
                
                # Create a new window for each feature's analysis
                self.create_results_window(results, feature)
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            raise

    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Toyota Fuel Economy Analysis (2021-2025)")
        self.root.geometry("600x800")

        # Create main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Create scrollable frame
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create frame for fixed buttons at bottom
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title_label = ttk.Label(
            scrollable_frame,
            text="Toyota Fuel Economy Analysis\n2021-2025",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=10)

        # Model Selection Frame
        model_frame = ttk.LabelFrame(scrollable_frame, text="Select Toyota Models", padding="10")
        model_frame.pack(fill=tk.X, pady=10, padx=10)

        # Add model checkbuttons
        self.model_vars = {}
        for model in sorted(self.df['Make/Model'].unique()):
            var = tk.BooleanVar()
            self.model_vars[model] = var
            ttk.Checkbutton(
                model_frame,
                text=model,
                variable=var
            ).pack(anchor=tk.W, pady=2)

        # Feature Selection Frame
        feature_frame = ttk.LabelFrame(scrollable_frame, text="Select Features", padding="10")
        feature_frame.pack(fill=tk.X, pady=10, padx=10)

        # Add feature checkbuttons
        self.feature_vars = {}
        for category, features in self.features.items():
            ttk.Label(
                feature_frame,
                text=category,
                font=('Helvetica', 10, 'bold')
            ).pack(anchor=tk.W, pady=5)
            
            for feature in features:
                var = tk.BooleanVar()
                self.feature_vars[feature] = var
                ttk.Checkbutton(
                    feature_frame,
                    text=feature,
                    variable=var
                ).pack(anchor=tk.W, pady=2)

        # Create a container frame for the button with left alignment
        button_container = ttk.Frame(button_frame)
        button_container.pack(fill=tk.X)
        
        # Analysis button with left alignment
        analyze_button = ttk.Button(
            button_container,
            text="Analyze Fuel Economy",
            command=self.analyze_wrapper,
            style='Accent.TButton'
        )
        analyze_button.pack(side=tk.LEFT, padx=20, pady=5)

        # Configure style
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))

        # Configure canvas scrolling
        self.root.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units")))
        scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self.root.mainloop()

    def update_selection(self):
        """Update the selected features and models display"""
        # Update selected features
        self.selected_features = [
            f for f, v in self.feature_vars.items() if v.get()
        ]
        if self.selected_features:
            self.feature_label.config(
                text="Selected Features:\n" + "\n".join(self.selected_features)
            )
        else:
            self.feature_label.config(text="Selected Features: None")

        # Update selected models
        self.selected_models = [
            m for m, v in self.model_vars.items() if v.get()
        ]
        if self.selected_models:
            self.model_label.config(
                text="Selected Models:\n" + "\n".join(self.selected_models)
            )
        else:
            self.model_label.config(text="Selected Models: None")

    def analyze_wrapper(self):
        """Wrapper for analysis with error handling"""
        try:
            # Update selections first
            self.selected_features = [f for f, v in self.feature_vars.items() if v.get()]
            self.selected_models = [m for m, v in self.model_vars.items() if v.get()]
            
            if not self.selected_features or not self.selected_models:
                messagebox.showerror("Error", "Please select at least one feature and one model!")
                return
            
            self.plot_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error details: {e}")

if __name__ == "__main__":
    app = ToyotaAnalysisApp() 