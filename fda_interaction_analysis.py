import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class DrugInteractionAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.clean_data = None
        self.model = None

    def load_and_clean_data(self):
        print("Loading and cleaning FDA Pharmacokinetic Data...")
        df = pd.read_csv("PK Studies.csv")
        
        df.columns = df.columns.str.strip()
        
        interactions = []
        
        # The FDA dataset is in a "wide" format. 
        # We need to loop through the rows and extract every interacting drug pair.
        for index, row in df.iterrows():
            main_drug = row.get('Generic Name')
            drug_class = row.get('Class')
            
            # The dataset has up to 16 interacting drugs per row.
            # Pandas renames duplicate columns like 'Drug 1', 'Drug 2', etc.
            # and 'AUCR', 'AUCR.1', 'AUCR.2', etc.
            for i in range(1, 17):
                drug_col = f'Drug {i}'
                type_col = 'Type' if i == 1 else f'Type.{i-1}'
                aucr_col = 'AUCR' if i == 1 else f'AUCR.{i-1}'
                
                if drug_col in df.columns and pd.notna(row[drug_col]):
                    interacting_drug = row[drug_col]
                    interaction_type = row.get(type_col, 'Unknown')
                    aucr_value = row.get(aucr_col, np.nan)
                    
                    try:
                        aucr_value = float(aucr_value)
                    except:
                        aucr_value = np.nan
                        
                    if pd.notna(aucr_value) and aucr_value > 0:
                        interactions.append({
                            'Main_Drug': main_drug,
                            'Interacting_Drug': interacting_drug,
                            'Main_Class': drug_class,
                            'Interaction_Type': interaction_type,
                            'AUCR': aucr_value
                        })
                        
        self.clean_data = pd.DataFrame(interactions)
        print(f"Extraction complete. Found {len(self.clean_data)} valid interaction pairs.")
        return self.clean_data

    def generate_network_graph(self, min_aucr=2.0):
        """
        Creates a network graph of severe drug interactions.
        min_aucr: Only plot interactions that double the exposure or more.
        """
        print(f"\nGenerating Network Graph for severe interactions (AUCR >= {min_aucr})...")
        
        if self.clean_data is None:
            print("Data not loaded. Call load_and_clean_data() first.")
            return

        severe_interactions = self.clean_data[self.clean_data['AUCR'] >= min_aucr]
        
        G = nx.DiGraph()
        
        for index, row in severe_interactions.iterrows():
            G.add_edge(row['Main_Drug'], row['Interacting_Drug'], weight=row['AUCR'])
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        d = dict(G.degree)
        node_sizes = [v * 100 + 500 for v in d.values()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1.5, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold')
        
        plt.title(f"High-Risk FDA Drug Interactions (AUCR >= {min_aucr})", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('interaction_network.png', dpi=300)
        print("Network graph saved as 'interaction_network.png'.")
        plt.show()

    def train_predictive_model(self):
        """
        Trains a Random Forest to predict the severity of the interaction (AUCR)
        based on the Drug Class and Interaction Type.
        """
        print("\nTraining Predictive Machine Learning Model...")
        
        df_ml = self.clean_data.copy()
        
        df_ml.dropna(subset=['Main_Class', 'Interaction_Type', 'AUCR'], inplace=True)
        
        le_class = LabelEncoder()
        le_type = LabelEncoder()
        
        df_ml['Class_Encoded'] = le_class.fit_transform(df_ml['Main_Class'])
        df_ml['Type_Encoded'] = le_type.fit_transform(df_ml['Interaction_Type'])
        
        X = df_ml[['Class_Encoded', 'Type_Encoded']]
        y = df_ml['AUCR']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("--- Model Evaluation ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        
        importances = self.model.feature_importances_
        print("\nFeature Importances:")
        print(f"Drug Class: {importances[0]:.2%}")
        print(f"Interaction Type (DDI/PBPK): {importances[1]:.2%}")
        
        return self.model

if __name__ == "__main__":
    FILE_PATH = "Dataset 1.xlsx - PK Studies.csv"
    
    try:
        # Initialize the Analyzer
        analyzer = DrugInteractionAnalyzer(FILE_PATH)
        
        # Step 1: Parse and clean the wide-format FDA data
        clean_df = analyzer.load_and_clean_data()
        
        # Step 2: Generate and save a network map of the interactions
        analyzer.generate_network_graph(min_aucr=2.5)
        
        # Step 3: Train an ML model to predict AUCR severity
        analyzer.train_predictive_model()
        
    except FileNotFoundError:
        print(f"Error: Could not find '{FILE_PATH}'. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")