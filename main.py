import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add src directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.Decision_Tree import DecisionTreeClassifier
from models.KNN import KNNClassifier
from models.Naive_Bayes import GaussianNaiveBayesClassifier
from models.SoftMax import SoftMaxClassifier
from models.Ensemble import HardVotingClassifier, StackingClassifier

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Title and description
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict the species of an Iris flower based on its measurements.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_options = {
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "SoftMax Regression": "softmax_model.pkl",
    "Hard Voting Ensemble": "hard_voting_ensemble.pkl",
    "Stacking Ensemble": "stacking_ensemble.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(model_options.keys())
)

# Load model function
@st.cache_resource
def load_model(model_path, model_type, _file_mtime):
    """Load a trained model from pickle file and reconstruct it"""
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # If it's already a model object, return it
    if hasattr(model_data, 'predict'):
        return model_data
    
    # Otherwise, reconstruct the model from saved dictionary
    if isinstance(model_data, dict):
        try:
            if model_type == "Decision Tree":
                model = DecisionTreeClassifier()
                model.load_model(model_path)
                return model
            elif model_type == "K-Nearest Neighbors":
                model = KNNClassifier()
                model.load_model(model_path)
                return model
            elif model_type == "Naive Bayes":
                model = GaussianNaiveBayesClassifier()
                model.load_model(model_path)
                return model
            elif model_type == "SoftMax Regression":
                model = SoftMaxClassifier()
                model.load_model(model_path)
                return model
            elif "Ensemble" in model_type:
                # Use static method from BaseEnsemble to properly load ensemble models
                from models.Ensemble import BaseEnsemble
                return BaseEnsemble.load_model(model_path)
        except Exception as e:
            st.error(f"Error reconstructing model: {str(e)}")
            return None
    
    return None

# Main input section
st.header("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Width of the sepal in centimeters"
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        help="Width of the petal in centimeters"
    )

# Predict button
if st.button("🔍 Predict Species", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    # Try to load and use the model
    model_file = model_options[selected_model_name]
    model_path = os.path.join("src","models", model_file)
    
    # Alternative path if models are in root
    if not os.path.exists(model_path):
        model_path = model_file
    
    # Get file modification time to bust cache when file changes
    file_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
    
    model = load_model(model_path, selected_model_name, file_mtime)
    
    if model is None:
        st.error(f"⚠️ Model file not found: {model_file}")
        st.info("Please train the model first using the notebooks in `src/notebooks/`")
        st.code(f"Expected path: {model_path}")
    else:
        try:
            # Convert DataFrame to numpy array with correct feature columns
            # Models expect feature values in the right order
            if hasattr(model, 'feature_cols'):
                # Use model's feature columns if available
                X_input = input_data[model.feature_cols].values
            else:
                # Default feature order
                feature_order = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                X_input = input_data[feature_order].values
            
            # Make prediction
            prediction = model.predict(X_input)
            
            # Display prediction
            st.success("### Prediction Result")
            
            # Map prediction to species name
            # Species mapping for numeric predictions (like SoftMax)
            species_mapping = {
                0: "Iris-setosa",
                1: "Iris-versicolor", 
                2: "Iris-virginica",
                "0": "Iris-setosa",
                "1": "Iris-versicolor",
                "2": "Iris-virginica"
            }
            
            # Clean up species name
            if isinstance(prediction, (list, np.ndarray)):
                pred_value = prediction[0]
            else:
                pred_value = prediction
            
            # Convert to species name if it's a numeric index
            if pred_value in species_mapping:
                species_name = species_mapping[pred_value]
            else:
                species_name = str(pred_value)
            
            # Remove "Iris-" prefix if present
            species_display = species_name.replace("Iris-", "").title()
            
            st.markdown(f"## 🌺 **{species_display}**")
            
            # Show input summary
            with st.expander("📊 Input Summary"):
                st.dataframe(input_data, use_container_width=True)
            
            # Species information
            species_info = {
                "Setosa": "Iris Setosa is characterized by short petals and relatively wide sepals.",
                "Versicolor": "Iris Versicolor has medium-sized petals and sepals.",
                "Virginica": "Iris Virginica typically has the longest petals and sepals."
            }
            
            species_key = species_display
            if species_key in species_info:
                st.info(f"ℹ️ {species_info[species_key]}")
                
        except Exception as e:
            error_msg = str(e)
            
            # Check for specific error types
            if "meta-classifier" in error_msg.lower() or "fit_meta" in error_msg.lower():
                st.error("❌ This Stacking Ensemble model is incomplete")
                st.warning("The meta-classifier wasn't trained when this model was saved.")
                st.info("💡 Please use **Hard Voting Ensemble** instead, or retrain the Stacking model using the training notebook.")
            else:
                st.error(f"❌ Error during prediction: {error_msg}")
                st.exception(e)

# Footer with information
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Dataset: Iris Flower Dataset | Models trained on IRIS_train.csv</small>
</div>
""", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This application uses machine learning models to classify Iris flowers into three species:
    - **Setosa**
    - **Versicolor**
    - **Virginica**
    
    The classification is based on four measurements of the flower.
    """)
    
    st.markdown("---")
    st.subheader("Quick Example Values")
    st.markdown("""
    **Iris Setosa:**
    - Sepal: 5.1 × 3.5 cm
    - Petal: 1.4 × 0.2 cm
    
    **Iris Versicolor:**
    - Sepal: 5.9 × 3.0 cm
    - Petal: 4.2 × 1.5 cm
    
    **Iris Virginica:**
    - Sepal: 6.5 × 3.0 cm
    - Petal: 5.5 × 2.0 cm
    """)
