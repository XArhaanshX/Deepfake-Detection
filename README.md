# Deepfake Detection

A deepfake detection system built using a hybrid **CNN + GNN pipeline**, deployed through an interactive **Streamlit interface** for real-time testing and visualization.

This project explores combining spatial feature extraction from convolutional networks with relational structure modeling via graph neural networks to improve robustness in detecting manipulated media.

## Overview

Deepfake generation techniques are becoming increasingly sophisticated, making detection challenging when relying on pixel-level cues alone.

This system approaches the problem by:

1. Using CNNs to extract visual features from frames/images  
2. Constructing graph representations from structural relationships  
3. Applying GNNs to reason over those relationships  
4. Serving predictions through a Streamlit web interface  

## Features

- Hybrid CNN + GNN architecture
- Interactive web interface via Streamlit
- Modular model definitions
- Lightweight deployment for local experimentation

## Installation

Clone or download the repository and install dependencies:

```
pip install torch torchvision numpy opencv-python streamlit
```

## Running the Application

Launch the interface with:

```
python -m streamlit run app.py
```

This opens the browser UI where you can run inference and explore the model.

## Project Structure

```
app.py                     # Streamlit interface
model_definitions.py       # Core architecture definitions
my_models.py               # Model variants / utilities
```

## Architecture Summary

### Convolutional Neural Networks (CNN)
Used for:

- Local feature extraction
- Texture inconsistencies
- Artifact detection

### Graph Neural Networks (GNN)
Used for:

- Modeling relational structure
- Capturing spatial dependencies
- Enhancing robustness beyond pixel cues

## Future Improvements

- Training pipeline automation
- Benchmark evaluation metrics
- Video sequence modeling
- Web3 / blockchain traceability integration
- Model compression for real-time deployment


## Author

Arhaansh Jhingan — exploring ML systems,swarm learning and applied deep learning architectures.
