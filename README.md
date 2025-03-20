# Music Genre Classification with Deep Learning 🎶🤖

## Overview
This repository implements an audio classification system using deep learning on the GTZAN dataset. The project focuses on classifying 5-second audio fragments into 10 musical genres using various neural network architectures and extensive hyperparameter tuning. Experiments were tracked and analyzed using Weights & Biases to gain deep insights into model performance and optimization strategies.

**Team:**  
Ezequiel Grinblat, Luca Mazzarello, and Camila Migdal  
**Course:** TD VI: Artificial Intelligence (2nd Semester, 2024)  
**Project:** TP3 – Audio Classification

## Project Structure and Key Features
The project is organized into two main parts: repository configuration & implementation details, and a comprehensive experimental analysis.

### Implementation & Repository Details
- **Audio Preprocessing:**  
  - Normalization and transformation of raw waveforms.
  - Conversion of audio to Mel spectrograms (with decibel scaling and normalization).
  - Data augmentation techniques: time stretching, pitch shifting, adding noise, and resampling.
  - Padding or truncating audio to a fixed length.

- **Neural Network Architectures:**  
  - **MLP (Simple Multi-Layer Perceptron):** A basic architecture with three linear layers using ReLU activation.
  - **MLP2:** A configurable MLP allowing customization of the number of hidden layers and nodes, with dropout regularization.
  - **CEL (Convolutional Neural Network - Baseline):** Incorporates convolutional layers with batch normalization, pooling, and dropout, ending in dense layers with ReLU.
  - **CEL2:** Similar to CEL but uses LeakyReLU to mitigate dead neurons.
  - **CEL3:** Similar to CEL but employs ELU for smoother handling of negative values.

- **Experiment Tracking:**  
  - Integration with Weights & Biases (wandb) for logging training/validation loss, accuracy, confusion matrices, and learning curves.
  - Experiment metadata and hyperparameter configurations are saved in JSON format.

- **Repository Structure:**  
  - `Cooking.ipynb` or `TP3_clasificador_musica.ipynb`: The main notebook with implementation and experiments.
  - `best_models/`: Directory where the best models are saved.
  - `wandb/`: Directory containing experiment logs and data.
  - `.env`: File for storing the wandb API key.

### Experimental Analysis & Results
A significant part of the project is dedicated to extensive experimentation and evaluation of hyperparameters:

1. **Initial Configuration & Dataset Splitting:**  
   - Stratified splitting of the GTZAN dataset into training, validation, and test sets.
   - Implementation of utility functions for parameter counting, optimizer/scheduler setup, and regularization (L1 and L2).

2. **MLP Architecture Experiments:**  
   - **Configurations Tested:**
     - *Configuration 1:* Four dense layers (256 nodes each), learning rate 0.001, batch size 32, no dropout.
     - *Configuration 2:* Two dense layers (512 nodes each), learning rate 0.0005, batch size 32, dropout 0.5.
     - *Configuration 3:* Three dense layers (1024, 512, 256 nodes), learning rate 0.0003, batch size 32, dropout 0.3.
   - **Results:**  
     - Best accuracy observed with Configuration 2, although Configuration 1 provided a balanced trade-off between validation loss and accuracy.
     - Configuration 3 showed the poorest performance.

3. **CNN Architecture Experiments:**  
   - **Configurations Tested:**
     - *Configuration 1:* A simple CNN with one dense layer (32 nodes), learning rate 0.001, batch size 8, dropout 0.2.
     - *Configuration 2:* Two dense layers (64 nodes each), learning rate 0.0005, batch size 16, dropout 0.3.
     - *Configuration 3:* Three dense layers (128, 128, 64 nodes), learning rate 0.0005, batch size 32, dropout 0.3.
   - **Results:**  
     - Experiments conducted on both waveform and spectrogram inputs.
     - For spectrograms, Configuration 3 provided the best performance while Configuration 1 suffered from overfitting.
  
4. **Activation Functions Experiments:**  
   - **Functions Compared:** ReLU, LeakyReLU, and ELU.
   - **Findings:**  
     - LeakyReLU excelled on waveform inputs by preserving negative information.
     - For spectrogram inputs, ReLU yielded the best balance between accuracy and loss.
     - ELU produced the worst results for waveform-based experiments.

5. **Optimizers and Schedulers Experiments:**  
   - **Comparisons:** ADAM vs. SGD; Plateau vs. Cosine schedulers.
   - **Findings:**  
     - SGD with a cosine scheduler provided the best overall performance for both spectrogram and waveform inputs.
     - ADAM-Plateau was identified as the worst performing configuration in some cases.

6. **Regularization Techniques:**  
   - **Methods Tested:** L1 and L2 regularization.
   - **Results:**  
     - L2 tended towards overfitting while L1 showed signs of underfitting; overall, L1 offered a more balanced performance.

7. **Final Evaluation:**  
   - The best model (selected based on accuracy, validation loss, and overall performance) was evaluated on the test set.
   - A top-5 ranking of experiments was compiled, with the final winning model achieving an accuracy around 0.3 and a balanced validation loss.

## How to Run the Project
1. **Environment Setup:**  
   - Create a new conda environment (e.g., Python 3.9) and install the required dependencies:
     ```bash
     conda create -n TDVI python=3.9
     conda activate TDVI
     pip install -r requirements.txt
     ```
2. **Weights & Biases Configuration:**  
   - Create an account at [Weights & Biases](https://wandb.ai/).
   - Add your API key to a `.env` file in the project root:
     ```
     WANDB_API_KEY=your_api_key_here
     ```
3. **Running the Notebook:**  
   - Open the main Jupyter Notebook (`TP3_clasificador_musica.ipynb` or `Cooking.ipynb`) and run all cells sequentially.
4. **Reviewing Results:**  
   - Monitor experiment logs on Weights & Biases.
   - Check the `best_models/` directory for saved models and review JSON metadata files for hyperparameter details.

## Acknowledgements
We thank our course instructors, peers, and mentors for their invaluable guidance and support throughout this project.

Happy coding and exploring musical data with deep learning! 🎵🚀
