# CS771-Machine-learning-Project
Implementation of Multi-level PUF (ML-PUF) security analysis and delay recovery for Arbiter PUFs. The project demonstrates that ML-PUFs can still be modeled using linear learning techniques and implements delay recovery algorithms for Arbiter PUFs.

# Problem Statement 

This project addresses two key challenges in hardware security:

1. Multi-level PUF (ML-PUF) Security Analysis – ML PUF is a variant of arbiter PUF designed to resist machine learning attacks. The task is to prove that linear models can still accurately predict ML-PUF responses despite the added complexity of cross connections and XOR operations.

2. Delay Recovery in Arbiter PUFs – Given a linear model representation of an Arbiter PUF, recover a valid set of non-negative delays (𝑝𝑖, 𝑞𝑖, 𝑟𝑖, 𝑠𝑖) that generate the same model, despite underdetermined constraints.

# Objective 
- Analyze and model ML PUF challenge-response behavior using linear learning techniques.
- Demonstrate that ML PUF responses are still predictable despite added complexity.
- Implement algorithms to recover feasible delay parameters from given linear models for 64-bit Arbiter PUFs.

# Datasets
- ML-PUF: 8-bit challenges with corresponding 1-bit responses (6400 train CRPs, 1600 test CRPs).
- Arbiter PUF: 10 linear models, each with a 64-dimensional weight vector and a bias term.

# Background

## Arbiter PUFs
An Arbiter PUF consists of a chain of multiplexers that either swap or maintain signal lines based on the challenge bits. The time difference between upper and lower signal paths determines the response bit through an arbiter (typically a flip-flop).

## Multi-level PUF (ML-PUF)
ML-PUF uses two Arbiter PUFs (PUF0 and PUF1) with their own set of multiplexers. Unlike traditional PUFs:
- Lower signals from both PUFs compete to generate Response0 via Arbiter0
- Upper signals from both PUFs compete to generate Response1 via Arbiter1
- The final response is the XOR of Response0 and Response1

This architecture was designed with the hypothesis that cross-connections and XOR operations would make ML attacks more difficult.

# Methodology

## Problem 1: ML-PUF Security Analysis

1. **Feature Engineering**: Transform the 8-bit challenges into an appropriate feature representation that captures the ML-PUF's behavior
2. **Linear Model Development**: Design and train linear models to predict the XOR-ed responses
3. **Model Evaluation**: Test the model's accuracy on unseen challenge-response pairs

## Problem 2: Delay Recovery

1. **Mathematical Formulation**: Express the relationship between delays (p_i, q_i, r_i, s_i) and the model parameters (w, b)
2. **Constraint Satisfaction**: Solve for non-negative delays that satisfy the underdetermined system of equations
3. **Verification**: Ensure the recovered delays generate the same linear model

# Implementation

## ML-PUF Modeling
- Feature transformation techniques to create linearly separable data
- Linear classification models (SVM/Logistic Regression)
- Hyperparameter tuning and cross-validation

## Delay Recovery
- Linear programming approach to find feasible solutions
- Handling the underdetermined system with 256 unknowns and 65 equations
- Ensuring non-negativity constraints on all delay parameters

# Results

## ML-PUF Security Analysis
- Achieved prediction accuracy on test data
- Comparison with baseline models
- Analysis of feature importance

## Delay Recovery
- Successfully recovered valid delay parameters for all 10 provided models
- Verification that recovered delays generate the same linear models
- Analysis of delay distributions

# Conclusion

This project demonstrates that:
1. Despite the added complexity of ML-PUFs, they remain vulnerable to machine learning attacks using appropriate feature engineering and linear models
2. It is possible to recover valid delay parameters from a given linear model of an Arbiter PUF, though these parameters are not unique

This project was completed as part of the CS771 Machine Learning course.
