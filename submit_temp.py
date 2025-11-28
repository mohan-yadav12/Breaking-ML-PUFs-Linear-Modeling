import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
from submit import my_map, my_fit, my_decode

# Load data
def load_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append([int(x) for x in line.strip().split()])
    return np.array(data)

# Load the public models for delay recovery
def load_public_models(filename):
    models = []
    with open(filename) as f:
        for line in f:
            model = [float(x) for x in line.strip().split()]
            models.append(np.array(model))
    return np.array(models)

# Calculate reconstruction error for delay recovery
def calculate_reconstruction_error(original_model, recovered_delays_p, recovered_delays_q, recovered_delays_r, recovered_delays_s):
    """
    Calculate the reconstruction error between original model and recovered delays
    Based on standard PUF model reconstruction
    """
    # Reconstruct the model from recovered delays
    reconstructed = np.zeros(len(original_model))
    
    # Standard arbiter PUF reconstruction formula
    for i in range(len(original_model)):
        if i < len(recovered_delays_p):  # For weight components
            # Basic reconstruction: w_i ≈ 0.5*(p_i - q_i + r_i - s_i)
            reconstructed[i] = 0.5 * (recovered_delays_p[i] - recovered_delays_q[i] + 
                                    recovered_delays_r[i] - recovered_delays_s[i])
        else:  # For bias term (if present)
            # Bias reconstruction from delay combination
            bias_contribution = 0.1 * (np.sum(recovered_delays_p) - np.sum(recovered_delays_q) +
                                     np.sum(recovered_delays_r) - np.sum(recovered_delays_s)) / len(recovered_delays_p)
            reconstructed[i] = bias_contribution
    
    # Calculate different error metrics
    mse_error = np.mean((original_model - reconstructed)**2)
    mae_error = np.mean(np.abs(original_model - reconstructed))
    l2_error = np.linalg.norm(original_model - reconstructed)
    relative_error = l2_error / (np.linalg.norm(original_model) + 1e-10)
    
    return {
        'mse': mse_error,
        'mae': mae_error,
        'l2_norm': l2_error,
        'relative': relative_error,
        'reconstructed': reconstructed
    }

# Test delay recovery functionality using submit.py functions
def test_delay_recovery():
    # Load public models
    public_models = load_public_models('public_mod.txt')
    print(f"Loaded {len(public_models)} public models, each with {len(public_models[0])} dimensions")
    
    # Test with one of the public models itself
    test_model = public_models[0]
    print(f"\nTesting delay recovery with public model 0:")
    print(f"Model weights (first 10): {test_model[:10]}")
    print(f"Model bias: {test_model[-1]}")
    
    # Use my_decode function from submit.py
    p, q, r, s = my_decode(test_model)
    
    print(f"\nRecovered delays using my_decode:")
    print(f"p (first 10): {p[:10]}")
    print(f"q (first 10): {q[:10]}")
    print(f"r (first 10): {r[:10]}")
    print(f"s (first 10): {s[:10]}")
    
    print(f"\nDelay statistics:")
    print(f"p: min={p.min():.6f}, max={p.max():.6f}, mean={p.mean():.6f}")
    print(f"q: min={q.min():.6f}, max={q.max():.6f}, mean={q.mean():.6f}")
    print(f"r: min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}")
    print(f"s: min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")
    
    # Calculate and print reconstruction error
    error_metrics = calculate_reconstruction_error(test_model, p, q, r, s)
    print(f"\nRECONSTRUCTION ERROR ANALYSIS:")
    print(f"Mean Squared Error (MSE): {error_metrics['mse']:.8f}")
    print(f"Mean Absolute Error (MAE): {error_metrics['mae']:.8f}")
    print(f"L2 Norm Error: {error_metrics['l2_norm']:.8f}")
    print(f"Relative Error: {error_metrics['relative']:.8f}")
    
    # Print first few values of original vs reconstructed
    print(f"\nOriginal vs Reconstructed (first 10 values):")
    print(f"Original:     {test_model[:10]}")
    print(f"Reconstructed: {error_metrics['reconstructed'][:10]}")
    
    return error_metrics

# Load and split train data
train_data = load_data('public_trn.txt')
X_train = train_data[:, :8]  # First 8 columns are challenge bits
y_train = train_data[:, 8]   # 9th column is response

# Load and split test data  
test_data = load_data('public_tst.txt')
X_test = test_data[:, :8]    # First 8 columns are challenge bits
y_test = test_data[:, 8]     # 9th column is response

# Feature mapping using submit.py function
X_train_mapped = my_map(X_train)
X_test_mapped = my_map(X_test)

print("Data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"X_train_mapped: {X_train_mapped.shape}")
print(f"X_test_mapped: {X_test_mapped.shape}")

# Train model using submit.py function
print("\nTraining model using my_fit from submit.py:")
start_time = time.time()
w, b = my_fit(X_train, y_train)
train_time = time.time() - start_time

print(f"Model weights shape: {w.shape}")
print(f"Model bias shape: {b.shape}")
print(f"Model weights (first 10): {w.flatten()[:10]}")
print(f"Model bias: {b}")
print(f"Training time: {train_time:.4f}s")

# Test the trained model
start_time = time.time()
X_test_mapped = my_map(X_test)
map_time = time.time() - start_time

y_pred = (X_test_mapped @ w.T + b > 0).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
print(f"Trained model accuracy: {accuracy:.4f}")
print(f"Map time: {map_time:.4f}s")

# Extended model configs based on your analysis
model_configs = [
    # Loss function comparison (Hinge vs Squared Hinge)
    ('LinearSVC', 'l2', 1e-4, 1.0, 'hinge'),
    ('LinearSVC', 'l2', 1e-4, 1.0, 'squared_hinge'),
    
    # Tolerance analysis
    ('LinearSVC', 'l2', 1e-4, 1.0, 'hinge'),      # Low tolerance
    ('LinearSVC', 'l2', 1e-3, 1.0, 'hinge'),      # Medium tolerance  
    ('LinearSVC', 'l2', 1e-2, 1.0, 'hinge'),      # High tolerance
    ('LogisticRegression', 'l2', 1e-4, 1.0, None), # Low tolerance
    ('LogisticRegression', 'l2', 1e-3, 1.0, None), # Medium tolerance
    ('LogisticRegression', 'l2', 1e-2, 1.0, None), # High tolerance
    
    # C value analysis for LinearSVC
    ('LinearSVC', 'l2', 1e-4, 0.01, 'hinge'),     # Low C
    ('LinearSVC', 'l2', 1e-4, 1.0, 'hinge'),      # Medium C
    ('LinearSVC', 'l2', 1e-4, 100.0, 'hinge'),    # High C
    
    # C value analysis for LogisticRegression
    ('LogisticRegression', 'l2', 1e-4, 0.01, None), # Low C
    ('LogisticRegression', 'l2', 1e-4, 1.0, None),  # Medium C
    ('LogisticRegression', 'l2', 1e-4, 100.0, None), # High C
    
    # Penalty analysis
    ('LinearSVC', 'l1', 1e-4, 1.0, 'hinge'),
    ('LinearSVC', 'l2', 1e-4, 1.0, 'hinge'),
    ('LogisticRegression', 'l1', 1e-4, 1.0, None),
    ('LogisticRegression', 'l2', 1e-4, 1.0, None),
]

print("\n" + "="*80)
print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
print("="*80)

print("\n1. Loss Function Analysis (LinearSVC - Hinge vs Squared Hinge):")
print(f"{'Loss Function':<15}{'Model Train Time (s)':<20}{'Map Time (s)':<15}{'Accuracy':<10}")
print("-" * 60)

# Loss function analysis
loss_configs = [
    ('hinge', 'Hinge'),
    ('squared_hinge', 'Squared Hinge')
]

for loss, loss_name in loss_configs:
    try:
        model = LinearSVC(penalty='l2', loss=loss, tol=1e-4, C=1.0, max_iter=10000)
        
        start_time = time.time()
        model.fit(X_train_mapped, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_loss = model.predict(X_test_mapped)
        map_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred_loss)
        print(f"{loss_name:<15}{train_time:<20.4f}{map_time:<15.4f}{acc:<10.4f}")
        
    except Exception as e:
        print(f"{loss_name:<15}{'Error':<20}{'Error':<15}{'Error':<10}")

print("\n2. Performance Comparison based on the value of C:")

print("\nAnalysis for LinearSVC with different C values:")
print(f"{'C Value':<10}{'Model Train Time (s)':<20}{'Map Time (s)':<15}{'Accuracy':<10}")
print("-" * 55)

c_values = [(0.01, 'Low'), (1.0, 'Medium'), (100.0, 'High')]
for c_val, c_name in c_values:
    try:
        model = LinearSVC(penalty='l2', loss='hinge', tol=1e-4, C=c_val, max_iter=10000)
        
        start_time = time.time()
        model.fit(X_train_mapped, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_c = model.predict(X_test_mapped)
        map_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred_c)
        print(f"{c_name:<10}{train_time:<20.4f}{map_time:<15.4f}{acc:<10.4f}")
        
    except Exception as e:
        print(f"{c_name:<10}{'Error':<20}{'Error':<15}{'Error':<10}")

print("\nAnalysis for LogisticRegression with different C values:")
print(f"{'C Value':<10}{'Model Train Time (s)':<20}{'Map Time (s)':<15}{'Accuracy':<10}")
print("-" * 55)

for c_val, c_name in c_values:
    try:
        model = LogisticRegression(penalty='l2', tol=1e-4, C=c_val, solver='liblinear', max_iter=10000)
        
        start_time = time.time()
        model.fit(X_train_mapped, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_c = model.predict(X_test_mapped)
        map_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred_c)
        print(f"{c_name:<10}{train_time:<20.4f}{map_time:<15.4f}{acc:<10.4f}")
        
    except Exception as e:
        print(f"{c_name:<10}{'Error':<20}{'Error':<15}{'Error':<10}")

print("\n3. Effect of Tolerance on Model Performance:")

print(f"{'Model':<20}{'Tolerance':<12}{'Model Train Time (s)':<20}{'Accuracy':<10}")
print("-" * 62)

tolerance_configs = [
    ('LinearSVC', 1e-4, 'Low'),
    ('LinearSVC', 1e-3, 'Medium'), 
    ('LinearSVC', 1e-2, 'High'),
    ('LogisticRegression', 1e-4, 'Low'),
    ('LogisticRegression', 1e-3, 'Medium'),
    ('LogisticRegression', 1e-2, 'High'),
]

for model_name, tol, tol_name in tolerance_configs:
    try:
        if model_name == 'LinearSVC':
            model = LinearSVC(penalty='l2', loss='hinge', tol=tol, C=1.0, max_iter=10000)
        else:
            model = LogisticRegression(penalty='l2', tol=tol, C=1.0, solver='liblinear', max_iter=10000)
        
        start_time = time.time()
        model.fit(X_train_mapped, y_train)
        train_time = time.time() - start_time
        
        y_pred_tol = model.predict(X_test_mapped)
        acc = accuracy_score(y_test, y_pred_tol)
        print(f"{model_name:<20}{tol_name:<12}{train_time:<20.4f}{acc:<10.4f}")
        
    except Exception as e:
        print(f"{model_name:<20}{tol_name:<12}{'Error':<20}{'Error':<10}")

print("\n4. Effect of Penalty on Model Performance:")

print(f"{'Model':<20}{'Penalty':<10}{'Model Train Time (s)':<20}{'Accuracy':<10}")
print("-" * 60)

penalty_configs = [
    ('LinearSVC', 'l1'),
    ('LinearSVC', 'l2'),
    ('LogisticRegression', 'l1'),
    ('LogisticRegression', 'l2'),
]

for model_name, penalty in penalty_configs:
    try:
        if model_name == 'LinearSVC':
            if penalty == 'l1':
                model = LinearSVC(penalty='l1', dual=False, tol=1e-4, C=1.0, max_iter=10000)
            else:
                model = LinearSVC(penalty='l2', loss='hinge', tol=1e-4, C=1.0, max_iter=10000)
        else:
            solver = 'liblinear'
            model = LogisticRegression(penalty=penalty, tol=1e-4, C=1.0, solver=solver, max_iter=10000)
        
        start_time = time.time()
        model.fit(X_train_mapped, y_train)
        train_time = time.time() - start_time
        
        y_pred_pen = model.predict(X_test_mapped)
        acc = accuracy_score(y_test, y_pred_pen)
        print(f"{model_name:<20}{penalty:<10}{train_time:<20.4f}{acc:<10.4f}")
        
    except Exception as e:
        print(f"{model_name:<20}{penalty:<10}{'Error':<20}{'Error':<10}")

# Test delay recovery functionality
print("\n" + "="*60)
print("DELAY RECOVERY TESTING")
print("="*60)
error_metrics = test_delay_recovery()

# Test delay recovery with all public models and print errors
print(f"\n" + "-"*60)
print("TESTING DELAY RECOVERY WITH ALL PUBLIC MODELS")
print("-"*60)

# The trained model from my_fit has different dimensions, so test with public models
public_models = load_public_models('public_mod.txt')

print("Testing delay recovery with all public models:")
print(f"{'Model':<8}{'MSE Error':<12}{'MAE Error':<12}{'L2 Error':<12}{'Relative Error':<15}{'Mean Delays':<20}")
print("-" * 79)

for i, pub_model in enumerate(public_models):
    p, q, r, s = my_decode(pub_model)
    
    # Calculate reconstruction error for each model
    error_metrics = calculate_reconstruction_error(pub_model, p, q, r, s)
    mean_delays = np.mean([p.mean(), q.mean(), r.mean(), s.mean()])
    
    print(f"{i:<8}{error_metrics['mse']:<12.6f}{error_metrics['mae']:<12.6f}{error_metrics['l2_norm']:<12.6f}{error_metrics['relative']:<15.6f}{mean_delays:<20.6f}")

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print("From the analysis, we observe the following:")
print("• Hinge function performs faster than Squared Loss function")
print("• The model is not performing well with lower C values like C ≤ 0.01")
print("• Tolerance increases training time without significant accuracy change")
print("• Both L1 and L2 penalties achieve good performance")
print("• Delay recovery errors vary significantly across different public models")
print("• Lower reconstruction errors indicate better delay recovery quality")