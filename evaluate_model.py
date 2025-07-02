
# Following code is working fine for fairness evaluation of xgboost
import pandas as pd
import numpy as np
import joblib
import json
import sys
import argparse
import logging
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, BoundaryAttack, HopSkipJump, ZooAttack
from sklearn.model_selection import train_test_split

# Suppress AIF360 warnings
logging.getLogger('aif360').setLevel(logging.ERROR)

# Redirect debug messages to stderr
def debug_print(message):
    print(message, file=sys.stderr)

# Model configuration
MODEL_CONFIGS = {
    'logistic_regression': {'requires_scaling': True},
    'random_forest': {'requires_scaling': False},
    'xgboost': {'requires_scaling': False}
}

# Load the model and dataset
def load_model_and_data(model_path, dataset_path, model_type_arg=None):
    # Determine model type
    model_type = model_type_arg
    if model_type is None:
        for key in MODEL_CONFIGS:
            if key in model_path.lower():
                model_type = key
                break
    if model_type is None:
        debug_print(f"Warning: Model type not recognized from model_path '{model_path}'. Defaulting to 'logistic_regression'.")
        model_type = 'logistic_regression'
    if model_type not in MODEL_CONFIGS:
        debug_print(f"Error: Invalid model_type '{model_type}'. Expected one of {list(MODEL_CONFIGS.keys())}.")
        print(json.dumps({"error": f"Invalid model_type '{model_type}'."}))
        sys.exit(1)

    debug_print(f"Using model_type: {model_type}")

    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        debug_print(f"Error loading model: {e}")
        print(json.dumps({"error": f"Error loading model: {e}"}))
        sys.exit(1)

    # Load the dataset
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        debug_print(f"Error loading dataset: {e}")
        print(json.dumps({"error": f"Error loading dataset: {e}"}))
        sys.exit(1)

    # Check for the presence of the target column
    if 'credit_risk' not in data.columns:
        debug_print("Error: 'credit_risk' column not found in dataset.")
        print(json.dumps({"error": "'credit_risk' column not found in dataset."}))
        sys.exit(1)

    # Check for NaN values in the target column
    initial_rows = len(data)
    data = data.dropna(subset=['credit_risk'])
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        debug_print(f"Dropped {dropped_rows} rows due to NaN values in credit_risk.")

    if len(data) == 0:
        debug_print("Error: No data remaining after dropping rows with NaN in credit_risk.")
        print(json.dumps({"error": "No data remaining after dropping rows with NaN in credit_risk."}))
        sys.exit(1)

    # The target column is 'credit_risk'
    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk']

    # Handle categorical variables (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Check for NaN or infinite values in features
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        debug_print("Found NaN or infinite values in features. Filling with column means.")
        X = X.fillna(X.mean())

    # Scale features if required
    if MODEL_CONFIGS[model_type]['requires_scaling']:
        scaler = StandardScaler()
        try:
            debug_print(f"Applying new StandardScaler for {model_type}.")
            X = scaler.fit_transform(X)
        except Exception as e:
            debug_print(f"Error scaling features: {e}")
            print(json.dumps({"error": f"Error scaling features: {e}"}))
            sys.exit(1)
    else:
        debug_print(f"No scaling applied for {model_type}.")
        X = X.values  # Convert to numpy array for consistency

    return model, X, y, data

def evaluate_fairness(model, X, y, data):
    notes = []
    score = 5

    # Debug: Print DataFrame info
    debug_print("\n--- Debugging DataFrame ---")
    debug_print(f"Raw DataFrame columns: {list(data.columns)}")
    debug_print(f"DataFrame shape: {data.shape}")

    # Standardize column names (handle case sensitivity)
    data = data.copy()
    data.columns = data.columns.str.strip().str.lower()

    # Check for required columns
    personal_status_cols = ['personal_status_a92', 'personal_status_a93', 'personal_status_a94']
    required_cols = ['age', 'credit_risk', 'foreign_worker_a202'] + personal_status_cols
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        debug_print(f"Error: Missing required columns: {missing_cols}")
        notes.append(f"Error: Missing columns: {', '.join(missing_cols)}")
        return score, notes

    # Note small dataset size
    if len(data) < 50:
        debug_print(f"Warning: Small dataset size: {len(data)} rows")
        notes.append(f"Warning: Small dataset size: {len(data)} rows")

    # Validate and binarize one-hot encoded columns
    for col in personal_status_cols + ['foreign_worker_a202']:
        if not data[col].isin([0, 1]).all():
            debug_print(f"Warning: Non-binary values in {col}: {data[col].unique()}")
            notes.append(f"Warning: Non-binary values in {col}")
            data[col] = (data[col] >= data[col].mean()).astype(int)

    # Derive sex from personal_status
    try:
        male_cols = ['personal_status_a93', 'personal_status_a94']
        data['sex'] = data[male_cols].eq(1).any(axis=1).astype(int)  # male=1, female=0
        debug_print(f"sex values: {data['sex'].value_counts().to_dict()}")
        if data['sex'].nunique() < 2:
            debug_print(f"Warning: Single group in sex: {data['sex'].unique()}")
            notes.append("Warning: Single group in sex")
    except Exception as e:
        debug_print(f"Error deriving 'sex': {str(e)}")
        notes.append(f"Error deriving 'sex': {str(e)}")
        return score, notes

    # Handle age columns
    try:
        if 'age_bin' in data.columns and data['age_bin'].isin([0, 1]).all() and data['age_bin'].nunique() == 2:
            debug_print(f"Using existing age_bin: {data['age_bin'].value_counts().to_dict()}")
        else:
            age_threshold = data['age'].mean()
            data['age_bin'] = data['age'].apply(lambda x: 1 if x >= age_threshold else 0)
            debug_print(f"Derived age_bin with threshold {age_threshold}: {data['age_bin'].value_counts().to_dict()}")
            if data['age_bin'].nunique() < 2:
                age_threshold = data['age'].median()
                data['age_bin'] = data['age'].apply(lambda x: 1 if x >= age_threshold else 0)
                debug_print(f"Fallback to median threshold {age_threshold}: {data['age_bin'].value_counts().to_dict()}")
        if data['age_bin'].nunique() < 2:
            debug_print(f"Warning: Single group in age_bin: {data['age_bin'].unique()}")
            notes.append("Warning: Single group in age_bin")
    except Exception as e:
        debug_print(f"Error creating 'age_bin': {str(e)}")
        notes.append(f"Error creating 'age_bin': {str(e)}")
        return score, notes

    # Process foreign_worker
    try:
        data['foreign_worker'] = data['foreign_worker_a202']
        debug_print(f"foreign_worker values: {data['foreign_worker'].value_counts().to_dict()}")
        if data['foreign_worker'].nunique() < 2:
            debug_print(f"Warning: Single group in foreign_worker: {data['foreign_worker'].unique()}")
            notes.append("Warning: Single group in foreign_worker")
    except Exception as e:
        debug_print(f"Error processing 'foreign_worker': {str(e)}")
        notes.append(f"Error processing 'foreign_worker': {str(e)}")
        return score, notes

    # Check for NaN or infinite values
    if data[['sex', 'age_bin', 'foreign_worker', 'credit_risk']].isna().sum().sum() > 0:
        debug_print("Error: NaN values in sensitive attributes or credit_risk")
        notes.append("Error: NaN values detected")
        return score, notes
    if np.any(np.isinf(data[['age', 'credit_risk']].values)):
        debug_print("Error: Infinite values in numerical columns")
        notes.append("Error: Infinite values detected")
        return score, notes

    # Define sensitive attributes
    sensitive_attributes = ['age_bin', 'sex', 'foreign_worker']
    fairness_results = {}

    # Prepare features for prediction
    feature_cols = [col for col in data.columns if col not in ['credit_risk', 'sex', 'age_bin', 'foreign_worker', 'personal_status']]
    X_test = data[feature_cols]
    if X_test.shape[1] != X.shape[1]:
        debug_print(f"Error: Feature count mismatch. Expected {X.shape[1]}, got {X_test.shape[1]}")
        notes.append(f"Error: Feature count mismatch")
        return score, notes

    # Generate predictions
    try:
        y_pred = model.predict(X_test)
        debug_print(f"Predictions: {y_pred[:5]}")
    except Exception as e:
        debug_print(f"Error generating predictions: {str(e)}")
        notes.append(f"Error generating predictions: {str(e)}")
        return score, notes

    for sensitive_attribute in sensitive_attributes:
        debug_print(f"\n--- Processing {sensitive_attribute} ---")
        # Set privileged and unprivileged groups
        if sensitive_attribute == 'sex':
            privileged_groups = [{'sex': 1}]  # male
            unprivileged_groups = [{'sex': 0}]  # female
            label = 'Personal Status'
        elif sensitive_attribute == 'age_bin':
            privileged_groups = [{'age_bin': 1}]  # >= mean/median age
            unprivileged_groups = [{'age_bin': 0}]  # < mean/median age
            label = 'Age'
        else:  # foreign_worker
            privileged_groups = [{'foreign_worker': 1}]  # non-foreign
            unprivileged_groups = [{'foreign_worker': 0}]  # foreign
            label = 'Foreign Worker'

        # Check for single groups
        if data[sensitive_attribute].nunique() < 2:
            debug_print(f"Skipping {sensitive_attribute}: Single group detected")
            notes.append(f"Fairness ({label}): Score 4/10")
            notes.append("Original accuracy: N/A")
            notes.append("Disparate Impact: N/A")
            notes.append("Statistical Parity Difference: N/A")
            notes.append("Equal Opportunity Difference: N/A")
            score -= 1
            continue

        # Create AIF360 datasets
        try:
            test_df = data[[sensitive_attribute, 'credit_risk'] + feature_cols]
            dataset_true = BinaryLabelDataset(
                df=test_df,
                label_names=['credit_risk'],
                protected_attribute_names=[sensitive_attribute],
                favorable_label=1,
                unfavorable_label=0
            )
            dataset_pred = dataset_true.copy()
            dataset_pred.labels = y_pred.reshape(-1, 1)

            # Compute fairness metrics
            classified_metric = ClassificationMetric(
                dataset_true,
                dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )

            accuracy = accuracy_score(y, y_pred)
            disp_impact = classified_metric.disparate_impact()
            stat_parity_diff = classified_metric.statistical_parity_difference()
            eq_opp_diff = classified_metric.equal_opportunity_difference() if not np.isnan(classified_metric.equal_opportunity_difference()) else 0

            fairness_results[sensitive_attribute] = {
                'accuracy': accuracy,
                'disparate_impact': disp_impact,
                'stat_parity_diff': stat_parity_diff,
                'eq_opp_diff': eq_opp_diff
            }

            # Calculate attribute score
            attr_score = 10
            if disp_impact > 1.5 or disp_impact < 0.5:
                attr_score -= 2
            elif disp_impact > 1.2 or disp_impact < 0.8:
                attr_score -= 1
            if abs(stat_parity_diff) > 0.3:
                attr_score -= 2
            elif abs(stat_parity_diff) > 0.1:
                attr_score -= 1
            if abs(eq_opp_diff) > 0.1:
                attr_score -= 1
            attr_score = max(0, min(10, attr_score))

            # Update overall score
            if (0.8 <= disp_impact <= 1.25 and abs(stat_parity_diff) <= 0.1 and abs(eq_opp_diff) <= 0.1):
                score += 1
                notes.append(f"{sensitive_attribute}: Model shows fairness")
            else:
                score -= 1
                notes.append(f"⚠️ {sensitive_attribute}: Model shows potential bias")

            # Print fairness metrics
            debug_print(f"\n--- Fairness Metrics for {sensitive_attribute} ---")
            debug_print(f"Accuracy: {accuracy:.4f}")
            debug_print(f"Disparate Impact: {disp_impact:.4f}")
            debug_print(f"Statistical Parity Difference: {stat_parity_diff:.4f}")
            debug_print(f"Equal Opportunity Difference: {eq_opp_diff:.4f}")

            # Add to notes
            notes.append(f"Fairness ({label}): Score {attr_score}/10")
            notes.append(f"Original accuracy: {accuracy:.2f}")
            notes.append(f"Disparate Impact: {disp_impact:.2f}")
            notes.append(f"Statistical Parity Difference: {stat_parity_diff:.2f}")
            notes.append(f"Equal Opportunity Difference: {eq_opp_diff:.2f}")

        except Exception as e:
            debug_print(f"Error processing {sensitive_attribute}: {str(e)}")
            notes.append(f"Error processing {sensitive_attribute}: {str(e)}")
            score -= 1
            continue

    score = max(0, min(10, score))
    debug_print(f"Final fairness score: {score}")
    debug_print(f"Final notes: {notes}")
    return score, notes


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from art.estimators.classification import SklearnClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

def evaluate_privacy(model, X, y, data):
    notes = []
    score = 5

    # Sensitive feature check
    sensitive_features = ['personal_status', 'age', 'foreign_worker']
    used_sensitive = [col for col in sensitive_features if col in data.columns]
    if used_sensitive:
        score -= 2
        notes.append(f"⚠️ Sensitive features used: {', '.join(used_sensitive)}")
    else:
        score += 1
        notes.append("✅ No sensitive features used.")

    # Determine model type
    model_class_str = str(type(model)).lower()
    is_sklearn = 'sklearn' in model_class_str or 'logisticregression' in model_class_str or 'randomforestclassifier' in model_class_str
    is_xgboost = 'xgbclassifier' in model_class_str or isinstance(model, xgb.XGBClassifier)

    try:
        # ART-based MIA for sklearn-compatible models
        if is_sklearn:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clip_values = (np.min(X), np.max(X))
            wrapped_model = SklearnClassifier(model=model, clip_values=clip_values)

            mia = MembershipInferenceBlackBox(wrapped_model)
            mia.fit(x=X_train, y=y_train, test_x=X_test)

            inferred_train = mia.infer(x=X_train)
            inferred_test = mia.infer(x=X_test)

            mia_score = (np.mean(inferred_train) + np.mean(np.logical_not(inferred_test))) / 2
            notes.append(f"🔍 MIA Accuracy: {mia_score:.2f}")

            if mia_score > 0.7:
                score -= 3
                notes.append("🔥 High risk: model leaks training membership info.")
            elif mia_score > 0.6:
                score -= 1
                notes.append("⚠️ Moderate privacy risk.")
            else:
                score += 1
                notes.append("✅ Low privacy leakage via MIA.")

        elif is_xgboost:
            # STEP 1: Sample down data for XGBoost MIA to avoid long training
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.5, stratify=y, random_state=1)

            # STEP 2: Create target model training data
            X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
                X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=42
            )

            target_model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', n_estimators=30, max_depth=3
            )
            target_model.fit(X_target_train, y_target_train)

            # STEP 3: Create shadow model data (reduced)
            X_shadow, _, y_shadow, _ = train_test_split(X_sample, y_sample, test_size=0.5, stratify=y_sample, random_state=2)
            X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
                X_shadow, y_shadow, test_size=0.5, stratify=y_shadow, random_state=2
            )

            shadow_model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', n_estimators=30, max_depth=3
            )
            shadow_model.fit(X_shadow_train, y_shadow_train)

            # STEP 4: Build attack dataset
            shadow_train_probs = shadow_model.predict_proba(X_shadow_train)
            shadow_test_probs = shadow_model.predict_proba(X_shadow_test)

            attack_X = np.vstack([shadow_train_probs, shadow_test_probs])
            attack_y = np.concatenate([np.ones(len(shadow_train_probs)), np.zeros(len(shadow_test_probs))])

            # Optional: Scale for logistic regression stability
            scaler = StandardScaler()
            attack_X_scaled = scaler.fit_transform(attack_X)

            # STEP 5: Train attack model (with fixed iterations)
            attack_model = LogisticRegression(max_iter=300, solver='lbfgs')
            attack_model.fit(attack_X_scaled, attack_y)

            # STEP 6: Apply on target model
            target_train_probs = target_model.predict_proba(X_target_train)
            target_test_probs = target_model.predict_proba(X_target_test)

            attack_preds_train = attack_model.predict(scaler.transform(target_train_probs))
            attack_preds_test = attack_model.predict(scaler.transform(target_test_probs))

            mia_preds = np.concatenate([attack_preds_train, attack_preds_test])
            mia_true = np.concatenate([np.ones(len(attack_preds_train)), np.zeros(len(attack_preds_test))])

            # STEP 7: Evaluate
            mia_auc = roc_auc_score(mia_true, mia_preds)
            notes.append(f"🎯 MIA ROC AUC Score: {mia_auc:.2f}")

            if mia_auc > 0.7:
                score -= 3
                notes.append("🔥 High risk: manual MIA shows strong membership leakage.")
            elif mia_auc > 0.6:
                score -= 1
                notes.append("⚠️ Moderate privacy risk in manual MIA.")
            else:
                score += 1
                notes.append("✅ Low privacy leakage via manual MIA.")

        # Unsupported model type
        else:
            notes.append(f"❌ MIA skipped: Unsupported model type '{model_class_str}'")
            score -= 1

    except Exception as e:
        debug_print(f"Privacy check failed (MIA): {str(e)}")
        notes.append(f"❌ Privacy check failed (MIA): {str(e)}")
        score -= 1

    return max(0, min(10, score)), notes


# NEED TO LOOK AT FGSM , PGD AND DEEPFOOL FOR LR

import numpy as np
from sklearn.metrics import accuracy_score
import sys
import os
import contextlib

# Try to import memory monitor
try:
    import psutil
except ImportError:
    psutil = None

# Try ART imports
try:
    from art.estimators.classification.scikitlearn import ScikitlearnClassifier
except ImportError:
    try:
        from art.estimators.classification import SklearnClassifier as ScikitlearnClassifier
    except ImportError:
        ScikitlearnClassifier = None

if ScikitlearnClassifier is not None:
    from art.attacks.evasion import BoundaryAttack, HopSkipJump

# Debug print to stderr only
def debug_print(message, enabled=True):
    if enabled:
        print(f"DEBUG: {message}", file=sys.stderr)
        sys.stderr.flush()

# Suppress stdout (to avoid ART output messing up JSON)
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Evaluate safety
def evaluate_safety(model, X, y, model_type):
    debug = False
    notes = []
    base_score = 10
    total_drop = 0
    successful_attacks = 0
    attempted_attacks = 0

    try:
        sys.stderr.flush()

        # Model diagnostics

        model_class = str(type(model))
        model_module = getattr(model, '__module__', 'unknown')
        has_n_estimators = hasattr(model, 'n_estimators')
        debug_print(f"Model class: {model_class}, module: {model_module}, has_n_estimators: {has_n_estimators}", debug)
        debug_print(f"X shape: {X.shape}, y shape: {y.shape}", debug)

        if psutil:
            mem = psutil.virtual_memory()
            debug_print(f"Memory usage: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available", debug)

        # Tree-based detection
        is_tree_based = any([
            'randomforestclassifier' in model_class.lower(),
            'xgbclassifier' in model_class.lower(),
            'sklearn.ensemble' in model_module.lower(),
            'xgboost' in model_module.lower(),
            has_n_estimators and ('ensemble' in model_module.lower() or 'xgboost' in model_module.lower())
        ])
        debug_print(f"Is tree-based: {is_tree_based}", debug)

        # Model type fallback
        inferred_model_type = model_type if model_type in ['logistic_regression', 'random_forest', 'xgboost'] else None
        if not inferred_model_type:
            inferred_model_type = 'xgboost' if 'xgbclassifier' in model_class.lower() else 'random_forest' if is_tree_based else 'logistic_regression'
            notes.append(f"Invalid model_type '{model_type}'. Using inferred: {inferred_model_type}.")
        effective_model_type = inferred_model_type

        debug_print(f"Effective model type: {effective_model_type}", debug)

        # Clip values
        try:
            clip_values = (float(np.min(X)), float(np.max(X)))
        except Exception as e:
            notes.append(f"Clip values failed: {str(e)}")
            clip_values = (0.0, 1.0)

        # Original accuracy
        try:
            y_pred = model.predict(X)
            acc_original = accuracy_score(y, y_pred)
            notes.append(f"Original accuracy: {acc_original:.2f}")
        except Exception as e:
            notes.append(f"Accuracy computation failed: {str(e)}")
            return max(0, min(10, base_score)), notes

        if ScikitlearnClassifier is None:
            notes.append("ART not available. Skipping adversarial attacks.")
            return max(0, min(10, base_score - 2)), notes

        # Wrap classifier
        try:
            classifier = ScikitlearnClassifier(model=model, clip_values=clip_values, use_logits=False)
        except Exception as e:
            notes.append(f"ScikitlearnClassifier failed: {str(e)}")
            return max(0, min(10, base_score - 2)), notes

        # Attacks list
        attacks = [
            ("Boundary", BoundaryAttack(estimator=classifier, targeted=False, max_iter=50, verbose=False)),
            ("HopSkipJump", HopSkipJump(classifier=classifier, targeted=False, max_iter=3, max_eval=20, init_eval=3, verbose=False))
        ]

        for name, attack in attacks:
            try:
                attempted_attacks += 1
                with suppress_stdout():
                    X_adv = attack.generate(x=X)
                y_adv_pred = model.predict(X_adv)
                acc_adv = accuracy_score(y, y_adv_pred)
                drop = acc_original - acc_adv
                total_drop += drop
                successful_attacks += int(drop > 0.05)
                notes.append(f"{name} attack → Accuracy drop: {drop:.2f}")
            except Exception as e:
                notes.append(f"{name} attack failed: {str(e)}")
                base_score -= 0.5

        # Final scoring
        if attempted_attacks == 0:
            penalty = 2 if effective_model_type in ['random_forest', 'xgboost'] else 3
            final_score = base_score - penalty
            notes.append("No attacks attempted.")
        elif successful_attacks >= 2:
            final_score = base_score - 5
        elif successful_attacks == 1:
            final_score = base_score - 3
        else:
            final_score = base_score

        if effective_model_type in ['random_forest', 'xgboost'] and final_score == base_score:
            final_score = min(final_score + 1, 10)

    except Exception as e:
        notes.append(f"Safety evaluation failed: {str(e)}")
        final_score = base_score - 6

    sys.stderr.flush()
    return max(0, min(10, final_score)), notes


# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser(description='Evaluate a model for fairness, privacy, and safety.')
#     parser.add_argument('model_path', help='Path to the model file (e.g., xgboost_model.pkl)')
#     parser.add_argument('dataset_path', help='Path to the test dataset (e.g., german_credit_test.csv)')
#     parser.add_argument('principles', help='List of principles to evaluate (e.g., ["fairness", "privacy", "safety"])')
#     parser.add_argument('--model_type', type=str, default=None, choices=['logistic_regression', 'random_forest', 'xgboost'], help='Explicitly specify model type (optional).')
#     args = parser.parse_args()

#     # Debug input arguments
#     debug_print(f"Received arguments: model_path='{args.model_path}', dataset_path='{args.dataset_path}', principles='{args.principles}', model_type='{args.model_type}'")

#     # Parse principles
#     try:
#         principles = json.loads(args.principles)
#     except Exception as e:
#         debug_print(f"Error parsing principles: {str(e)}")
#         print(json.dumps({"error": f"Error parsing principles: {e}"}))
#         sys.exit(1)

#     # Load model and data
#     try:
#         model, X, y, data = load_model_and_data(model_path=args.model_path, dataset_path=args.dataset_path, model_type_arg=args.model_type)
#     except Exception as e:
#         debug_print(f"Error in load_model_and_data: {str(e)}")
#         print(json.dumps({"error": f"Error loading model or data: {e}"}))
#         sys.exit(1)

#     # Determine model_type for evaluate_safety
#     model_types = [key for key in MODEL_CONFIGS if key in args.model_path.lower() or key == args.model_type]
#     if not model_types:
#         debug_print(f"Warning: No model type matched in model_path '{args.model_path}' or model_type '{args.model_type}'. Defaulting to 'logistic_regression'.")
#         model_type = 'logistic_regression'
#     else:
#         model_type = model_types[0]
#     debug_print(f"Selected model_type for evaluation: {model_type}")

#     # Evaluate based on principles
#     report = {}
#     for principle in principles:
#         try:
#             if principle == 'fairness':
#                 score, notes = evaluate_fairness(model, X, y, data)
#             elif principle == 'privacy':
#                 score, notes = evaluate_privacy(model, X, y, data)
#             elif principle == 'safety':
#                 score, notes = evaluate_safety(model, X, y, model_type)
#             else:
#                 score, notes = 5, ["Direct model evaluation not implemented for this principle."]
#             report[principle] = {"score": score, "notes": notes}
#         except Exception as e:
#             debug_print(f"Error evaluating principle {principle}: {str(e)}")
#             report[principle] = {"score": 5, "notes": [f"Error evaluating principle: {e}"]}

#     # Output the report
#     print(json.dumps(report))
#     report_path = "evaluation_report.json"
#     try:
#         with open(report_path, 'w') as f:
#             json.dump(report, f, indent=2)
#     except Exception as e:
#         debug_print(f"Could not save report to JSON: {e}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate a model for fairness, privacy, and safety.')
    parser.add_argument('model_path', help='Path to the model file (e.g., xgboost_model.pkl)')
    parser.add_argument('dataset_path', help='Path to the test dataset (e.g., german_credit_test.csv)')
    parser.add_argument('principles', help='List of principles to evaluate (e.g., ["fairness", "privacy", "safety"])')
    parser.add_argument('--model_type', type=str, default=None, choices=['logistic_regression', 'random_forest', 'xgboost'], help='Explicitly specify model type (optional).')
    args = parser.parse_args()

    # Debug input arguments
    debug_print(f"Received arguments: model_path='{args.model_path}', dataset_path='{args.dataset_path}', principles='{args.principles}', model_type='{args.model_type}'")

    # Parse principles
    try:
        principles = json.loads(args.principles)
        debug_print(f"Parsed principles: {principles}")
        if not isinstance(principles, list):
            raise ValueError("Principles must be a list")
    except Exception as e:
        debug_print(f"Error parsing principles: {str(e)}")
        print(json.dumps({"error": f"Error parsing principles: {e}"}))
        sys.exit(1)

    # Validate principles
    valid_principles = ['fairness', 'privacy', 'safety']
    principles = [p for p in principles if p in valid_principles]
    if not principles:
        debug_print("No valid principles provided")
        print(json.dumps({"error": "No valid principles provided. Expected: fairness, privacy, safety"}))
        sys.exit(1)

    # Load model and data
    try:
        model, X, y, data = load_model_and_data(model_path=args.model_path, dataset_path=args.dataset_path, model_type_arg=args.model_type)
    except Exception as e:
        debug_print(f"Error in load_model_and_data: {str(e)}")
        print(json.dumps({"error": f"Error loading model or data: {e}"}))
        sys.exit(1)

    # Determine model_type for evaluate_safety
    model_types = [key for key in MODEL_CONFIGS if key in args.model_path.lower() or key == args.model_type]
    if not model_types:
        debug_print(f"Warning: No model type matched in model_path '{args.model_path}' or model_type '{args.model_type}'. Defaulting to 'logistic_regression'.")
        model_type = 'logistic_regression'
    else:
        model_type = model_types[0]
    debug_print(f"Selected model_type for evaluation: {model_type}")

    # Evaluate based on principles
    report = {}
    for principle in principles:
        try:
            debug_print(f"Evaluating principle: {principle}")
            if principle == 'fairness':
                score, notes = evaluate_fairness(model, X, y, data)
            elif principle == 'privacy':
                score, notes = evaluate_privacy(model, X, y, data)
            elif principle == 'safety':
                score, notes = evaluate_safety(model, X, y, model_type)
            else:
                score, notes = 5, ["Direct model evaluation not implemented for this principle."]
            report[principle] = {"score": score, "notes": notes}
        except Exception as e:
            debug_print(f"Error evaluating principle {principle}: {str(e)}")
            report[principle] = {"score": 5, "notes": [f"Error evaluating principle: {e}"]}

    # Output the report
    print(json.dumps(report))
    report_path = "evaluation_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        debug_print(f"Could not save report to JSON: {e}")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()