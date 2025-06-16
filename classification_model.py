import os
import json
import torch
import random
import optuna
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from models import EEGNet, MidCNN, DeeperTemporalCNN, SherpaModel, SoftmaxWrapper


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(source, data_dir):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, data_dir)
    X = np.load(os.path.join(data_dir, f"X_{source}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{source}.npy")).astype(int)
    ids = np.load(os.path.join(data_dir, f"ids_{source}.npy"))
    return X, y, ids

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def get_model_by_name(name, n_output, input_shape, dropout, device, **kwargs):
    if name == "eegnet":
        return EEGNet(n_classes=n_output, Chans=input_shape[0], Samples=input_shape[1], dropoutRate=dropout, **kwargs).to(device)
    elif name == "midcnn":
        return MidCNN(n_output, input_shape, dropout_rate=dropout, **kwargs).to(device)
    elif name == "deepcnn":
        return DeeperTemporalCNN(n_output, input_shape, dropout_rate=dropout, **kwargs).to(device)
    elif name == "sherpamodel":
        model = SherpaModel(
            n_output=n_output,
            input_shape=input_shape,
            dropout_rate=dropout,
            fc_drop=kwargs.get("fc_drop", dropout),
            fc1_units=kwargs.get("fc1_units", 128),
            fc2_units=kwargs.get("fc2_units", 64),
            fc3_units=kwargs.get("fc3_units", 32),
            kernel_size=kwargs.get("kernel_size", 50),
            conv_filters=kwargs.get("conv_filters", [16, 32, 64, 128])
        ).to(device)
        return model
    else:
        raise ValueError(f"Unknown model name: {name}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=100, patience=20):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(max_epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        if val_loader is None:
            continue 
        else:
            avg_epoch_loss = 0.0
            total_val = 0
            model.eval()
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    out = model(Xb)
                    loss = criterion(out, yb)
                    avg_epoch_loss += loss.item() * yb.size(0)
                    total_val += yb.size(0)
            avg_epoch_loss /= total_val

            scheduler.step(avg_epoch_loss)  # Adjust learning rate

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if val_loader is not None:
        model.load_state_dict(best_model_state)
    return model


def objective(trial, X, y, device, model_name):
    dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    fc_drop = trial.suggest_float("fc_drop", 0.2, 0.6, step=0.1)
    fc1_units = trial.suggest_categorical("fc1_units", [128, 256])
    fc2_units = trial.suggest_categorical("fc2_units", [32, 64, 128]) 
    fc3_units = trial.suggest_categorical("fc3_units", [16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [25, 50, 75])
    conv_filters = trial.suggest_categorical("conv_filters", [(16, 32, 64, 128), (8, 16, 32, 64)])
    input_shape = (X.shape[1], X.shape[2])
    n_output = len(np.unique(y))
    accs = []

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in splitter.split(X):
        model = get_model_by_name(model_name, n_output, input_shape, dropout, device, fc_drop=fc_drop,
                                    fc1_units=fc1_units, fc2_units=fc2_units, fc3_units=fc3_units, kernel_size=kernel_size, conv_filters=list(conv_filters))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_train = torch.tensor(X[train_idx]).float()
        y_train = torch.tensor(y[train_idx]).long()
        X_val = torch.tensor(X[val_idx]).float()
        y_val = torch.tensor(y[val_idx]).long()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

        train_model(model, train_loader, val_loader, criterion, optimizer, device)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        accs.append(correct / total)

    return np.mean(accs) if accs else 0.0


def evaluate_on_test_set(model, X_test, y_test, class_labels, device, results_dir):
    model.eval()
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_test_tensor = torch.tensor(y_test).long().to(device)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    criterion = nn.CrossEntropyLoss()
    total, correct, test_loss = 0, 0, 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            out = model(Xb)
            test_loss += criterion(out, yb).item() * yb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    avg_loss = test_loss / total
    acc = correct / total
    print(f"\nHoldout Test Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")

    with open(os.path.join(results_dir, "holdout_test_results.json"), "w") as f:
        json.dump({"test_accuracy": acc, "test_loss": avg_loss}, f, indent=4)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Holdout Test Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\nClassification Report:\n")
    print(report)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to: {report_path}")


def run_kfold_evaluation(X, y, best_params, device, results_dir, model_name, n_output):

    input_shape = (X.shape[1], X.shape[2])
    criterion = nn.CrossEntropyLoss()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_losses, all_accs, train_losses, train_accs, fold_runtimes, best_models = [], [], [], [], [], []
    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        print(f"Fold {fold+1}: Trials {len(train_idx)} train / {len(val_idx)} val")

        model = get_model_by_name(model_name, n_output, input_shape, best_params["dropout"], device, fc_drop=best_params.get("fc_drop", best_params["dropout"]),
                                      fc1_units=best_params.get("fc1_units", 128), fc2_units=best_params.get("fc2_units", 64), fc3_units=best_params.get("fc3_units", 32), 
                                      kernel_size=best_params.get("kernel_size", 50), conv_filters=best_params.get("conv_filters", [16, 32, 64, 128]))

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=0.01)

        X_train = torch.tensor(X[train_idx]).float()
        X_val = torch.tensor(X[val_idx]).float()
        y_train = torch.tensor(y[train_idx]).long()
        y_val = torch.tensor(y[val_idx]).long()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

        model = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        best_models.append(model)

        # Evaluate training
        model.eval()
        train_loss, train_correct, train_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                train_loss += criterion(out, yb).item() * yb.size(0)
                preds = out.argmax(dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += yb.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * yb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_loss = val_loss / total
        acc = correct / total
        print(f"  Train Acc = {train_acc:.4f}, Train Loss = {avg_train_loss:.4f}")
        print(f"  Val   Acc = {acc:.4f}, Val   Loss = {avg_loss:.4f}")

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        all_losses.append(avg_loss)
        all_accs.append(acc)
        fold_runtimes.append((time.time() - fold_start) / 60.0)

    total_runtime_min = (time.time() - total_start) / 60.0
    eval_dict = {
        "loss": all_losses,
        "acc": all_accs,
        "avg_loss": np.mean(all_losses),
        "avg_acc": np.mean(all_accs),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "fold_runtimes_min": fold_runtimes,
        "total_runtime_min": total_runtime_min,
        "model": model_name
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "kfold_eval_dict.json"), "w") as f:
        json.dump(eval_dict, f, indent=4)

    pd.DataFrame({
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": all_losses,
        "val_acc": all_accs
    }).to_csv(os.path.join(results_dir, "kfold_metrics.csv"), index=False)

    best_model_idx = int(np.argmax(all_accs))

    print(f"\nTotal K-Fold runtime: {total_runtime_min:.2f} minutes")

    return best_models[best_model_idx]


def retrain_full(X, y, best_params, device, model_name, n_output):
    input_shape = (X.shape[1], X.shape[2])
    model = get_model_by_name(
        model_name,
        n_output,
        input_shape,
        best_params["dropout"],
        device,
        fc1_units=best_params.get("fc1_units", 128),
        fc2_units=best_params.get("fc2_units", 64),
        fc3_units=best_params.get("fc3_units", 32),
        fc_drop=best_params.get("fc_drop", 0.3),
        kernel_size=best_params.get("kernel_size", 50),
        conv_filters=best_params.get("conv_filters", [16, 32, 64, 128])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X).float(),
                                            torch.tensor(y).long()), batch_size=64, shuffle=True)

    model = train_model(model, train_loader, None, criterion, optimizer, device)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_source", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["eegnet", "midcnn", "deepcnn", "sherpamodel"])
    parser.add_argument("--class_labels", type=str, nargs='+', required=True)
    args = parser.parse_args()

    print("Result path", args.results_dir)

    set_seed()
    device = get_device()
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")

    X, y, ids = load_data(args.data_source, args.data_dir)
    n_output = len(np.unique(y))
    # Split off a random 10% holdout test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    print(f"Train+Val shape: {X_trainval.shape}, Test shape: {X_test.shape}")
    print("Unique labels in full y:", np.unique(y))
    print("Unique labels in y_trainval:", np.unique(y_trainval))
    print("Number of output classes (n_output):", len(np.unique(y_trainval)))

    full_results_dir = os.path.join(args.results_dir, args.data_source)
    os.makedirs(full_results_dir, exist_ok=True)
    params_path = os.path.join(full_results_dir, "best_params.json")
    print("param path: ", params_path)

    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
        print("Loaded existing best parameters.")
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_trainval, y_trainval, device, args.model), n_trials=15)
        best_params = study.best_trial.params
        best_params["D"] = 1
        save_json(best_params, params_path)

    print("Best parameters:", best_params)
    run_kfold_evaluation(X_trainval, y_trainval, best_params, device, full_results_dir, model_name=args.model, n_output=n_output)
    finalmodel = retrain_full(X_trainval, y_trainval, best_params, device, model_name=args.model, n_output=n_output)
    torch.save(finalmodel.state_dict(), os.path.join(full_results_dir, "finalmodel.pt"))
    print("Final Model output shape:", finalmodel(torch.randn(1, X.shape[1], X.shape[2]).to(device)).shape)
    evaluate_on_test_set(finalmodel, X_test, y_test, args.class_labels, device, full_results_dir)

    # SHAP
    shap_start = time.time()
    X_trainval = (torch.tensor(X_trainval).float()).to(device)
    X_test = (torch.tensor(X_test).float()).to(device)
    softmax_model = SoftmaxWrapper(finalmodel)
    explainer = shap.GradientExplainer(softmax_model, X_trainval)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.array(shap_values)
    print("SHAP value shape: ", shap_values.shape)
    np.save(os.path.join(full_results_dir, "shap_values.npy"), np.array(shap_values, dtype=object))
    print("Saved Shap for test set")
    shap_runtime_min = (time.time() - shap_start) / 60.0
    print(f"\nSHAP runtime: {shap_runtime_min:.2f} minutes")
        

if __name__ == "__main__":
    main()
