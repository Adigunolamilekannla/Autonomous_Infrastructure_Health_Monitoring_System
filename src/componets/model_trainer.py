import sys, os
import torch
import mlflow
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,random_split
from dotenv import load_dotenv

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.dir_manager import MakeDirectory
from src.utils.model import classification_report, regression_report, LSTMNet,get_bridge_img_model_optimizer,create_target
from src.utils.load_save import generate_report

# ------------------------------------------------------
# ✅ Load environment variables securely
# ------------------------------------------------------
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


# ------------------------------------------------------
# ✅ CONFIGURATION CLASS
# ------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    transform_train_bridge_data: str = "artifacts/transformed_data/bridge_data/train_bridge_data.pt"
    transform_test_bridge_data: str = "artifacts/transformed_data/bridge_data/test_bridge_data.pt"

    transform_train_rul_data: str = "artifacts/transformed_data/rul_data/train_rul_data.csv"
    transform_test_rul_data: str = "artifacts/transformed_data/rul_data/test_rul_data.csv"

    transform_train_bridge_image_dataset: str = "artifacts/transformed_data/bridge_image_dataset/train_image_dataset.pt"
    transform_test_bridge_image_dataset: str = "artifacts/transformed_data/bridge_image_dataset/test_image_dataset.pt"

    trained_bridge_sensor_model: str = "artifacts/models/bridge_model.pt"
    trained_rul_model: str = "artifacts/models/rul_model.joblib"
    trained_bridge_img_model:str = "artifacts/models/bridge_img_model.pt"
    trained_fulsion_model:str = os.path.join("artifacts", "models", "fulsion_model.joblib")
    scaler_model_path:str = os.path.join("artifacts", "models", "scaler_model.joblib")



    


# ------------------------------------------------------
# ✅ MODEL TRAINER CLASS
# ------------------------------------------------------
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        try:
            logging.info("Initializing ModelTrainerConfig...")
            self.config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------
    # ✅ MLflow tracking
    # ------------------------------------------------------
    def mlflow_tracking(self, model_name, model, train_metrics, test_metrics, register_model=True, framework="sklearn"):
        """Logs model, parameters, and metrics to MLflow"""
        try:
            logging.info(f"Starting MLflow tracking for {model_name}")

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment("Bridge_Health_Monitoring")

            with mlflow.start_run(run_name=model_name):
                # Log model
                if framework == "sklearn":
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=model_name if register_model else None,
                    )
                elif framework == "pytorch":
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=model_name if register_model else None,
                    )

                # Log parameters (if available)
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Log metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)

            logging.info(f"MLflow tracking completed for {model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------
    # ✅ Train and Evaluate RUL Data (Random Forest)
    # ------------------------------------------------------
    def train_evaluate_rul_data(self, train_path, test_path):
        try:
            logging.info("Training and evaluating RUL data...")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop("RUL", axis=1)
            y_train = train_df["RUL"]
            X_test = test_df.drop("RUL", axis=1)
            y_test = test_df["RUL"]

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='r2')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_report = regression_report(y_train, y_train_pred)
            test_report = regression_report(y_test, y_test_pred)

            # Log with MLflow
            self.mlflow_tracking("RandomForestRegressor", best_model, train_report, test_report)

            # Save model
            MakeDirectory(self.config.trained_rul_model)
            joblib.dump(best_model, self.config.trained_rul_model)
            logging.info(f"Model saved successfully at {self.config.trained_rul_model}")

            return {"train_rul": train_report, "test_rul": test_report}

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------
    # ✅ Train and Evaluate Bridge Sensor Data (LSTM)
    # ------------------------------------------------------
    def train_evaluate_bridge_sensor_data(self, x_train_path, x_test_path):
        try:
            logging.info("Training and evaluating Bridge Sensor LSTM model...")

            train = torch.load(x_train_path)
            test = torch.load(x_test_path)

            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            test_loader = DataLoader(test, batch_size=32, shuffle=False)

            input_size = 23
            num_hidden = 20
            num_layers = 10
            num_epochs = 20
            learning_rate = 0.001

            net = LSTMNet(input_size, num_hidden, num_layers)
            lossFun = torch.nn.BCELoss()
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                net.train()
                epoch_loss = 0.0
                for X, y in train_loader:
                    y_pred, _ = net(X)
                    y = y.view(-1)
                    y_pred = y_pred.view(-1)

                    loss = lossFun(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                logging.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

            # Evaluation
            net.eval()
            y_train, y_train_pred, y_test, y_test_pred = [], [], [], []

            with torch.no_grad():
                for X, y in train_loader:
                    preds, _ = net(X)
                    preds = (preds >= 0.5).float()
                    y_train_pred.extend(preds.view(-1).tolist())
                    y_train.extend(y.view(-1).tolist())

                for X, y in test_loader:
                    preds, _ = net(X)
                    preds = (preds >= 0.5).float()
                    y_test_pred.extend(preds.view(-1).tolist())
                    y_test.extend(y.view(-1).tolist())

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            y_train_pred = np.array(y_train_pred)
            y_test_pred = np.array(y_test_pred)

            train_report = classification_report(y_train, y_train_pred)
            test_report = classification_report(y_test, y_test_pred)

            # Log with MLflow
            self.mlflow_tracking("LSTM_Bridge_Sensor", net, train_report, test_report, framework="pytorch")

            # Save model
            MakeDirectory(self.config.trained_bridge_sensor_model)
            torch.save(net.state_dict(), self.config.trained_bridge_sensor_model)
            logging.info(f"Bridge Sensor Model saved at {self.config.trained_bridge_sensor_model}")

            return {"train_bridge_sensor": train_report, "test_bridge_sensor": test_report}

        except Exception as e:
            raise CustomException(e, sys)
        


    def train_bridge_image(self, x_train_path, x_test_path):
        """
        x_train_path, x_test_path: paths to torch-saved ImageFolder-like dataset objects
        Each sample should be (image_tensor, label) with label in {0,1}.
        """
        try:
            logging.info("Training bridge image model...")

            # load datasets (torch objects saved earlier)
            train_dataset = torch.load(x_train_path)
            test_dataset = torch.load(x_test_path)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # get model + optimizer + loss from helper
            net, optimizer, lossFun = get_bridge_img_model_optimizer()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)

            num_epochs = 2
            accumulation_steps = 4

            for epoch in range(num_epochs):
                net.train()
                running_loss = 0.0
                running_acc = 0.0

                for i, (X, y) in enumerate(train_loader):
                    X = X.to(device).float()
                    y = y.to(device).float().view(-1)

                    # forward
                    y_pred = net(X).view(-1)
                    loss = lossFun(y_pred, y)
                    loss = loss / accumulation_steps
                    loss.backward()

                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()

                    preds = (y_pred >= 0.5).float()
                    acc = (preds == y).float().mean().item()

                    running_loss += loss.item() * accumulation_steps
                    running_acc += acc

                avg_loss = running_loss / len(train_loader)
                avg_acc = running_acc / len(train_loader)
                logging.info(f"Image Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")

            # ============================================
            # 🔍 Evaluation phase
            # ============================================
            net.eval()
            y_train, y_train_pred, y_test, y_test_pred = [], [], [], []

            with torch.no_grad():
                # ---- Evaluate test set ----
                for X, y in test_loader:
                    X = X.to(device).float()
                    y = y.to(device).float().view(-1)
                    y_pred = net(X).view(-1)
                    preds = (y_pred >= 0.5).float()
                    y_test.extend(y.cpu().numpy().tolist())
                    y_test_pred.extend(preds.cpu().numpy().tolist())

                # ---- Evaluate train set ----
                for X, y in train_loader:
                    X = X.to(device).float()
                    y = y.to(device).float().view(-1)
                    y_pred = net(X).view(-1)
                    preds = (y_pred >= 0.5).float()
                    y_train.extend(y.cpu().numpy().tolist())
                    y_train_pred.extend(preds.cpu().numpy().tolist())

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            y_train_pred = np.array(y_train_pred)
            y_test_pred = np.array(y_test_pred)

            # reports
            train_report = classification_report(y_train, y_train_pred)
            test_report = classification_report(y_test, y_test_pred)

            # ============================================
            # 🧠 MLflow tracking for Bridge Image model
            # ============================================
            self.mlflow_tracking("CNN_Bridge_Image", net, train_report, test_report, framework="pytorch")

            # ============================================
            # 💾 Save model
            # ============================================
            MakeDirectory(self.config.trained_bridge_img_model)
            torch.save(net.state_dict(), self.config.trained_bridge_img_model)
            logging.info(f"Bridge image model saved to {self.config.trained_bridge_img_model}")

            return {"train_bridge_image": train_report, "test_bridge_image": test_report}

        except Exception as e:
            raise CustomException(e, sys)


    def train_evaluate_fusion_model(self):
        try:
            """
            Load predictions from RUL, bridge image, and sensor models,
            combine them, and train a fusion model (RandomForestClassifier)
            based on their outputs.

            * bridge_image_prediction: (0 → GOOD) (1 → BAD)
            * rul_model_prediction: >=180 GOOD, 50<RUL<180 MODERATE, <50 BAD
            * bridge_sensor_prediction: 0 → Standing, 1 → Collapsed
            """

            # === Load RUL dataset and model ===
            rul_dataset = pd.read_csv("notebooks/transformed_data/rul_dataset.csv").drop("Unnamed: 0", axis=1)
            trained_rul_model = joblib.load(self.config.trained_rul_model)
            rul_model_prediction = trained_rul_model.predict(rul_dataset).round()

            all_model_prediction_dataset = pd.DataFrame({"rul_model_prediction": rul_model_prediction})




            # === Load bridge image dataset and model ===
            dataset = torch.load("data/image_dataset/road_image_dataset/image_data.pt")
            train_size = int(0.075 * len(dataset))
            test_size = len(dataset) - train_size
            image_data, _ = random_split(dataset, [train_size, test_size])
            data_loader = DataLoader(image_data, batch_size=32, shuffle=True)

            cnn_net = get_bridge_img_model_optimizer()[0]
            trained_bridge_img_model = torch.load(self.config.trained_bridge_img_model)
            cnn_net.load_state_dict(trained_bridge_img_model)
            cnn_net.eval()

            bridged_image_prediction = []
            with torch.no_grad():
                for X_val, _ in data_loader:
                    y_pred = cnn_net(X_val)
                    preds = (y_pred >= 0.5).float()
                    bridged_image_prediction.extend(preds.tolist())

            bridged_image_prediction = np.array(bridged_image_prediction).squeeze()
            all_model_prediction_dataset["bridge_image_prediction"] = bridged_image_prediction





            # === Load sensor dataset and model ===
            sensor_dataset = pd.read_csv("notebooks/transformed_data/bridged_sensor_dataset.csv").drop("Unnamed: 0", axis=1)
            input_size, num_hidden, num_layers = 23, 20, 10
            lstm_net = LSTMNet(input_size, num_hidden, num_layers)
            trained_bridge_sensor_model = torch.load(self.config.trained_bridge_sensor_model)
            lstm_net.load_state_dict(trained_bridge_sensor_model)

            scaler_model = joblib.load(self.config.scaler_model_path)
            scaled_x_train = torch.tensor(scaler_model.transform(sensor_dataset)).float()

            lstm_net.eval()
            with torch.no_grad():
                y_pred, _ = lstm_net(scaled_x_train)
                bridge_sensor_prediction = (y_pred >= 0.5).float().squeeze().detach().numpy()

            all_model_prediction_dataset["bridge_sensor_prediction"] = bridge_sensor_prediction





            # === Create fusion target and train RandomForest ===
            all_model_prediction_dataset["fusion_target"] = all_model_prediction_dataset.apply(create_target, axis=1)
            X = all_model_prediction_dataset.drop("fusion_target", axis=1)
            y = all_model_prediction_dataset["fusion_target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(class_weight="balanced")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_report = classification_report(y_train, y_train_pred)
            y_test_report = classification_report(y_test, y_test_pred)

            # === MLflow tracking ===
            self.mlflow_tracking("RandomForestClassifier", model, y_train_report, y_test_report, True, "sklearn")

            # === Save fusion model ===
            MakeDirectory(self.config.trained_fulsion_model)
            joblib.dump(model, self.config.trained_fulsion_model)

            return {"train_report": y_train_report, "test_report": y_test_report}

        except Exception as e:
            raise CustomException(e, sys)

        # ------------------------------------------------------
    # ✅ Run full model training pipeline
    # ------------------------------------------------------

    def initiate_models(self):
        try:
            logging.info("🚀 Model training pipeline started...")

            # Train RUL model
            rul_results = self.train_evaluate_rul_data(
                self.config.transform_train_rul_data,
             self.config.transform_test_rul_data
            )

            # Train Bridge Sensor model
            bridge_results = self.train_evaluate_bridge_sensor_data(
                self.config.transform_train_bridge_data,
                self.config.transform_test_bridge_data
            )

            # Train Bridge Image model
            bridge_img_results = self.train_bridge_image(self.config.transform_train_bridge_image_dataset,
                                                         self.config.transform_test_bridge_image_dataset)


            # Train Fulsion Model 

            fulsion_data_report = self.train_evaluate_fusion_model()
            # Save both reports
            generate_report(rul_results)
            generate_report(bridge_results)
            generate_report(bridge_img_results)
            generate_report(fulsion_data_report)

            logging.info("✅ All models trained and evaluated successfully.")
            return {**rul_results, **bridge_results, **bridge_img_results, **fulsion_data_report}

        except Exception as e:
            raise CustomException(e, sys)
