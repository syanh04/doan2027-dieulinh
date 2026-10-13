"""
forecaster.py - Mo-dun du bao luu luong giao thong thong minh.
Su dung mo hinh Holt's Linear Trend voi Damped Trend va Moving Average.
Tuyet doi tuan thu nguyen tac du lieu:
- Khong gia lap du lieu sin-wave.
- Neu lich su < 3 diem: tra ve status "collecting", khong du bao doan mo.
- Ho tro tinh toan Backtest (MAE, RMSE, MAPE) tren tap du lieu thuc te cua camera do.
"""
import time
import numpy as np


class TrafficForecaster:
    def __init__(self, alpha=0.35, beta=0.15, phi=0.92):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi

    def predict(self, history, horizon_minutes=30, model_type="holt_winters"):
        """
        Du bao luu luong trong tuong lai dua tren history thuc te.
        history: danh sach [{ "timestamp": int, "flow": float, "time_str": str, "video_time_sec": float }]
        horizon_minutes: 15, 30, hoac 60 phut.
        model_type: 'holt_winters' hoac 'moving_average'
        """
        if not history or len(history) < 3:
            return {
                "status": "collecting",
                "message": "Dang thu thap du lieu (Can toi thieu 3 phut du lieu that)",
                "history": history or [],
                "forecast": [],
                "lower_bound": [],
                "upper_bound": [],
                "metrics": {"mae": None, "rmse": None, "mape": None, "sample_count": len(history) if history else 0},
                "horizon_minutes": horizon_minutes
            }

        flows = np.array([float(h["flow"]) for h in history if h.get("flow") is not None], dtype=np.float64)
        if len(flows) < 3:
            return {
                "status": "collecting",
                "message": "Cac diem luu luong chua du gia tri hop le",
                "history": history,
                "forecast": [],
                "lower_bound": [],
                "upper_bound": [],
                "metrics": {"mae": None, "rmse": None, "mape": None, "sample_count": len(flows)},
                "horizon_minutes": horizon_minutes
            }

        n = len(flows)

        # 1. Tinh Backtest Metrics tren tap lich su (Fit & In-sample Error)
        backtest_metrics = self._calculate_backtest_metrics(flows)

        # 2. Xay dung du bao tuy theo mo hinh
        last_item = history[-1]
        last_time = int(last_item.get("timestamp", int(time.time())))
        
        step_minutes = 1 if horizon_minutes <= 15 else (2 if horizon_minutes <= 30 else 5)
        num_steps = max(5, horizon_minutes // step_minutes)

        forecast_points = []
        lower_points = []
        upper_points = []

        if model_type == "moving_average":
            window_size = min(5, n)
            ma_val = float(np.mean(flows[-window_size:]))
            std_err = float(np.std(flows[-window_size:])) if window_size > 1 else 10.0
            std_err = max(5.0, std_err)

            for step in range(1, num_steps + 1):
                future_sec = last_time + step * step_minutes * 60
                future_time_str = time.strftime("%H:%M:%S", time.localtime(future_sec))
                ci_spread = 1.96 * std_err * np.sqrt(step)

                forecast_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(ma_val, 1)})
                lower_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(max(0.0, ma_val - ci_spread), 1)})
                upper_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(ma_val + ci_spread, 1)})

        else:
            # Holt's Damped Trend
            level = flows[0]
            trend = (flows[-1] - flows[0]) / max(1, n - 1)
            fitted = np.zeros(n)
            fitted[0] = level

            for t in range(1, n):
                prev_level = level
                level = self.alpha * flows[t] + (1 - self.alpha) * (prev_level + self.phi * trend)
                trend = self.beta * (level - prev_level) + (1 - self.beta) * (self.phi * trend)
                fitted[t] = prev_level + self.phi * trend

            residuals = flows - fitted
            std_err = float(np.std(residuals)) if len(residuals) > 1 else 15.0
            std_err = max(5.0, std_err)

            curr_level = level
            curr_trend = trend

            for step in range(1, num_steps + 1):
                future_sec = last_time + step * step_minutes * 60
                future_time_str = time.strftime("%H:%M:%S", time.localtime(future_sec))

                damped_trend_sum = sum((self.phi ** i) for i in range(1, step + 1)) * curr_trend
                y_hat = max(0.0, curr_level + damped_trend_sum)
                ci_spread = 1.96 * std_err * np.sqrt(step)

                forecast_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(float(y_hat), 1)})
                lower_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(float(max(0.0, y_hat - ci_spread)), 1)})
                upper_points.append({"time_str": future_time_str, "timestamp": future_sec, "flow": round(float(y_hat + ci_spread), 1)})

        return {
            "status": "ready",
            "message": f"Mo hinh {model_type.upper()} san sang voi {n} mau quan sat",
            "history": history[-30:],
            "forecast": forecast_points,
            "lower_bound": lower_points,
            "upper_bound": upper_points,
            "metrics": backtest_metrics,
            "horizon_minutes": horizon_minutes
        }

    def _calculate_backtest_metrics(self, flows):
        """
        Tinh MAE, RMSE, MAPE qua phuong phap 1-step ahead cross-validation hoac fit-error.
        """
        n = len(flows)
        if n < 4:
            return {"mae": None, "rmse": None, "mape": None, "sample_count": n}

        # Chia tap train / val: dung 70% dau de du doan 30% cuoi (toi thieu 2 diem test)
        split = max(2, int(n * 0.7))
        train = flows[:split]
        test = flows[split:]

        if len(test) == 0:
            return {"mae": None, "rmse": None, "mape": None, "sample_count": n}

        # Fit tren train
        level = train[0]
        trend = (train[-1] - train[0]) / max(1, len(train) - 1)
        for t in range(1, len(train)):
            prev_level = level
            level = self.alpha * train[t] + (1 - self.alpha) * (prev_level + self.phi * trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * (self.phi * trend)

        # Predict tung buoc tren test
        predictions = []
        for step in range(1, len(test) + 1):
            damped_sum = sum((self.phi ** i) for i in range(1, step + 1)) * trend
            pred = max(0.0, level + damped_sum)
            predictions.append(pred)

        predictions = np.array(predictions)
        test_arr = np.array(test)

        errors = np.abs(test_arr - predictions)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean((test_arr - predictions) ** 2)))

        # Tranh chia cho 0 khi tinh MAPE
        nonzero_mask = test_arr > 0
        if np.any(nonzero_mask):
            mape = float(np.mean(errors[nonzero_mask] / test_arr[nonzero_mask]) * 100.0)
        else:
            mape = None

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2) if mape is not None else None,
            "sample_count": n
        }


# Global singleton instance
forecaster = TrafficForecaster()
