# Body-Movement-Prediction-with-Time-Series

This is a machine learning project that uses YOLO pose estimation and time series forecasting to predict human body movements in video frames.

## Usage

- **Pose Estimation**: Extract body key points per frame from video using YOLOv11
- **Time Series Analysis**: Conduct ACF and PACF analysis to determine optimal lag values
- **Strategy Comparison**: Investigate recursive vs direct multi-step forecasting strategies
- **Multi-Step Forecasting**: Predict future body positions of limited frames ahead
- **Skeleton Visualization**: Visualize and compare actual vs predicted poses formed by keypoints

## Result Demo
predicted - predicted keypoints coordinates
green circles - actual keypoints coordinates
