import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QHBoxLayout
import pyqtgraph as pg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def create_model(data, x_col, y_col):
    X = data[[x_col]]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

class RegressionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.load_button = QPushButton('Load Data', self)
        self.load_button.clicked.connect(self.load_data_file)

        self.data_label = QLabel('No data loaded', self)

        self.x_col_input = QLineEdit(self)
        self.y_col_input = QLineEdit(self)

        self.train_button = QPushButton('Train Model', self)
        self.train_button.clicked.connect(self.train_model)

        self.result_label = QLabel('', self)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Data Visualization")

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.data_label)

        col_input_layout = QHBoxLayout()
        col_input_layout.addWidget(QLabel('X Column:', self))
        col_input_layout.addWidget(self.x_col_input)
        col_input_layout.addWidget(QLabel('Y Column:', self))
        col_input_layout.addWidget(self.y_col_input)

        layout.addLayout(col_input_layout)
        layout.addWidget(self.train_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)
        self.setWindowTitle('Linear Regression Tool')

    def load_data_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.data = load_data(file_name)
            self.data_label.setText(f'Data Loaded: {file_name}')

    def train_model(self):
        x_col = self.x_col_input.text()
        y_col = self.y_col_input.text()

        if not self.data.empty and x_col in self.data.columns and y_col in self.data.columns:
            model, mse = create_model(self.data, x_col, y_col)
            self.result_label.setText(f'Model Trained. Mean Squared Error: {mse:.2f}')

            X = self.data[[x_col]].values
            y = self.data[y_col].values
            y_pred = model.predict(X)

            self.plot_widget.clear()
            self.plot_widget.plot(X.ravel(), y_pred, pen='r')
            self.plot_widget.scatterPlot(X.ravel(), y, symbol='o', size=10)


        else:
            self.result_label.setText('Error: Invalid column names or no data loaded')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RegressionApp()
    window.show()
    sys.exit(app.exec_())

