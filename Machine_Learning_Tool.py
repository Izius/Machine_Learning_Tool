import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Machine_Learning_Tool(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("800x800")
        self.title("Machine Learning Tool")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, Linear_Regression, Logistic_Classification, Linear_Regression_Output,
                  Logistic_Classification_Output):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name, *args):
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name == "Linear_Regression_Output":
            frame.set_output(*args)
        elif page_name == "Logistic_Classification_Output":
            frame.set_output(*args)


    def set_file_path(self, file_path):
        self.file_path = file_path
        return self.file_path


class StartPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="Welcome to Machine Learning Tool!")
        label.pack(side="top", fill="x", pady=10)
        button1 = ttk.Button(self, text="Go to Linear Regression",
                             command=lambda: controller.show_frame("Linear_Regression"))
        button1.pack()
        button2 = ttk.Button(self, text="Go to Logistic Classification",
                             command=lambda: controller.show_frame("Logistic_Classification"))
        button2.pack()

        file_button = ttk.Button(self, text="Select File", command=self.select_file)
        file_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.controller.set_file_path(file_path)


class Linear_Regression(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="This is Linear Regression Page")
        label.pack(side="top", fill="x", pady=10)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack()
        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack()
        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_linear_regression(test_size=test_size_text.get("1.0", "end-1c")))
        button2.pack()

    def fit_linear_regression(self, test_size=0.3):
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size))
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            lin_intercept = model.intercept_
            lin_coefficients = model.coef_
            mae = mean_absolute_error(y_test, y_pred)
            root = np.sqrt(mean_squared_error(y_test, y_pred))

            self.controller.show_frame("Linear_Regression_Output", lin_intercept, lin_coefficients, mae, root)


class Logistic_Classification(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="This is Logistic Classification Page")
        label.pack(side="top", fill="x", pady=10)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack()

        penalty_text = tk.Text(self, height=1, width=10)
        penalty_text.pack()
        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack()

        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_logistic_classification(penalty=penalty_text.get("1.0", "end-1c"),
                                                                                                 test_size=test_size_text.get("1.0", "end-1c")))
        button2.pack()

    def fit_logistic_classification(self, penalty, test_size=0.3):
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size))
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)
            model = LogisticRegression(penalty=f'{penalty}')
            model.fit(scaled_X_train, y_train)
            y_pred = model.predict(scaled_X_test)
            log_intercept = model.intercept_
            log_coefficients = model.coef_
            accuracy_score_ = accuracy_score(y_test, y_pred, normalize=True)
            precision_score_ = precision_score(y_test, y_pred, average='macro')
            recall_score_ = recall_score(y_test, y_pred, average='macro')
            f1_score_ = f1_score(y_test, y_pred, average='macro')

            self.controller.show_frame("Logistic_Classification_Output", log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_)


class Linear_Regression_Output(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = ttk.Label(self, text="Linear Regression Output")
        label.pack(side="top", fill="x", pady=10)

        ttk.Label(self, text="Intercept:").pack()
        self.intercept_label = ttk.Label(self, text="")
        self.intercept_label.pack()

        ttk.Label(self, text="Coefficients:").pack()
        self.coefficient_labels = []

        ttk.Label(self, text="Mean Absolute Error:").pack()
        self.mae_label = ttk.Label(self, text="")
        self.mae_label.pack()

        ttk.Label(self, text="Root Mean Squared Error:").pack()
        self.root_label = ttk.Label(self, text="")
        self.root_label.pack()



    def set_output(self, lin_intercept, lin_coefficients, mae, root):
        self.intercept_label.config(text=lin_intercept)
        for i, coef in enumerate(lin_coefficients):
            label = ttk.Label(self, text=f"Coefficient {i + 1}: {coef}")
            label.pack()
            self.coefficient_labels.append(label)
        self.mae_label.config(text=f'{mae}')
        self.root_label.config(text=f'{root}')
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button1.pack()


class Logistic_Classification_Output(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = ttk.Label(self, text="Logistic Classification Output")
        label.pack(side="top", fill="x", pady=10)

        ttk.Label(self, text="Intercept:").pack()
        self.intercept_label = ttk.Label(self, text="")
        self.intercept_label.pack()

        ttk.Label(self, text="Coefficients:").pack()
        self.coefficient_labels = []

    def set_output(self, log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_):
        self.intercept_label.config(text=log_intercept)
        for i, coef in enumerate(log_coefficients):
            label1 = ttk.Label(self, text=f"Coefficient {i + 1}: {coef}")
            label1.pack()
            self.coefficient_labels.append(label1)
        label2 = ttk.Label(self, text=f"Accuracy Score: {accuracy_score_}")
        label2.pack()
        label3 = ttk.Label(self, text=f"Precision Score: {precision_score_}")
        label3.pack()
        label4 = ttk.Label(self, text=f"Recall Score: {recall_score_}")
        label4.pack()
        label5 = ttk.Label(self, text=f"F1 Score: {f1_score_}")
        label5.pack()
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button1.pack()




if __name__ == "__main__":
    app = Machine_Learning_Tool()
    app.mainloop()