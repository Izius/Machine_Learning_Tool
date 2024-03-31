import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


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


class StartPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="This is the start page")
        label.pack(side="top", fill="x", pady=10)
        button1 = ttk.Button(self, text="Go to Linear Regression",
                             command=lambda: controller.show_frame("Linear_Regression"))
        button2 = ttk.Button(self, text="Go to Logistic Classification",
                             command=lambda: controller.show_frame("Logistic_Classification"))
        button1.pack()
        button2.pack()
        file_button = ttk.Button(self, text="Select File", command=self.select_file)
        file_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.controller.set_file_path(file_path)
            return file_path


class Linear_Regression(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="This is Linear Regression Page")
        label.pack(side="top", fill="x", pady=10)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack()
        button2 = ttk.Button(self, text="Fit!", command=self.fit_linear_regression)
        button2.pack()

    def fit_linear_regression(self):
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            model = LinearRegression()
            model.fit(X, y)

            lin_intercept = model.intercept_
            lin_coefficients = model.coef_

            self.controller.show_frame("Linear_Regression_Output", lin_intercept, lin_coefficients)


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

        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_logistic_classification(penalty_text.get("1.0", "end-1c")))
        button2.pack()

    def fit_logistic_classification(self, penalty):
        print(penalty)
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            model = LogisticRegression(penalty=f'{penalty}')
            model.fit(X, y)
            log_intercept = model.intercept_
            log_coefficients = model.coef_

            self.controller.show_frame("Logistic_Classification_Output", log_intercept, log_coefficients)


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

        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack()

    def set_output(self, lin_intercept, lin_coefficients):
        self.intercept_label.config(text=lin_intercept)
        for i, coef in enumerate(lin_coefficients):
            label = ttk.Label(self, text=f"Coefficient {i + 1}: {coef}")
            label.pack()
            self.coefficient_labels.append(label)


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

        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack()

    def set_output(self, log_intercept, log_coefficients):
        self.intercept_label.config(text=log_intercept)
        for i, coef in enumerate(log_coefficients):
            label = ttk.Label(self, text=f"Coefficient {i + 1}: {coef}")
            label.pack()
            self.coefficient_labels.append(label)


if __name__ == "__main__":
    app = Machine_Learning_Tool()
    app.mainloop()