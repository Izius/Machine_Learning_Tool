import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image, ImageTk
import json


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

        for F in (StartPage, Linear_Regression, Logistic_Classification, Polynomial_Regression, KNNClassification, Linear_Regression_Output,
                  Logistic_Classification_Output, Polynomial_Regression_Output, KNNClassification_Output):

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

        elif page_name == "Polynomial_Regression_Output":
            frame.set_output(*args)

        elif page_name == "KNNClassification_Output":
                frame.set_output(*args)

    def set_file_path(self, file_path):

        self.file_path = file_path

        return self.file_path


class StartPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label = ttk.Label(self, text="Welcome to Machine Learning Tool!", background="lightblue")
        label.pack(side="top", fill="x", pady=10)

        button1 = ttk.Button(self, text="Go to Linear Regression",
                             command=lambda: controller.show_frame("Linear_Regression"))
        button1.pack()

        button2 = ttk.Button(self, text="Go to Logistic Classification",
                             command=lambda: controller.show_frame("Logistic_Classification"))
        button2.pack()

        button3 = ttk.Button(self, text="Go to Polynomial Regression",
                             command=lambda: controller.show_frame("Polynomial_Regression"))
        button3.pack()

        button4 = ttk.Button(self, text="Go to KNN Classification",
                             command=lambda: controller.show_frame("KNNClassification"))
        button4.pack()

        file_button = ttk.Button(self, text="Select File", command=self.select_file)
        file_button.pack()

        self.display_image()

    def display_image(self):
        im1 = Image.open('images/ml.png')
        newsize = (500, 300)
        im1 = im1.resize(newsize)
        self.iim1 = ImageTk.PhotoImage(im1)
        image_label = tk.Label(self, image=self.iim1)
        image_label.pack(pady=100)

    def select_file(self):

        file_path = filedialog.askopenfilename()

        if file_path:
            self.controller.set_file_path(file_path)


class Linear_Regression(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label1 = ttk.Label(self, text="Linear regression", background="lightblue")
        label1.pack(side="top", fill="x", pady=20)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack(side='top', pady=10)

        label2 = ttk.Label(self, text="Enter test size:", background="lightblue")
        label2.pack()

        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack(pady=5)

        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_linear_regression(test_size=test_size_text.get("1.0", "end-1c")))
        button2.pack(pady=10)

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
        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label1 = ttk.Label(self, text="Logistic classification (using Stochastic Average Gradient Ascent)", background="lightblue")
        label1.pack(side="top", fill="x", pady=10)

        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack(pady=10)

        label2 = ttk.Label(self, text="Enter penalty (l2, l1 or elasticnet):", background="lightblue")
        label2.pack(pady=5)

        penalty_text = tk.Text(self, height=1, width=10)
        penalty_text.pack(pady=10)

        label3 = ttk.Label(self, text="Enter test size:", background="lightblue")
        label3.pack(pady=5)

        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack(pady=10)

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

            model = LogisticRegression(penalty=f'{penalty}', solver='saga')
            model.fit(scaled_X_train, y_train)

            y_pred = model.predict(scaled_X_test)

            log_intercept = model.intercept_
            log_coefficients = model.coef_

            accuracy_score_ = accuracy_score(y_test, y_pred, normalize=True)
            precision_score_ = precision_score(y_test, y_pred, average='macro')
            recall_score_ = recall_score(y_test, y_pred, average='macro')
            f1_score_ = f1_score(y_test, y_pred, average='macro')

            self.controller.show_frame("Logistic_Classification_Output", log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_)


class Polynomial_Regression(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label1 = ttk.Label(self, text="Polynomial regression", background="lightblue")
        label1.pack(side="top", fill="x", pady=20)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack(side='top', pady=10)

        label2 = ttk.Label(self, text="Enter test size:", background="lightblue")
        label2.pack()

        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack(pady=5)

        degree_label = ttk.Label(self, text="Enter the order of the polynomial:", background="lightblue")
        degree_label.pack()

        degree_text = tk.Text(self, height=1, width=10)
        degree_text.pack(pady=5)

        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_polynomial_regression(test_size=test_size_text.get("1.0", "end-1c"),
                             degree=degree_text.get("1.0", "end-1c")))
        button2.pack(pady=10)

    def fit_polynomial_regression(self, test_size=0.3, degree=2):
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            polynomial_converter = PolynomialFeatures(degree=int(degree), include_bias=False)
            poly_features = polynomial_converter.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=float(test_size))

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            intercept = model.intercept_
            coefficients = model.coef_

            mae = mean_absolute_error(y_test, y_pred)
            root = np.sqrt(mean_squared_error(y_test, y_pred))

            self.controller.show_frame("Polynomial_Regression_Output", intercept, coefficients, mae, root, degree)

class KNNClassification(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label1 = ttk.Label(self, text="KNN classification", background="lightblue")
        label1.pack(side="top", fill="x", pady=20)
        button1 = ttk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame("StartPage"))
        button1.pack(side='top', pady=10)

        label2 = ttk.Label(self, text="Enter test size:", background="lightblue")
        label2.pack()

        test_size_text = tk.Text(self, height=1, width=10)
        test_size_text.pack(pady=5)

        label3 = ttk.Label(self, text="Enter number of neighbors:", background="lightblue")
        label3.pack()

        n_neighbors_text = tk.Text(self, height=1, width=10)
        n_neighbors_text.pack(pady=5)

        button2 = ttk.Button(self, text="Fit!", command=lambda: self.fit_knn_classification(test_size=test_size_text.get("1.0", "end-1c"),
                                                                                                 n_neighbors=n_neighbors_text.get("1.0", "end-1c")))
        button2.pack()

    def fit_knn_classification(self, test_size=0.3, n_neighbors=5):
        file_path = self.controller.file_path
        if file_path:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size))
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)

            model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
            model.fit(scaled_X_train, y_train)

            y_pred = model.predict(scaled_X_test)

            accuracy_score_ = accuracy_score(y_test, y_pred, normalize=True)
            precision_score_ = precision_score(y_test, y_pred, average='macro')
            recall_score_ = recall_score(y_test, y_pred, average='macro')
            f1_score_ = f1_score(y_test, y_pred, average='macro')

            self.controller.show_frame("KNNClassification_Output", accuracy_score_, precision_score_, recall_score_, f1_score_)


class Linear_Regression_Output(ttk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label = ttk.Label(self, text="Linear Regression Output", background="lightblue")
        label.pack(side="top", fill="x", pady=10)

        self.intercept_label = ttk.Label(self, text="", background="lightblue")
        self.intercept_label.pack()

        self.mae_label = ttk.Label(self, text="", background="lightblue")
        self.mae_label.pack()

        self.root_label = ttk.Label(self, text="", background="lightblue")
        self.root_label.pack()

        self.coefficient_labels = []


    def set_output(self, lin_intercept, lin_coefficients, mae, root):

        self.intercept_label.config(text=f'Intercept: {round(lin_intercept, 2)}')

        for i, coef in enumerate(lin_coefficients):
            label = ttk.Label(self, text=f"Coefficient {i + 1}: {round(coef, 2)}", background="lightblue")
            label.pack()
            self.coefficient_labels.append(label)

        self.mae_label.config(text=f'Mean Absolute Error: {round(mae, 2)}')
        self.root_label.config(text=f'Root Mean Squared Error: {round(root, 2)}')

        button1 = ttk.Button(self, text='Export to txt file',
                             command=lambda: self.export_to_txt(lin_intercept, lin_coefficients, mae, root))
        button1.pack()

        button2 = ttk.Button(self, text='Export to JSON file',
                             command=lambda: self.export_to_json(lin_intercept, lin_coefficients, mae, root))
        button2.pack()

        button3 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button3.pack()

    def export_to_txt(self, lin_intercept, lin_coefficients, mae, root):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])

        if file_path:
            with open(file_path, 'w') as file:
                file.write(f'Intercept: {lin_intercept}\n')
                file.write(f'Mean Absolute Error: {mae}\n')
                file.write(f'Root Mean Squared Error: {root}\n')
                for i, coef in enumerate(lin_coefficients):
                    file.write(f'Coefficient {i + 1}: {coef}\n')

                file.close()

    def export_to_json(self, lin_intercept, lin_coefficients, mae, root):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])

        if file_path:
            data = {
                "Intercept": lin_intercept,
                "Mean Absolute Error": mae,
                "Root Mean Squared Error": root,
            }
            for i, coef in enumerate(lin_coefficients):
                data[f'Coefficient {i + 1}'] = coef

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
                file.close()


class Logistic_Classification_Output(ttk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller
        label = ttk.Label(self, text="Logistic Classification Output", background="lightblue")
        label.pack(side="top", fill="x", pady=10)

        self.intercept_label = ttk.Label(self, text="")
        self.intercept_label.pack()

        self.coefficient_labels = []

    def set_output(self, log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_):

        for i in range(len(log_intercept)):
            log_intercept[i] = str(round(log_intercept[i], 2))

        self.intercept_label.config(text=f'Intercept: '+', '.join(log_intercept), background="lightblue")

        for i, coef in enumerate(log_coefficients):

            label1 = ttk.Label(self, text=f"Coefficient {i + 1}: {round(coef, 2)}", background="lightblue")
            label1.pack()
            self.coefficient_labels.append(label1)

        label2 = ttk.Label(self, text=f"Accuracy Score: {round(accuracy_score_, 2)}", background="lightblue")
        label2.pack()

        label3 = ttk.Label(self, text=f"Precision Score: {round(precision_score_, 2)}", background="lightblue")
        label3.pack()

        label4 = ttk.Label(self, text=f"Recall Score: {round(recall_score_, 2)}", background="lightblue")
        label4.pack()

        label5 = ttk.Label(self, text=f"F1 Score: {round(f1_score_, 2)}", background="lightblue")
        label5.pack()

        button1 = ttk.Button(self, text='Export to txt file',
                             command=lambda: self.export_to_txt(log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_))
        button1.pack()

        button2 = ttk.Button(self, text='Export to JSON file',
                             command=lambda: self.export_to_json(log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_))
        button2.pack()

        button3 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button3.pack()

    def export_to_txt(self, log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])

        if file_path:
            with open(file_path, 'w') as file:
                file.write(f'Intercept: {log_intercept}\n')
                file.write(f'Accuracy Score: {accuracy_score_}\n')
                file.write(f'Precision Score: {precision_score_}\n')
                file.write(f'Recall Score: {recall_score_}\n')
                file.write(f'F1 Score: {f1_score_}\n')
                for i, coef in enumerate(log_coefficients):
                    file.write(f'Coefficient {i + 1}: {coef}\n')

                file.close()

    def export_to_json(self, log_intercept, log_coefficients, accuracy_score_, precision_score_, recall_score_, f1_score_):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])

        if file_path:
            data = {
                "Intercept": log_intercept,
                "Accuracy Score": accuracy_score_,
                "Precision Score": precision_score_,
                "Recall Score": recall_score_,
                "F1 Score": f1_score_,
            }
            for i, coef in enumerate(log_coefficients):
                data[f'Coefficient {i + 1}'] = coef

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
                file.close()


class Polynomial_Regression_Output(ttk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label = ttk.Label(self, text="Polynomial Regression Output", background="lightblue")
        label.pack(side="top", fill="x", pady=10)

        self.intercept_label = ttk.Label(self, text="")
        self.intercept_label.pack()

        self.mae_label = ttk.Label(self, text="")
        self.mae_label.pack()

        self.root_label = ttk.Label(self, text="")
        self.root_label.pack()

        self.coefficient_labels = []

    def set_output(self, intercepts, coefficients, mae, root, degree):

        i = 1
        j = 0
        for coef in coefficients:

            if j == int(degree)+1:
                j = 0
                i += 1

            label1 = ttk.Label(self, text=f"Coeficient {i}{j}: {round(coef, 2)}", background="lightblue")
            label1.pack()
            self.coefficient_labels.append(label1)
            j += 1

        self.mae_label.config(text=f'Mean Absolute Error: {round(mae, 2)}', background="lightblue")
        self.root_label.config(text=f'Root Mean Squared Error: {round(root, 2)}', background="lightblue")

        button1 = ttk.Button(self, text='Export to txt file',
                                command=lambda: self.export_to_txt(intercepts, coefficients, mae, root, degree))
        button1.pack()

        button2 = ttk.Button(self, text='Export to JSON file',
                                command=lambda: self.export_to_json(intercepts, coefficients, mae, root, degree))
        button2.pack()

        button3 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button3.pack()

    def export_to_txt(self, intercepts, coefficients, mae, root, degree):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])

        if file_path:
            with open(file_path, 'w') as file:
                file.write(f'Mean Absolute Error: {mae}\n')
                file.write(f'Root Mean Squared Error: {root}\n')
                for i, coef in enumerate(coefficients):
                    file.write(f'Coefficient {i + 1}: {coef}\n')

                file.close()

    def export_to_json(self, intercepts, coefficients, mae, root, degree):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])

        if file_path:
            data = {
                "Mean Absolute Error": mae,
                "Root Mean Squared Error": root,
            }
            for i, coef in enumerate(coefficients):
                data[f'Coefficient {i + 1}'] = coef

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
                file.close()

class KNNClassification_Output(ttk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent, style="Frame.TFrame")
        self.controller = controller

        label = ttk.Label(self, text="KNN Classification Output", background="lightblue")
        label.pack(side="top", fill="x", pady=10)

        self.accuracy_label = ttk.Label(self, text="")
        self.accuracy_label.pack()

        self.precision_label = ttk.Label(self, text="")
        self.precision_label.pack()

        self.recall_label = ttk.Label(self, text="")
        self.recall_label.pack()

        self.f1_label = ttk.Label(self, text="")
        self.f1_label.pack()

    def set_output(self, accuracy_score_, precision_score_, recall_score_, f1_score_):

        self.accuracy_label.config(text=f'Accuracy Score: {round(accuracy_score_, 2)}', background="lightblue")
        self.precision_label.config(text=f'Precision Score: {round(precision_score_, 2)}', background="lightblue")
        self.recall_label.config(text=f'Recall Score: {round(recall_score_, 2)}', background="lightblue")
        self.f1_label.config(text=f'F1 Score: {round(f1_score_, 2)}', background="lightblue")

        button1 = ttk.Button(self, text='Export to txt file',
                                command=lambda: self.export_to_txt(accuracy_score_, precision_score_, recall_score_, f1_score_))
        button1.pack()

        button2 = ttk.Button(self, text='Export to JSON file',
                                command=lambda: self.export_to_json(accuracy_score_, precision_score_, recall_score_, f1_score_))
        button2.pack()

        button3 = ttk.Button(self, text="Go to Start Page", command=lambda: self.controller.show_frame("StartPage"))
        button3.pack()

    def export_to_txt(self, accuracy_score_, precision_score_, recall_score_, f1_score_):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])

        if file_path:
            with open(file_path, 'w') as file:
                file.write(f'Accuracy Score: {accuracy_score_}\n')
                file.write(f'Precision Score: {precision_score_}\n')
                file.write(f'Recall Score: {recall_score_}\n')
                file.write(f'F1 Score: {f1_score_}\n')

                file.close()

    def export_to_json(self, accuracy_score_, precision_score_, recall_score_, f1_score_):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])

        if file_path:
            data = {
                "Accuracy Score": accuracy_score_,
                "Precision Score": precision_score_,
                "Recall Score": recall_score_,
                "F1 Score": f1_score_,
            }

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
                file.close()

if __name__ == "__main__":

    app = Machine_Learning_Tool()
    style = ttk.Style()
    style.configure(style="Frame.TFrame", background="lightblue")
    app.mainloop()
