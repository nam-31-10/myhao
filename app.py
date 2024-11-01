from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import subprocess
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy.stats import chi2
app = Flask(__name__)

# Hàm tính toán biểu thức
def calculate(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return str(e)

# Hàm giải phương trình
def solve_equation(equation, variable):
    var = sp.symbols(variable)
    solution = sp.solve(equation, var)
    return solution

# Hàm tính nguyên hàm
def integrate_function(expression, variable):
    var = sp.symbols(variable)
    expr = sp.sympify(expression)
    integral = sp.integrate(expr, var)
    return integral

def plot_function_and_variation(expression, variable, x_range):
    var = sp.symbols(variable)
    
    try:
        # Chuyển đổi biểu thức
        expression = sp.sympify(expression)
    except Exception as e:
        return f"Biểu thức không hợp lệ: {e}"

    # Tạo hàm từ biểu thức
    function = sp.lambdify(var, expression, "numpy")
    derivative = sp.diff(expression, var)
    derivative_lambdified = sp.lambdify(var, derivative, "numpy")

    x_values = np.linspace(x_range[0], x_range[1], 400)
    y_values = function(x_values)
    y_derivative_values = derivative_lambdified(x_values)

    # Tạo đồ thị với Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='f(x)',
                              hoverinfo='text', text=[f'f({x:.2f}) = {y:.2f}' for x, y in zip(x_values, y_values)]))
    fig.add_trace(go.Scatter(x=x_values, y=y_derivative_values, mode='lines', name="f'(x)",
                              hoverinfo='text', text=[f"f'({x:.2f}) = {dy:.2f}" for x, dy in zip(x_values, y_derivative_values)],
                              line=dict(color='orange')))
    fig.update_layout(title='Đồ thị của f(x) và f\'(x)',
                      xaxis_title=variable,
                      yaxis_title='Giá trị',
                      showlegend=True)

    plot_url = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
    return plot_url

# Hàm vẽ hàm 3D
def plot_surface_function(expression, x_range, y_range):
    x, y = sp.symbols('x y')
    expr = sp.sympify(expression)
    function = sp.lambdify((x, y), expr, "numpy")

    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)

    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Trục X')
    ax.set_ylabel('Trục Y')
    ax.set_zlabel('Trục Z')
    ax.set_title(f'Biểu đồ mặt phẳng của {expression}')

    # Lưu đồ thị vào đối tượng BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Hàm giải đệ quy
def recursive_function(n, function_str, base_case):
    if n == 1:
        return base_case
    else:
        previous_value = recursive_function(n - 1, function_str, base_case)
        function = function_str.replace('U(n)', str(previous_value)).replace('U(n+1)', str(sp.symbols('U')))
        return eval(function)

# Hàm tính giới hạn
def limit_function(expression, variable, point):
    var = sp.symbols(variable)
    expr = sp.sympify(expression)
    limit = sp.limit(expr, var, point)
    return limit

# Miền nghiệm
def plot_feasible_region(inequalities):
    x, y = sp.symbols('x y')
    # Chuyển đổi các bất phương trình từ chuỗi sang biểu thức sympy
    try:
        inequalities_expr = [sp.sympify(ineq) for ineq in inequalities]
    except Exception as e:
        return f"Error in inequalities: {e}"

    # Tạo lưới điểm
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Đánh giá các bất phương trình
    Z = np.ones_like(X)
    for ineq in inequalities_expr:
        # Đánh giá điều kiện cho từng phần tử trong mảng
        condition = np.array([[ineq.subs({x: val_x, y: val_y}) for val_x, val_y in zip(x_row, y_row)]
                              for x_row, y_row in zip(X, Y)])
        Z *= np.where(condition, 1, 0)

    # Tạo đồ thị
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, alpha=0.5, levels=1, colors=['lightblue'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Miền nghiệm của các bất phương trình')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()

    # Lưu đồ thị vào đối tượng BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url
# Hàm tìm ràng buộc cho x
def find_x_constraints(expression, a, b):
    x = sp.symbols('x')
    expr = sp.sympify(expression)

    # Tạo bất phương trình
    inequality1 = expr > a
    inequality2 = expr < b

    # Giải bất phương trình
    solutions1 = sp.solve_univariate_inequality(inequality1, x, relational=False)
    solutions2 = sp.solve_univariate_inequality(inequality2, x, relational=False)

    # Tính giao của hai miền
    constraints = sp.Intersection(solutions1, solutions2)
    return constraints

# Hàm tìm cực trị của hàm hai biến
def find_extrema_of_two_variables(expression):
    x, y = sp.symbols('x y')
    expr = sp.sympify(expression)

    # Tính đạo hàm riêng
    derivative_x = sp.diff(expr, x)
    derivative_y = sp.diff(expr, y)

    # Giải hệ phương trình
    critical_points = sp.solve([derivative_x, derivative_y], (x, y))

    # Tính ma trận Hessian
    hessian_matrix = sp.Matrix([[sp.diff(derivative_x, x), sp.diff(derivative_x, y)],
                                 [sp.diff(derivative_y, x), sp.diff(derivative_y, y)]])
    results = []
    for point in critical_points:
        hessian_value = hessian_matrix.subs({x: point[0], y: point[1]})
        eigenvalues = hessian_value.eigenvals()
        
        if all(value > 0 for value in eigenvalues):  # Cực tiểu
            results.append((point, "Cực tiểu"))
        elif all(value < 0 for value in eigenvalues):  # Cực đại
            results.append((point, "Cực đại"))
        else:
            results.append((point, "Điểm yên ngựa"))
    return results
# Hàm vẽ phân phối chuẩn
def plot_normal_distribution(mean=0, std_dev=1):
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Phân phối chuẩn', color='blue')
    plt.fill_between(x, y, alpha=0.2, color='blue')
    plt.title('Phân phối chuẩn')
    plt.xlabel('Giá trị')
    plt.ylabel('Tần suất')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Hàm vẽ phân phối chi bình phương
def plot_chi_squared_distribution(df):
    x = np.linspace(0, 3 * df, 1000)
    y = chi2.pdf(x, df)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f'Chi-squared Distribution (df={df})', color='green')
    plt.fill_between(x, y, alpha=0.2, color='green')
    plt.title('Phân phối chi bình phương')
    plt.xlabel('Giá trị')
    plt.ylabel('Tần suất')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


# Hàm vẽ phân phối Poisson
def plot_poisson_distribution(lmbda):
    x = np.arange(0, 20)
    y = poisson.pmf(x, lmbda)

    plt.figure(figsize=(8, 4))
    plt.bar(x, y, label=f'Poisson Distribution (λ={lmbda})', color='purple', alpha=0.7)
    plt.title('Phân phối Poisson')
    plt.xlabel('Giá trị')
    plt.ylabel('Xác suất')
    plt.xticks(x)
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url
# Route chính
@app.route('/')
def index():
    return render_template('index.html')

# Route tính toán
@app.route('/calculate', methods=['GET', 'POST'])
def calculate_view():
    result = None
    plot_url = None
    if request.method == 'POST':
        expression = request.form['expression']
        result = calculate(expression)

        # Gọi Manim để tạo hình ảnh
        try:
            subprocess.run(["manim", "-pql", "render.py", "MathScene"], check=True)
            os.rename("media/images/MathScene.png", "static/images/rendered_image.png")
            plot_url = "/static/images/rendered_image.png"
        except Exception as e:
            print(f"Error generating image: {e}")

    return render_template('calculate.html', result=result, plot_url=plot_url)

# Route giải phương trình
@app.route('/solve', methods=['GET', 'POST'])
def solve_view():
    result = None
    if request.method == 'POST':
        equation = request.form['equation']
        variable = request.form['variable']
        result = solve_equation(equation, variable)
    return render_template('solve.html', result=result)

# Route vẽ hàm
@app.route('/plot', methods=['GET', 'POST'])
def plot_view():
    plot_url = None
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        x_start = float(request.form['x_start'])
        x_end = float(request.form['x_end'])
        plot_url = plot_function_and_variation(expression, variable, (x_start, x_end))
    return render_template('plot.html', plot_url=plot_url)

# Route giải nguyên hàm
@app.route('/integrate', methods=['GET', 'POST'])
def integrate_view():
    integral = None
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        integral = integrate_function(expression, variable)
    return render_template('integrate.html', integral=integral)

# Route giải đệ quy
@app.route('/recursive', methods=['GET', 'POST'])
def recursive_view():
    result = None
    if request.method == 'POST':
        base_case = float(request.form['base_case'])
        function_str = request.form['func_str']
        n = int(request.form['n'])
        result = recursive_function(n, function_str, base_case)
    return render_template('recursive.html', result=result)

# Route tính giới hạn
@app.route('/limit', methods=['GET', 'POST'])
def limit_view():
    limit = None
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        point = float(request.form['point'])
        limit = limit_function(expression, variable, point)
    return render_template('limit.html', limit=limit)

# Route vẽ hàm 3D
@app.route('/surface', methods=['GET', 'POST'])
def surface_view():
    plot_url = None
    if request.method == 'POST':
        expression = request.form['expression']
        x_start = float(request.form['x_start'])
        x_end = float(request.form['x_end'])
        y_start = float(request.form['y_start'])
        y_end = float(request.form['y_end'])
        plot_url = plot_surface_function(expression, (x_start, x_end), (y_start, y_end))
    return render_template('surface.html', plot_url=plot_url)

# Route miền nghiệm
@app.route('/feasible_region', methods=['GET', 'POST'])
def feasible_region_view():
    plot_url = None
    if request.method == 'POST':
        inequalities = request.form['inequalities'].strip().split('\n')  # Nhận dữ liệu từ textarea
        if inequalities:  # Kiểm tra xem có bất phương trình nào không
            plot_url = plot_feasible_region(inequalities)
    return render_template('feasible_region.html', plot_url=plot_url)

# Route tìm ràng buộc cho x
@app.route('/constraints', methods=['GET', 'POST'])
def constraints_view():
    constraints = None
    if request.method == 'POST':
        expression = request.form['expression']
        a = float(request.form['a'])
        b = float(request.form['b'])
        constraints = find_x_constraints(expression, a, b)
    return render_template('constraints.html', constraints=constraints)

# Route tìm cực trị
@app.route('/extrema', methods=['GET', 'POST'])
def extrema_view():
    extrema = None
    if request.method == 'POST':
        expression = request.form['expression']
        extrema = find_extrema_of_two_variables(expression)
    return render_template('extrema.html', extrema=extrema)
# Route vẽ phân phối chuẩn
@app.route('/normal_distribution', methods=['GET', 'POST'])
def normal_distribution_view():
    plot_url = None
    if request.method == 'POST':
        mean = float(request.form['mean'])
        std_dev = float(request.form['std_dev'])
        plot_url = plot_normal_distribution(mean, std_dev)
    return render_template('normal_distribution.html', plot_url=plot_url)

# Route vẽ phân phối chi bình phương

@app.route('/chi_squared_distribution', methods=['GET', 'POST'])
def chi_squared_distribution_view():
    plot_url = None
    if request.method == 'POST':
        df = int(request.form['df'])
        plot_url = plot_chi_squared_distribution(df)
    return render_template('chi_squared_distribution.html', plot_url=plot_url)
# Route vẽ phân phối Poisson
@app.route('/poisson_distribution', methods=['GET', 'POST'])
def poisson_distribution_view():
    plot_url = None
    if request.method == 'POST':
        lmbda = float(request.form['lmbda'])
        plot_url = plot_poisson_distribution(lmbda)
    return render_template('poisson_distribution.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
