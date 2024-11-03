#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define M_PI 3.14159265358979323846

class Matrix {  // 矩阵
public:
    std::vector<std::vector<double>> data;
    int rows, cols;
    // 构造函数
    Matrix(int rows, int cols, double init=NULL) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, init)) {
        if (rows * cols == 0)
            throw std::invalid_argument("Dimention can't be 0");
        if (init == NULL) {
            // 生成随机数填充矩阵
            for (int i = 0; i < rows; ++i) 
                for (int j = 0; j < cols; ++j) 
                    data[i][j] = std::rand() / static_cast<float>(RAND_MAX);
        }
    }
    explicit Matrix(std::vector<std::vector<double>> mat_data) : rows(mat_data.size()), cols(mat_data[0].size()), data(mat_data) {}
    explicit Matrix(std::vector<double> vec_data) : rows(1), cols(vec_data.size()) {
        data.push_back(vec_data);
    }

    std::vector<double> to_vector() {
        if (cols == 1){
            std::vector<double> res;
            for (auto x : data)
                res.push_back(x[0]);
            return res;
        }
        else if (rows == 1)
            return data[0];
        else
            throw std::invalid_argument("Matrix can not be transform into vector");
    }

    double& operator()(int row, int col){ // 获取矩阵的元素
        if (row >= rows)
            throw std::invalid_argument("Invalid row index");
        if (col >= cols)
            throw std::invalid_argument("Invalid col index");
        return data[row][col];
    }
    
    double operator()(int row, int col) const { // 获取矩阵的元素
        if (row >= rows)
            throw std::invalid_argument("Invalid row index");
        if (col >= cols)
            throw std::invalid_argument("Invalid col index");
        return data[row][col];
    } 

    Matrix transpose() const{
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(j, i) = data[i][j];
        return result;
    }

    Matrix sqrt() const const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = std::sqrt(data[i][j]);
        return result;
    }

    Matrix pow(int p) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = std::pow(data[i][j], p);
        return result;
    }

    Matrix log() const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = std::log(std::max(data[i][j], 1e-15));
        return result;
    }

    Matrix exp() const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(i, j) = std::exp(data[i][j]);
        return result;
    }

    double mean() const{
        double res = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res += data[i][j];
        return res / (rows * cols);
    }

    Matrix sum(int dim) const {
        if (dim == 0) {
            Matrix res(1, cols, 0.0);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    res(0, j) += data[i][j];
            return res;
        }
        else {
            Matrix res(rows, 1, 0.0);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    res(i, 0) += data[i][j];
            return res;
        }
    }

    void reset(double init=0.0) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = init;
    }

    const Matrix mat(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix A's columns must be equal to Matrix B's rows");
        Matrix result(rows, other.cols, 0.0);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                for (int k = 0; k < cols; ++k)
                    result(i, j) += data[i][k] * other(k, j);
        return result;
    }

    Matrix operator=(const Matrix& other){
        if (this != &other) {
            cols = other.cols; 
            rows = other.rows;
            data = other.data;
        }
        return *this;
    }

    // 重载加法运算符
    const Matrix operator+(const Matrix& other) const {
        if (rows == other.rows && cols == other.cols){
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] + other.data[i][j];
            return result;
        }
        else if (rows == other.rows && other.cols == 1) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] + other.data[i][0];
            return result;
        }
        else if (rows == other.rows && cols == 1) {
            Matrix result(rows, other.cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < other.cols; ++j)
                    result.data[i][j] = data[i][0] + other.data[i][j];
            return result;
        }
        else if (rows == 1 && cols == other.cols) {
            Matrix result(other.rows, cols);
            for (int i = 0; i < other.rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[0][j] = data[i][j] + other.data[i][j];
            return result;
        }
        else if (other.rows == 1 && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] + other.data[0][j];
            return result;
        }
        else
            throw std::invalid_argument("Matrix dimensions must match");
    }

    // 重载减法运算符
    const Matrix operator-(const Matrix& other) const {

        if (rows == other.rows && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] - other.data[i][j];
            return result;
        }
        else if (rows == other.rows && other.cols == 1) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] - other.data[i][0];
            return result;
        }
        else if (rows == other.rows && cols == 1) {
            Matrix result(rows, other.cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < other.cols; ++j)
                    result.data[i][j] = data[i][0] - other.data[i][j];
            return result;
        }
        else if (rows == 1 && cols == other.cols) {
            Matrix result(other.rows, cols);
            for (int i = 0; i < other.rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[0][j] = data[i][j] - other.data[i][j];
            return result;
        }
        else if (other.rows == 1 && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] - other.data[0][j];
            return result;
        }
        else
            throw std::invalid_argument("Matrix dimensions must match");
    }

    // 重载乘法运算符
    const Matrix operator*(const Matrix& other) const {
        if (rows == other.rows && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] * other.data[i][j];
            return result;
        }
        else if (rows == other.rows && other.cols == 1) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] * other.data[i][0];
            return result;
        }
        else if (rows == other.rows && cols == 1) {
            Matrix result(rows, other.cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < other.cols; ++j)
                    result.data[i][j] = data[i][0] * other.data[i][j];
            return result;
        }
        else if (rows == 1 && cols == other.cols) {
            Matrix result(other.rows, cols);
            for (int i = 0; i < other.rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[0][j] = data[i][j] * other.data[i][j];
            return result;
        }
        else if (other.rows == 1 && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result.data[i][j] = data[i][j] * other.data[0][j];
            return result;
        }
        else
            throw std::invalid_argument("Matrix dimensions must match");
    }

    // 重载除法运算符
    const Matrix operator/(const Matrix& other) const {
        if (rows == other.rows && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    if (other.data[i][j] == 0)
                        throw std::invalid_argument("Cannot divide by zero");
                    result.data[i][j] = data[i][j] / other.data[i][j];
                }
            return result;
        }
        else if (rows == other.rows && other.cols == 1) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    if (other.data[i][0] == 0)
                        throw std::invalid_argument("Cannot divide by zero");
                    result.data[i][j] = data[i][j] / other.data[i][0];
                }
            return result;
        }
        else if (rows == other.rows && cols == 1) {
            Matrix result(rows, other.cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < other.cols; ++j) {
                    if (other.data[i][j] == 0)
                        throw std::invalid_argument("Cannot divide by zero");
                    result.data[i][j] = data[i][0] / other.data[i][j];
                }
            return result;
        }
        else if (rows == 1 && cols == other.cols) {
            Matrix result(other.rows, cols);
            for (int i = 0; i < other.rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    if (other.data[i][j] == 0)
                        throw std::invalid_argument("Cannot divide by zero");
                    result.data[0][j] = data[i][j] / other.data[i][j];
                }
            return result;
        }
        else if (other.rows == 1 && cols == other.cols) {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    if (other.data[0][j] == 0)
                        throw std::invalid_argument("Cannot divide by zero");
                    result.data[i][j] = data[i][j] + other.data[0][j];
                }
            return result;
        }
        else
            throw std::invalid_argument("Matrix dimensions must match");
    }

    // 重载+=运算符
    Matrix& operator+=(const Matrix& other) {
        *this = *this + other;
        return *this;
    }

    // 重载-=运算符
    Matrix& operator-=(const Matrix& other) {
        *this = *this - other;
        return *this;
    }

    // 重载*=运算符
    Matrix& operator*=(const Matrix& other) {
        *this = *this * other;
        return *this;
    }

    // 重载/=运算符
    Matrix& operator/=(double scalar) {
        *this = *this / scalar;
        return *this;
    }

    // 重载+运算符（与double数）
    const Matrix operator+(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) 
            for (int j = 0; j < cols; ++j) 
                result.data[i][j] = data[i][j] + scalar;
        return result;
    }

    friend const Matrix operator+(double scalar, const Matrix& mat) {
        Matrix result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) 
            for (int j = 0; j < mat.cols; ++j) 
                result.data[i][j] = mat.data[i][j] + scalar;
        return result;
    }

    // 重载-运算符（与double数）
    const Matrix operator-(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) 
            for (int j = 0; j < cols; ++j) 
                result.data[i][j] = data[i][j] - scalar;
        return result;
    }

    friend const Matrix operator-(double scalar, const Matrix& mat) {
        Matrix result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) 
            for (int j = 0; j < mat.cols; ++j) 
                result.data[i][j] = scalar - mat.data[i][j];
        return result;
    }

    // 重载*运算符（与double数）
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) 
            for (int j = 0; j < cols; ++j) 
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    friend const Matrix operator*(double scalar, const Matrix& mat) {
        Matrix result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) 
            for (int j = 0; j < mat.cols; ++j) 
                result.data[i][j] = scalar * mat.data[i][j];
        return result;
    }

    // 重载/运算符（与double数）
    const Matrix operator/(double scalar) const {
        if (scalar == 0)
            throw std::invalid_argument("Cannot divide by zero");
        return *this * (1.0 / scalar);
    }

    friend const Matrix operator/(double scalar, const Matrix& mat) {
        Matrix result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) 
            for (int j = 0; j < mat.cols; ++j) {
                if (mat(i, j) == 0)
                    throw std::invalid_argument("Cannot divide by zero");
                result.data[i][j] = scalar / mat(i, j);
            }
        return result;
    }

    void print() const { // 打印矩阵
        for (const auto& row : data) {
            for (const auto& elem : row) 
                std::cout << elem << "\t";
            std::cout << std::endl;
        }
    }
};

class AdamOptimizer { // Adam优化器
public:
    float beta1, beta2, epsilon;
    int rows, cols;
    Matrix momentum; // 第一矩估计
    Matrix velocity; // 第二矩估计
    int step;

    AdamOptimizer(int rows, int cols, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : rows(rows), cols(cols), beta1(beta1), beta2(beta2), epsilon(epsilon), step(0), momentum(rows, cols, 0.0), velocity(rows, cols, 0.0){}

    void update(Matrix& weights, const Matrix& grad, double learnling_rate) {
        step++;
        momentum = beta1 * momentum + (1 - beta1) * grad;
        velocity = beta2 * velocity + (1 - beta2) * grad.pow(2);
        weights -= learnling_rate * momentum / (1 - std::pow(beta1, step)) / ((velocity / (1 - std::pow(beta2, step))).sqrt() + epsilon);
    }
};

Matrix min_matrix(Matrix a, Matrix b) {  // 返回两个matrix每个位置最小值构成的矩阵
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::invalid_argument("Matrix dimensions must match");
    Matrix res(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            res(i, j) = std::min(a(i, j), b(i, j));
    return res;
}

Matrix max_matrix(Matrix a, Matrix b) {  // 返回两个matrix每个位置最小值构成的矩阵
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::invalid_argument("Matrix dimensions must match");
    Matrix res(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            res(i, j) = std::max(a(i, j), b(i, j));
    return res;
}

Matrix min_matrix(const Matrix& a, double b) {  // 返回两个matrix每个位置最小值构成的矩阵
    Matrix res(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            res(i, j) = std::min(a(i, j), b);
    return res;
}

Matrix max_matrix(const Matrix& a, double b) {  // 返回两个matrix每个位置最小值构成的矩阵
    Matrix res(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            res(i, j) = std::max(a(i, j), b);
    return res;
}

Matrix sigmoid(const Matrix& input) {
    auto res = 1.0 / (1.0 + (0.0 - input).exp());
    return res;
}

Matrix sigmoid_grad(const Matrix& output, const Matrix& grad) {
    return grad * output * (1 - output);
}

Matrix relu(const Matrix& input) {
    return max_matrix(input, 0.0);
}

Matrix relu_grad(const Matrix& input, const Matrix& grad) {
    Matrix result(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            result(i, j) = input(i, j) > 0 ? grad(i, j) : 0;
    return result;
}

Matrix softmax(Matrix& input) {
    Matrix result(input.rows, input.cols);
    double exp_sum;
    for (int i = 0; i < input.rows; ++i) {
        exp_sum = 0;
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = std::exp(input(i, j));
            exp_sum += input(i, j);
        }
        for (int j = 0; j < input.cols; ++j) 
            result(i, j) = input(i, j) / exp_sum;
    }
    return result;
}

Matrix softmax_grad(const Matrix& input) {
    Matrix result(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            for (int k = 0; k < input.cols; ++k) {
                if (k == j)
                    result(i, k) = input(i, j) - input(i, k) * input(i, j);
                else
                    result(i, k) = -input(i, k) * input(i, j);
            }
    return result;
}

double crossEntropyLoss(const Matrix& pred, const Matrix& trueResult) {
    if (pred.rows != trueResult.rows || pred.cols != trueResult.cols)
        throw std::invalid_argument("Matrix dimensions must match");
    return -(trueResult * max_matrix(pred, 1e-15).log()).mean();
}

Matrix crossEntropyLoss_grad(double loss, const Matrix& trueResult) {
    return trueResult / loss;
}

Matrix softmax_CE_gard(const Matrix& input, const Matrix& trueResult) {
    return input - trueResult;
}

double MSE(const Matrix& pred, const Matrix& trueResult) {
    if (pred.rows != trueResult.rows || pred.cols != trueResult.cols)
        throw std::invalid_argument("Matrix dimensions must match");
    return (pred - trueResult).pow(2).mean();
};

Matrix MSE_grad(const Matrix& predict, const Matrix& trueResult) {
    Matrix res(predict.rows, predict.cols);
    return 2 * (predict - trueResult) / (predict.rows * predict.cols);
};

class LinearLayer {  // 线性层
private:
    int inputSize, outputSize;

public:
    Matrix weights, bias;
    AdamOptimizer optim_weight, optim_bias;

    LinearLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize), weights(inputSize, outputSize), optim_weight(inputSize, outputSize), bias(1, outputSize), optim_bias(1, outputSize) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j)  // 随机初始化权重和偏置
                weights(j, i) = std::rand() / static_cast<float>(RAND_MAX);
            bias(0, i) = std::rand() / static_cast<float>(RAND_MAX);
        }
    }

    Matrix forward(const Matrix& input) { // 前向传播
        return input.mat(weights) + bias;
    }

    Matrix backward(const Matrix& input, const Matrix& grad, float lr) { // 后向传播
        auto dw = input.transpose().mat(grad);
        optim_weight.update(weights, dw, lr);

        auto db = grad.sum(0);
        optim_bias.update(bias, db, lr);
        auto dx = grad.mat(weights.transpose());
        return dx;
    }
};

class GaussianDistribution {  // 多维高斯分布类
private:
    int rows, cols; // 参数维度
    std::mt19937 gen; // 随机数生成器
    Matrix mu; // 均值向量
    Matrix sigma; // 标准差向量

public:
    // 构造函数
    GaussianDistribution(const Matrix& mu, const Matrix& sigma)
        : mu(mu), sigma(sigma), gen(std::random_device{}()) {
        if (mu.rows != sigma.rows || mu.cols != sigma.cols)
            throw std::invalid_argument("Mu and sigma must have same shape");
        rows = mu.rows; cols = mu.cols;
    }

    // 生成采样结果
    Matrix sample() {
        Matrix sample(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                sample(i, j) = std::normal_distribution<double>(mu(i, j), sigma(i, j))(gen);
        return sample;
    }

    // 计算给定采样结果的概率对数
    Matrix log_proba(const Matrix& action) {
        return -0.5 * log(2 * M_PI) - sigma.log() - 0.5 * (action - mu).pow(2) / sigma.pow(2);
    }

    /*Matrix log_proba_grad(const Matrix& grad) {
        return (mu - grad) / sigma.pow(2);
    }*/

    std::pair<Matrix, Matrix> log_proba_grad(const Matrix& grad, const Matrix& action) {
        auto mu_grad = (-1 / mu) + (action - sigma).pow(2) / mu.pow(3) * grad;
        auto sigma_grad = (action - sigma) / mu.pow(2) * grad;
        return std::make_pair(mu_grad, sigma_grad);
    }
};
