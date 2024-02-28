#include <iostream>


class Matrix {
private:
    int rows;
    int cols;
    int** matrix;

public:
    Matrix(int rows, int cols) : rows(rows), cols(cols) { //构造函数，接受行列
        matrix = new int*[rows];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new int[cols];
        }
    }

    Matrix(int rows, int cols, int** matrix_in) : rows(rows), cols(cols) { // 构造函数，接受行列和二维数组

        matrix = new int*[rows];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new int[cols];
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = matrix_in[i][j];
            }
        }
    }

    template <size_t rows, size_t cols> 
    Matrix(int (&matrix)[rows][cols]) : rows(rows), cols(cols) { // 构造函数模板，接受已知大小的二维数组引用
        this->matrix = new int*[rows];
        for (size_t i = 0; i < rows; i++) {
            this->matrix[i] = new int[cols];

            for (size_t j = 0; j < cols; j++) {
                this->matrix[i][j] = matrix[i][j];
            }
        }
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) { // 构造函数，接受矩阵
        matrix = new int*[rows];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new int[cols];
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = other.matrix[i][j];
            }
        }
    }


    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), matrix(nullptr) { // 构造函数，移动矩阵
        matrix = other.matrix;
        other.matrix = nullptr;
    }



    ~Matrix() { //析构函数
        for (int i = 0; i < rows; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }


    friend std::istream& operator>>(std::istream& in, Matrix& matrix) { //重载提取运算符 >>(输入)
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                in >> matrix.matrix[i][j];
            }
        }
        return in;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) { //重载提取运算符 >>(输出)
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                os << matrix.matrix[i][j] << " ";
            }
            os << std::endl;
        }
        return os;
    }


    Matrix operator*(const Matrix& other) const { //重载*运算符(乘法)
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
        }

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.matrix[i][j] = 0;
                for (int k = 0; k < cols; ++k) {
                    result.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return result;
    }

    Matrix& operator=(const Matrix& other) { //重载=运算符(赋值)
        if (this != &other) {
            int** newMatrix = new int*[other.rows];
            for (int i = 0; i < other.rows; ++i) {
                newMatrix[i] = new int[other.cols];
                for (int j = 0; j < other.cols; ++j) {
                    newMatrix[i][j] = other.matrix[i][j];
                }
            }
            
            for (int i = 0; i < rows; ++i) {
                delete[] matrix[i];
            }
            delete[] matrix;

            rows = other.rows;
            cols = other.cols;

            matrix = newMatrix;
        }
        return *this;
    }


Matrix& operator+(const Matrix& other){ //重载+运算符(加法)
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for addition.");
    }

    int** result = new int*[rows];
    for (int i = 0; i < rows; i++) {
        result[i] = new int[cols];
    }

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result[i][j] = matrix[i][j] + other.matrix[i][j];
        }
    }

    Matrix* result_m = new Matrix(rows, cols, result);
    return *result_m;
}


    Matrix& operator-(const Matrix& other){ //重载-运算符(减法)
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for subtraction.");
    }

    int** result = new int*[rows];
    for (int i = 0; i < rows; i++) {
        result[i] = new int[cols];
    }

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result[i][j] = matrix[i][j] - other.matrix[i][j];
        }
    }

    Matrix* result_m = new Matrix(rows, cols, result);
    return *result_m;
}

    Matrix operator/(const Matrix& other) { //重载/运算符(除法)
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions are not compatible for division.");
        }

        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                int sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += matrix[i][k] * other.matrix[k][j];
                }
                result.matrix[i][j] = sum;
            }
        }
        return result;
    }

    bool operator==(const Matrix& other) { //重载==运算符(比较)
        if (rows != other.rows || cols != other.cols) {
            return false;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] != other.matrix[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
};