/**
 * 完善 Matrix 类，使得下面的代码可以正常运行:
 * class Matrix {
 * public:
 * TODO: Someting
 * private:
 *     int rows;
 *     int cols;
 *     int **matrix;
 * };
 *
 * 评分标准：
 * 1. 代码能够正常的运行就有 100 分，若代码不能正常运行，则按以下标准进行评分：
 * Matrix::Matrix(int rows, int cols):                              20 points
 * Matrix::~Matrix():                                               20 points
 * std::istream& operator>>(std::istream& in, Matrix& matrix)       20 points
 * std::ostream& operator<<(std::ostream& os, const Matrix& matrix) 20 points
 * Matrix Matrix::operator*(Matrix &other)                          20 points
 * 2. 在正常运行的基础上，若还有其他的功能，则按以下标准进行评分：
 * Matrix::Matrix(int rows, int cols, int** matrix)                 10 points
 * template <size_t rows, size_t cols> Matrix::Matrix(int (&matrix)[rows][cols])        10 points
 * Matrix::Matrix(const Matrix& other)                              10 points
 * Matrix::Matrix(Matrix&& other)                                   10 points
 * Matrix& Matrix::operator=(const Matrix& other)                   10 points
 * Matrix& Matrix::operator=(Matrix &&other)                        10 points
 * Matrix Matrix::operator+(const Matrix& other)                    10 points
 * Matrix Matrix::operator-(const Matrix& other)                    10 points
 * Matrix Matrix::operator/(const Matrix& other)                    10 points
 * bool Matrix::operator==(const Matrix& other)                     10 points
 */
#include<iostream>
#include "matrix.hpp"

int main() {

    int rows, cols;
    std::cout << "input rows and cols of A\n";
    std::cin >> rows >> cols;
    Matrix A(rows, cols);
    std::cout << "input values of A\n";
    std::cin >> A;

    std::cout << "input rows and cols of B\n";
    std::cin >> rows >> cols;
    Matrix B(rows, cols);
    std::cout << "input values of B\n";
    std::cin >> B;
    std::cout << "\nA*B\n";
    std::cout << A * B;


    int row_t = 2, col_t = 2; //测试
    int** arr_t = new int*[2];
    for (int i = 0; i < 2; ++i) {
        arr_t[i] = new int[2];
        for (int j = 0; j < 2; ++j) {
            arr_t[i][j] =1 + i * 2 + j;
        }
    }

    Matrix C(row_t, col_t, arr_t);
    std::cout << "\nC\n" << C;
/*输出
0 1
2 3
*/


    int arr_t_2[2][2] = {{1, 2}, {3, 4}};
    Matrix t(arr_t_2);
    std::cout << "\nt\n" << t;
/*输出
1 2
3 4
*/

    Matrix A4(A);
    std::cout <<"\nA4\n" << A4;
/*输出
1 2
3 4
*/

    Matrix temp = A;
    Matrix A5(std::move(temp));
    std::cout << "\nA5\n" << A5;
/*输出
1 2
3 4
*/
    Matrix A6 = A;
    std::cout << "\nA6\n" << A6;

    Matrix A7 = A + B;
    std::cout << "\nA+B\n" << A7;

    Matrix A8 = A - B;
    std::cout << "\nA-B\n" << A8;

    Matrix A9 = A / B;
    std::cout << "\nA/B\n" << A9;

    if(A == B){
        std::cout << "\nA==B";
    }
    else{
        std::cout << "\nA!=B";
    }
    return 0;
}