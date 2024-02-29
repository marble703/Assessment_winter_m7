/**
 * @file question_second.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

 /*
 创建一个 C++ 程序，定义一个带有虚函数 calculateArea 的基类 Shape。
 从基类派生出两个类 Circle 和 Rectangle。在每个派生类中实现 calculateArea 函数，
 分别计算圆形和矩形的面积。在主函数中，创建两个派生类的实例，
 分别调用 calculateArea 函数，并显示结果。
 */

#include <iostream>

class Shape {
public:
    virtual double calculateArea() const = 0;
};

class Circle : public Shape {
private:
    double r;

public:
    Circle(double rad) : r(rad) {}

    double calculateArea() const override {
        return 3.14159 * r * r;
    }
};

class Rectangle : public Shape {
private:
    double w;
    double h;

public:
    Rectangle(double weight, double height) : w(weight), h(height) {}

    double calculateArea() const override {
        return w * h;
    }
};

int main() {
    Circle circle(5);
    Rectangle rectangle(4, 6);

    std::cout << "Circle area: " << circle.calculateArea() << std::endl;
    std::cout << "Rectangle area: " << rectangle.calculateArea() << std::endl;

    return 0;
}
