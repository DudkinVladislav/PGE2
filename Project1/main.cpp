#include <iostream>
#include <fstream>
#include <conio.h>
#include <map>
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "math.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc\imgproc.hpp>
#include <Windows.h>
#include <GL/glew.h>
#include "optimization.h"
#include <GL/freeglut.h>
#include "Header.h"
#include <chrono>
#ifdef __APPLE__
#include <CL/opencl.h>
#else
#include <CL/cl.h>
#endif 
using namespace std;
#define MAX_SOURCE_SIZE (0x100000)
#include <omp.h>
#include <thread>
#include <mutex>
using namespace std;
using namespace alglib;
using namespace std::chrono;
struct expr;
vector<double> solve_lm(expr f, int nomer_thread, double* consts, int consts_size);
void funclm1(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm2(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm3(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm4(const real_1d_array& x, real_1d_array& fi, void* ptr); 
void funclm5(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm6(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm7(const real_1d_array& x, real_1d_array& fi, void* ptr);
void funclm8(const real_1d_array& x, real_1d_array& fi, void* ptr);
//void funclmjac(const real_1d_array& x, real_1d_array& fi, real_2d_array& jac, void* ptr);

int ligh(int x, int y)
{
    return (x + y - 1) / y * y;
}
double* x;
double* y;
double** z1;
double** z2;
double** z3;
mutex mtx;
unsigned char bitextract(const unsigned int byte, const unsigned int mask) {
    if (mask == 0) {
        return 0;
    }


    // определение количества нулевых бит справа от маски
    int
        maskBufer = mask,
        maskPadding = 0;

    while (!(maskBufer & 1)) {
        maskBufer >>= 1;
        maskPadding++;
    }

    // применение маски и смещение
    return (byte & mask) >> maskPadding;
}
struct pixel //структура одного пикселя
{
    double r;
    double g;
    double b;
    
};

void changeViewPort1(int w, int h)
{
    glViewport(0, 0, w, h);
}
void changeViewPort2(int w, int h)
{
    glViewport(0, 0, w, h);
}
void changeViewPort3(int w, int h)
{
    glViewport(0, 0, w, h);
}

vector < expr > expressions1;
vector < expr > expressions2;
vector < expr > expressions3;
cl_platform_id platform_id;
cl_uint ret_num_platforms;
cl_device_id device_id;
cl_context context;
cl_uint ret_num_devices;
cl_command_queue command_queue1, command_queue2, command_queue3, command_queue4, command_queue5, command_queue6, command_queue7, command_queue8;
cl_program program = NULL;
cl_int ret;
int size;

int population = 10;
double* bx;
double* by;
pixel** orig;
pixel** colors;
int n;
int m;
double tochn=1;
int fitconstssize;
double* fitx;
double* fity;
double** fitz;



int fllm = 0;
struct Node {
    int type = 0;
    int uniq = 0;
    map<int, Node*> children;

    bool insert(vector<int> expr) {
        bool ins = false;
        Node* newNode = children[expr[0]];

        // does this branch exist?
        if (newNode == nullptr) {
            newNode = new Node;
            newNode->type = expr[0];
            newNode->children = map<int, Node*>();
            children[expr[0]] = newNode;
            ins = true;
        }

        // recursive call to insert
        if (expr.size() > 1) {
            ins = newNode->insert(vector<int>(expr.begin() + 1, expr.end())) || ins;
        }

        // visitation accounting
        // (-1 is the root of the memoization tree)
        newNode->uniq++;
        if (type == -1) {
            uniq++;
        }
        return ins;
    }
};

double znach[5][5] = { 0 };
map <int, string> op = {  //словарь типов операций
    {1, "+"},
    {2, "*"},
    {3, "-"},
    {4, "/"},
    {5, "COS"},
    {6, "SIN"},
    {7, "TAN"},
    {8, "LOG"},
    {9, "EXP^"},
    {10, "SQRT"},
    {11, "^"},
    {12, "CONST"},
    {13, "X"},
    {14, "Y"},
    {15, "Z"}
};

expr createconst(int unic);

struct expr 
{ //структура выражения
    int var = 0;
    int zzz = 0;
    string name;
    int type;
    int kolvoparam;
    vector<expr>params;
    int uniqconst;
    double tochnost=INFINITE;
    int dlin;
    vector<double> consts;
    int dlina() //функция нахождения длины выражения
    {
        int dlin = 1;
        for (int i = 0; i < kolvoparam; i++)
        {
            dlin+=params[i].dlina();
        }
        return dlin;
    }
    void qsortRecursive(vector<expr> &mas, int size) {
        //Указатели в начало и в конец массива
        int i = 0;
        int j = size - 1;

        //Центральный элемент массива
        expr mid = mas[size / 2];

        //Делим массив
        do {
            //Пробегаем элементы, ищем те, которые нужно перекинуть в другую часть
            //В левой части массива пропускаем(оставляем на месте) элементы, которые меньше центрального
            while (mas[i].type < mid.type) {
                i++;
            }
            //В правой части пропускаем элементы, которые больше центрального
            while (mas[j].type > mid.type) {
                j--;
            }

            //Меняем элементы местами
            if (i <= j) {
                expr tmp1 = mas[i];
                expr tmp2 = mas[j];
                mas[i] = tmp2;
                mas[j] = tmp1;

                i++;
                j--;
            }
        } while (i <= j);


        //Рекурсивные вызовы, если осталось, что сортировать
        if (j > 0) {
            //"Левый кусок"
            qsortRecursive(mas, j + 1);
        }
        if (i < size) {
            //"Првый кусок"
            vector<expr> mas1;
            for(int a=i;a<size;a++)
            {
                mas1.push_back(mas[a]);
            }
            qsortRecursive(mas1, size - i);
            for (int b = i; b < size; b++)
            {
                expr temp;
                temp = mas1[b - i];
                mas[b] = temp;
            }
        }
    }

    int rasstanovka_nomerov_consts(int nom)
    {
        int nomer = nom;
        
        for (int i = 0; i < params.size(); i++)
        {
            if (params[i].type == 12)
            {
                params[i].name= "CONST" + to_string(nomer+1);
                nomer++;
            }
            else {
                if ((params[i].type != 13) && (params[i].type != 14) && (params[i].type != 15))
                {
                    nomer = params[i].rasstanovka_nomerov_consts(nomer);

                }
            }
        }
        return nomer;
    }

    void makeint(vector<int>& num) //функция превращения 
    {
        if ((type == 1)||(type == 2))
        {
            qsortRecursive(params, params.size());
        }
        if(kolvoparam==0)
        {
            num.push_back(type);
            return;
        }
        num.push_back(type);
        num.push_back(0);
        for (int i = 0; i < kolvoparam; i++)
        {
            params[i].makeint(num);
        }
        num.push_back(0);
    }
    void print() //функция распечатки выражения
    {
        kolvoparam = params.size();
        if (kolvoparam == 0)
        {
            cout << name;
            return;
        }
        
        cout << "(";
        cout << name;
        for (int i = 0; i < kolvoparam; i++)
        {
            if (i == 0)
                cout << " ";
            params[i].print();
            if (i != kolvoparam - 1)
                cout << " ";
        }
        cout << ")";
    }

    double tochn(double* x, double* y, double** z, double* consts, int consts_size, cl_command_queue command_queue)
    {
        expr fff = *this;
        double sum=0;
        //double max_tochnost = -1;
        double* znach=new double[n*m]; 
        (*this).res(fff, x, y, z, consts, znach, command_queue);
        for (int i = 0; i < n*m; i++)
        {
            double tochnost_v_tochke = pow(znach[i], 2);
            sum += tochnost_v_tochke;


            //if (tochnost_v_tochke > max_tochnost)
            //{
            //    max_tochnost = tochnost_v_tochke;
            //}
        }
        sum /= (n * m);
        
        delete[] znach;
        return sqrt(sum);
    }
    void res(expr ex, double* x, double* y, double** zz, double* consts, double *result, cl_command_queue command_queue)
    {

        if (ex.type == 15)
        {
            for(int i=0;i<n;i++)
            {  
                for (int j = 0; j < m; j++)
                {
                    result[i * m+j]=zz[i][j];
                }
            
            }
        }
        if (ex.type == 13)
        {
            for (int i = 0; i < n; i++)
            {
            for (int j = 0; j < m; j++)
            {
                result[i * m + j] = x[i];
            }
            }
        }
        if (ex.type == 14)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    result[i * m + j] = y[j];
                }
            }
        }
        if (ex.type == 12)
        {
            if (ex.name.find("CONST") != std::string::npos)
            {
                string names = ex.name.substr(5, (*ex.name.end() - 1) - 5);
                int n1 = stoi(names);

                for (int i = 0; i < n * m; i++)
                {
                    result[i] = consts[n1 - 1];
                }
            }
            else
            {
                double c = { (double)stoi(name) };
                for (int i = 0; i < n * m; i++)
                {
                    result[i] = c;
                }
            }
        }
        if (ex.type == 1)
        {
            if ((ex.params[0].type == 15) && (ex.zzz == 1)&&(ex.params[1].type!=3))
            {
                cl_kernel kernel;
                kernel = clCreateKernel(program, "sumz", &ret);
                cl_mem inputBuff = NULL;
                cl_mem outputBuff = NULL;
                cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m * params.size());
                cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
                for (int i = 0; i < ex.params.size(); i++)
                {
                    cl_double* temp = (cl_double*)malloc(sizeof(cl_double) * n*m);
                    params[i].res(params[i],x, y, zz, consts, temp, command_queue);
                    memcpy(input + i * n*m, temp, n*m * sizeof(cl_double));
                    free(temp);
                }
                int size1 = ex.params.size();
                inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double) * ex.params.size(), input, &ret);
                outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
                ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double) * ex.params.size(), input, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), output, 0, NULL, NULL);
                ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
                ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
                ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
                ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
                ret = clSetKernelArg(kernel, 4, sizeof(int), &size1);
                size_t local_work_size[2] = { 128, 1 };
                size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
                ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
                ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
                clReleaseKernel(kernel);
                clReleaseMemObject(inputBuff);
                clReleaseMemObject(outputBuff);
                free(input);
                free(output);
                
            }
            else
            {
                cl_kernel kernel;
                kernel = clCreateKernel(program, "summ", &ret);
                cl_mem inputBuff = NULL;
                cl_mem outputBuff = NULL;
                int size1 = ex.params.size();
                cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m * size1);
                cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
                for (int i = 0; i < size1; i++)
                {
                    cl_double* temp = (cl_double*)malloc(sizeof(cl_double) * n*m);
                    ex.params[i].res(params[i], x, y, zz, consts, temp, command_queue);
                    memcpy(&input[ i * n*m], temp, n*m * sizeof(cl_double));
                    free(temp);
                }
                inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double) * size1, input, &ret);
                outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), output, &ret);
                ret = clEnqueueWriteBuffer(command_queue, inputBuff, 1, 0, n*m * sizeof(cl_double) * size1, input, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, outputBuff, 1, 0, n*m * sizeof(cl_double), output, 0, NULL, NULL);
                ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuff);
                ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuff);
                ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
                ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
                ret = clSetKernelArg(kernel, 4, sizeof(int), &size1);
                size_t local_work_size[2] = { 128, 1 };
                size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
                ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
                ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
                clReleaseKernel(kernel);
                clReleaseMemObject(inputBuff);
                clReleaseMemObject(outputBuff);
                free(input);
                free(output);
            }
        }
        if (ex.type == 2)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "prr", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            int size1 = params.size();
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m * size1);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            for (int i = 0; i < size1; i++)
            {
                cl_double* temp = (cl_double*)malloc(sizeof(cl_double) * n*m);
                params[i].res(params[i], x, y, zz, consts, temp, command_queue);
                memcpy(input + i * n*m, temp, n*m * sizeof(cl_double));
                free(temp);
            }
            outputBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, n*m * sizeof(cl_double), output, &ret);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, n*m * sizeof(cl_double) * size1, input, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, 1, 0, n*m * size1 * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, outputBuff, 1, 0, n*m * sizeof(cl_double), output, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(cl_int), &m);
            ret = clSetKernelArg(kernel, 4, sizeof(cl_int), &size1);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, NULL, NULL);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 3)
        {
            if (ex.zzz == 1)
            {
                cl_kernel kernel;
                kernel = clCreateKernel(program, "minz", &ret);
                cl_mem inputBuff = NULL;
                cl_mem outputBuff = NULL;
                cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
                cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
                params[0].res(params[0],x, y, zz, consts, input, command_queue);
                inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
                outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), output, &ret);
                ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), output, 0, NULL, NULL);
                ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
                ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
                ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
                ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
                size_t local_work_size[2] = { 128, 1 };
                size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
                ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
                ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
                clReleaseKernel(kernel);
                clReleaseMemObject(inputBuff);
                clReleaseMemObject(outputBuff);
                free(input);
                free(output);
            }
            else {
                cl_kernel kernel;
                kernel = clCreateKernel(program, "minn", &ret);
                cl_mem inputBuff = NULL;
                cl_mem outputBuff = NULL;
                cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
                cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n * m);
                params[0].res(params[0], x, y, zz, consts, input, command_queue);
                inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n * m * sizeof(cl_double), input, &ret);
                outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * m * sizeof(cl_double), output, &ret);
                ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n * m * sizeof(cl_double), input, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, outputBuff, CL_TRUE, 0, n * m * sizeof(cl_double), output, 0, NULL, NULL);
                ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
                ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
                ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
                ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
                size_t local_work_size[2] = { 128, 1 };
                size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
                ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
                ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n * m * sizeof(double), result, 0, 0, 0);
                clReleaseKernel(kernel);
                clReleaseMemObject(inputBuff);
                clReleaseMemObject(outputBuff);
                free(input);
                free(output);
            } 
        }
        if (ex.type == 4)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "divv", &ret);
            cl_mem inputBuff1 = NULL;
            cl_mem inputBuff2 = NULL;
            cl_mem outputBuff = NULL;
            
            cl_double* input1 = (cl_double*)malloc(sizeof(cl_double) * n * m);
            cl_double* input2 = (cl_double*)malloc(sizeof(cl_double) * n * m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n * m);
            params[0].res(params[0],x, y, zz, consts, input1, command_queue);
            params[1].res(params[1],x, y, zz, consts, input2, command_queue);
            inputBuff1 = clCreateBuffer(context, CL_MEM_READ_ONLY, n * m * sizeof(cl_double), input1, &ret);
            inputBuff2 = clCreateBuffer(context, CL_MEM_READ_ONLY, n * m * sizeof(cl_double), input2, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * m * sizeof(cl_double), output, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff1, 1, 0, n * m * sizeof(cl_double), input1, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff2, 1, 0, n * m * sizeof(cl_double), input2, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, outputBuff, 1, 0, n * m * sizeof(cl_double), output, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff1);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&inputBuff2);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 4, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n * m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff1);
            clReleaseMemObject(inputBuff2);
            clReleaseMemObject(outputBuff);
            free(input1);
            free(input2);
            free(output);
        }
        if (ex.type == 5)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "coss", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), output, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 6)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "sinn", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 7)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "tann", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 8)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "logg", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 9)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "expp", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0],x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 10)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "sqrtt", &ret);
            cl_mem inputBuff = NULL;
            cl_mem outputBuff = NULL;
            cl_double* input = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input, command_queue);
            inputBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff, CL_TRUE, 0, n*m * sizeof(cl_double), input, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff);
            ret = clSetKernelArg(kernel, 2, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff);
            clReleaseMemObject(outputBuff);
            free(input);
            free(output);
        }
        if (ex.type == 11)
        {
            cl_kernel kernel;
            kernel = clCreateKernel(program, "poww", &ret);
            cl_mem inputBuff1 = NULL;
            cl_mem inputBuff2 = NULL;
            cl_mem outputBuff = NULL;
           
            cl_double* input1 = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* input2 = (cl_double*)malloc(sizeof(cl_double) * n*m);
            cl_double* output = (cl_double*)malloc(sizeof(cl_double) * n*m);
            params[0].res(params[0], x, y, zz, consts, input1, command_queue);
            params[1].res(params[1], x, y, zz, consts, input2, command_queue);
            inputBuff1 = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input1, &ret);
            inputBuff2 = clCreateBuffer(context, CL_MEM_READ_ONLY, n*m * sizeof(cl_double), input2, &ret);
            outputBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m * sizeof(cl_double), NULL, &ret);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff1, CL_TRUE, 0, n*m * sizeof(cl_double), input1, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, inputBuff2, CL_TRUE, 0, n*m * sizeof(cl_double), input2, 0, NULL, NULL);
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&outputBuff);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inputBuff1);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&inputBuff2);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &n);
            ret = clSetKernelArg(kernel, 4, sizeof(int), &m);
            size_t local_work_size[2] = { 128, 1 };
            size_t global_work_size[2] = { ligh(n, local_work_size[0]), ligh(m, local_work_size[1]) };
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            ret = clEnqueueReadBuffer(command_queue, outputBuff, CL_TRUE, 0, n*m * sizeof(double), result, 0, 0, 0);
            clReleaseKernel(kernel);
            clReleaseMemObject(inputBuff1);
            clReleaseMemObject(inputBuff2);
            clReleaseMemObject(outputBuff);
            free(input1);
            free(input2);
            free(output);
        }
    }
    double res_in_point(double x, double y, double z, vector<double> consts)
    {
        //функция нахождения результата выражения в конкретной точке
        if (type == 15)
        {
            return z;
        }
        if(type==13)
        {
            return x;
        }
        if(type==14)
        {
            return y;
        }
        if (type == 12)
        {
            if (name.find("CONST") != std::string::npos)
            {
                string names = name.substr(5, (*name.end() - 1) - 5);
                int n1 = stoi(names);
                return consts[n1-1];
            }
            else
            {
                int n1 = stoi(name);
                return n1;
            }
        }
        if (type == 1)
        {
            double sum = 0;
            
            if ((params[0].type == 15)&&(zzz==1))
            {
                double znach = params[1].res_in_point(x, y, z, consts);
                if (znach > 255)
                {
                    znach = 255;
                }
                if (znach < 0)
                {
                    znach = 0;
                }
                sum += znach;
                sum += params[0].res_in_point(x, y, z, consts);
                return sum;
            }
            for (int i = 0; i < kolvoparam; i++)
            {

                sum += params[i].res_in_point(x, y, z, consts);
            }
            
            return sum;
        }
        if (type == 2)
        {
            double pr=1;
            for (int i = 0; i < kolvoparam; i++)
            {
                pr *= params[i].res_in_point(x, y, z, consts);
            }
            return pr;
        }
        if (type == 3)
        {
            return -(params[0].res_in_point(x, y, z, consts));
        }
        if (type == 4)
        {
            return (params[0].res_in_point(x, y, z, consts) / params[1].res_in_point(x, y, z, consts));
        }
        if (type == 5)
        {
            return cos(params[0].res_in_point(x, y, z, consts));
        }
        if (type == 6)
        {
            return sin(params[0].res_in_point(x, y, z, consts));
        }
        if (type == 7)
        {
            return tan(params[0].res_in_point(x, y, z, consts));
        }
        if (type == 8)
        {
            return log(abs(params[0].res_in_point(x, y, z, consts)));
        }
        if (type == 9)
        {
            return exp(params[0].res_in_point(x, y, z, consts));
        }
        if (type == 10)
        {
            return sqrt(abs(params[0].res_in_point(x, y, z, consts)));
        }
        if (type == 11)
        {
            return pow(params[0].res_in_point(x, y, z, consts), params[1].res_in_point(x, y, z, consts));
        }
    }
   vector< double> fit(int thread_nomer,cl_command_queue command_queue, double **zz)
    { //функция поиска лучших констант
       vector<int> v;
       uniqconst = (*this).rasstanovka_nomerov_consts(0);
       v.clear();
       v.resize(0);
        expr f;
        f.type = 1;
        f.name = op[f.type];
        f.kolvoparam = 2;
        expr z;
        z.type = 15;
        z.name = op[z.type];
        z.kolvoparam = 0;
        z.uniqconst = 0;
        f.uniqconst = 0;
        f.params.push_back(z);
        f.zzz = 1;
        expr nn;
        if (type == 3)
        {
            nn.zzz = 1;
            nn.type = params[0].type;
            nn.name = params[0].name;
            nn.kolvoparam = params[0].kolvoparam;
            nn.params = params[0].params;
            nn.uniqconst = params[0].uniqconst;
        }
        else
        {
            nn.zzz = 1;
            nn.type = 3;
            nn.name = op[nn.type];
            nn.kolvoparam = 1;
            nn.uniqconst = uniqconst;
            expr ol;
            ol.type = type;
            ol.name = name;
            ol.kolvoparam = kolvoparam;
            ol.params = params;
            ol.uniqconst = uniqconst;
            nn.params.push_back(ol) ;
        }
        f.uniqconst += nn.uniqconst;
        f.params.push_back(nn);
        //vector<expr> grad;
        //vector <vector<expr>> gessian;
        double* consts=new double[uniqconst];
        for (int i = 0; i < uniqconst; i++)
        {
            consts[i]=1;
        }
        vector<double> new_consts;
        double* ccc;
        new_consts=solve_lm( f,thread_nomer, consts, uniqconst);
        ccc = new double[new_consts.size()];
        for (int i = 0; i < new_consts.size();i++)
        {
            ccc[i] = new_consts[i];
        }

        
        (*this).tochnost = f.tochn(x, y, zz, ccc, new_consts.size(), command_queue);
        (*this).consts = new_consts;
        delete[] ccc;
        delete[] consts;
        f.zzz = 0;
        f.params[1].zzz = 0;
        if (isnan((*this).tochnost))
        {
            (*this).tochnost = 10000000;
        }
        if (tochnost == INFINITE)
        {
            (*this).tochnost = 10000000;
        }
        return new_consts;
    }
    expr dif(int nomerconst, int time)
    {//функция дифференцирования
        expr d;
        if(time==0)
        { 
        d.type = 2;
        d.name = op[d.type];
        d.kolvoparam = 2;
        d.uniqconst = uniqconst;
        d.params.push_back(*this);
        expr two;
        two.kolvoparam = 0;
        two.type = 12;
        two.name = "2";
        d.params.push_back(two);
        }
        expr proizvodnay;
        string nomer = "CONST" + to_string(nomerconst);
        if (type==15 || type==14 || type==13)
        {
            expr zero;
            zero.type = 12;
            zero.name = "0";
            zero.kolvoparam = 0;
            zero.uniqconst = 0;
            if(time==0)
            {
                d.params.push_back(zero);
                return d;
            }
            return zero;
        }
        if (type == 12)
        {
            if (name == nomer)
            {
                expr one;
                one.type = 12;
                one.name = "1";
                one.kolvoparam = 0;
                one.uniqconst = 0;
                if (time == 0)
                {
                    d.params.push_back(one);
                    return d;
                }
                return one;
            }
            else
            {
                expr zero;
                zero.type = 12;
                zero.name = "0";
                zero.kolvoparam = 0;
                zero.uniqconst = 0;
                if (time == 0)
                {
                    d.params.push_back(zero);
                    return d;
                }
                return zero;
            }
        }
        if (type == 1)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }
            if (glag == 1)
            {
                proizvodnay.type = 12;
                proizvodnay.name = "1";
                proizvodnay.kolvoparam = 0;
                proizvodnay.uniqconst = 0;
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
            else
            {
                proizvodnay.type = type;
                proizvodnay.name = op[proizvodnay.type];
                int flag=0;
                int nom;
                for (int i = 0; i < params.size(); i++)
                {
                    if (params[i].dif(nomerconst, 1).name != "0")
                    {
                        proizvodnay.params.push_back(params[i].dif(nomerconst, 1));
                    }
                }
                proizvodnay.kolvoparam = proizvodnay.params.size();
                if (proizvodnay.kolvoparam == 0)
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam=0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
                if (proizvodnay.kolvoparam == 1)
                {
                    expr newe;
                    newe.type = proizvodnay.params[0].type;
                    newe.name= proizvodnay.params[0].name;
                    newe.kolvoparam= proizvodnay.params[0].kolvoparam;
                    newe.params = proizvodnay.params[0].params;
                    
                    if (time == 0)
                    {

                        
                        d.params.push_back(newe);
                        d.kolvoparam = params.size();
                        
                        return d;
                    }
                    return newe;
                }
                if (time == 0)
                {
                    
                    d.params.push_back(proizvodnay);
                    d.kolvoparam = params.size();
                    return d;
                }
                return proizvodnay;
            }
        }
        if (type == 2)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }
            if (glag == 1)
            {
                if (kolvoparam > 2)
                {
                    proizvodnay.type = 2;
                    proizvodnay.name = op[proizvodnay.type];
                    for (int i = 0; i < params.size(); i++)
                        if (params[i].name != nomer)
                        {
                            proizvodnay.params.push_back(params[i]);
                        }
                    proizvodnay.kolvoparam = proizvodnay.params.size();
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    if (params[0].name == nomer)
                    {
                        proizvodnay=params[1];
                    }
                    else
                    {
                        proizvodnay = params[0];
                    }
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
            }
            else
            {
                int flag=0;
                int nom=0;
                int nom_f;
                proizvodnay.type = type;
                proizvodnay.name = op[proizvodnay.type];
                vector <expr> proizv_chasti;
                vector <expr> chast_expr;
                vector <expr> zero_proizv;
                for (int i = 0; i < params.size(); i++)
                {    
                    if (params[i].dif(nomerconst, 1).name!="0")
                    {
                       
                        proizv_chasti.push_back(params[i].dif(nomerconst, 1));
                        chast_expr.push_back(params[i]);
                        flag = 1;
                        nom ++;
                        nom_f = i;
                    }
                    else
                    {
                        zero_proizv.push_back(params[i]);
                    }
                }
                if (flag == 1)
                {
                   
                    if (nom == 1)
                    {
                        for (int i = 0; i < params.size(); i++)
                        {
                            if (i != nom_f)
                                proizvodnay.params.push_back(params[i]);
                            
                        }
                        proizvodnay.params.push_back(proizv_chasti[0]);
                        
                    }
                    else
                    {
                        expr proizvod;
                        proizvod.type = 1;
                        proizvod.name = op[proizvod.type];
                        for (int i = 0; i < chast_expr.size(); i++)
                        {
                            expr proizv;
                            proizv.type = 2;
                            proizv.name = op[proizv.type];
                            proizv.params.push_back(proizv_chasti[i]);
                            for (int j = 0; j < chast_expr.size(); j++)
                            {
                                if (i != j)
                                {
                                    proizv.params.push_back(chast_expr[j]);
                                }
                            }
                            proizv.kolvoparam = proizv.params.size();
                            proizvod.params.push_back(proizv);
                        }
                        proizvod.kolvoparam = proizvod.params.size();
                        if (zero_proizv.size() > 0)
                        {
                            proizvodnay.type = 2;
                            proizvodnay.name = op[proizvodnay.type];
                            for (int i = 0; i < zero_proizv.size(); i++)
                            {
                                proizvodnay.params.push_back(zero_proizv[i]);
                            }
                            proizvodnay.params.push_back(proizvod);
                            proizvodnay.kolvoparam = proizvodnay.params.size();
                        }
                        else
                        {
                            if (time == 0)
                            {
                                d.params.push_back(proizvod);
                                return d;
                            }
                            return proizvod;
                        }
                        
                    }
                }
                proizvodnay.kolvoparam = proizvodnay.params.size();
                if (proizvodnay.kolvoparam == 0)
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
                if (proizvodnay.kolvoparam == 1)
                {
                    expr newe;
                    newe.type = proizvodnay.params[0].type;
                    newe.name = proizvodnay.params[0].name;
                    newe.kolvoparam = proizvodnay.params[0].kolvoparam;
                    newe.params = proizvodnay.params[0].params;
                    if (time == 0)
                    {
                        d.params.push_back(newe);
                        return d;
                    }
                    return newe;
                }
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
        }
        if (type == 3)
        {
            
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }
            if (glag == 1)
            {
                proizvodnay.type = 12;
                proizvodnay.name = "-1";
                proizvodnay.kolvoparam = 0;
                proizvodnay.uniqconst = 0;
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
            else
            {
                proizvodnay.type = type;
                proizvodnay.name = op[proizvodnay.type];
                    proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    if (proizvodnay.params[0].name == "0")
                    {
                        proizvodnay.params.pop_back();
                    }
                proizvodnay.kolvoparam = proizvodnay.params.size();
                proizvodnay.uniqconst = uniqconst--;
                if (proizvodnay.kolvoparam == 0)
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
                if (proizvodnay.params[0].type == 3)
                {
                    proizvodnay.type = proizvodnay.params[0].params[0].type;
                    proizvodnay.name = op[proizvodnay.type];
                    proizvodnay.kolvoparam= proizvodnay.params[0].params[0].kolvoparam;
                    for (int i = 0; i < proizvodnay.kolvoparam; i++)
                    {
                        proizvodnay.params.push_back(proizvodnay.params[0].params[0].params[i]);
                    }
                    proizvodnay.params.erase(proizvodnay.params.begin());
                }
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
        }
        if (type == 4)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }
            if (glag == 1)
            {
                proizvodnay.type = type;
                proizvodnay.name = op[proizvodnay.type];
                proizvodnay.kolvoparam = 2;
                proizvodnay.uniqconst = 0;
                expr one;
                one.type = 12;
                one.name = "1";
                one.kolvoparam = 0;
                one.uniqconst = 0;
                proizvodnay.params.push_back(one);
                proizvodnay.params.push_back( params[1]);
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                    return proizvodnay;  
            }
            else
            {
                if (params[0].dif(nomerconst, 1).name == "0" and params[1].dif(nomerconst, 1).name == "0")
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
                proizvodnay.uniqconst = uniqconst--;
                if ((params[0].dif(nomerconst, 1).name != "0")&&(params[1].dif(nomerconst,1).name=="0"))
                {
                    proizvodnay.type = type;
                    proizvodnay.name = op[proizvodnay.type];
                    proizvodnay.kolvoparam = 2;
                    proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    proizvodnay.params.push_back(params[1]);
                }
                if ((params[1].dif(nomerconst, 1).name != "0")&&(params[0].dif(nomerconst,1).name=="0"))
                {
                    proizvodnay.type = 3;
                    proizvodnay.name = op[proizvodnay.type];
                    proizvodnay.kolvoparam = 1;
                    expr min;
                    min.type = 4;
                    min.name = op[min.type];
                    min.kolvoparam = 2;
                    expr verh;
                    verh.type = 2;
                    verh.name = op[verh.type];
                    verh.params.push_back(params[1].dif(nomerconst, 1));
                    if (params[0].type == 2)
                    {
                        for (int i = 0; i < params[0].params.size(); i++)
                        {
                            verh.params.push_back(params[0].params[i]);
                        }
                    }
                    else
                    {
                        verh.params.push_back(params[0]);
                    }
                    verh.kolvoparam = verh.params.size();
                    expr two;
                    two.type = 12;
                    two.name = "2";
                    two.kolvoparam = 0;
                    two.uniqconst = 0;
                    expr niz;
                    niz.type = 11;
                    niz.name = op[niz.type];
                    niz.kolvoparam = 2;
                    niz.uniqconst = params[1].uniqconst;
                    niz.params.push_back(params[1]);
                    niz.params.push_back(two);
                    min.params.push_back(verh);
                    min.params.push_back(niz);
                    proizvodnay.params.push_back(min);
                }
                if ((params[1].dif(nomerconst, 1).name != "0") && (params[0].dif(nomerconst, 1).name != "0"))
                {
                    proizvodnay.type = 4;
                    proizvodnay.name = op[proizvodnay.type];
                    proizvodnay.kolvoparam = 2;
                    expr verh;
                    verh.type = 1;
                    verh.name = op[verh.type];
                    expr verh1;
                    verh1.type = 2;
                    verh1.name = op[verh1.type];
                    verh1.kolvoparam = 2;
                    verh1.params[0] = params[0].dif(nomerconst, 1);
                    verh.params[1] = params[1];
                    expr verh2;
                    verh2.type = 3;
                    verh2.name = op[verh.type];
                    verh2.kolvoparam = 1;
                    expr verh22;
                    verh22.type = 2;
                    verh22.name = op[verh22.type];
                    verh22.kolvoparam = 2;
                    verh22.params[0] = params[0];
                    verh22.params[1]= params[1].dif(nomerconst, 1);
                    verh2.params[0] = verh22;
                    verh.kolvoparam = 2;
                    verh.params[0] = verh1;
                    verh.params[1] = verh2;
                    expr two;
                    two.type = 12;
                    two.name = "2";
                    two.kolvoparam = 0;
                    two.uniqconst = 0;
                    expr niz;
                    niz.type = 11;
                    niz.name = op[niz.type];
                    niz.kolvoparam = 2;
                    niz.params[0] = params[1];
                    niz.params[1] = two;
                    proizvodnay.params.push_back(verh);
                    proizvodnay.params.push_back(niz);
                }
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
        }
        if (type == 5)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }

            proizvodnay.type = 3;
            proizvodnay.name = op[proizvodnay.type];
            proizvodnay.kolvoparam = 1;
            if (glag == 1)
            {
                proizvodnay.uniqconst = 0;
                expr mal;
                mal.type = 6;
                mal.name = op[mal.type];
                mal.uniqconst = 0;
                mal.params.push_back(params[0]);
                mal.kolvoparam = 1;
                proizvodnay.params.push_back(mal);
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
            else
            {
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                    expr mul;
                    mul.type = 2;
                    mul.name = op[mul.type];
                    if (params[0].dif(nomerconst, 1).type == 2)
                    {
                        mul.params = params[0].dif(nomerconst, 1).params;
                    }
                    else
                    {
                        mul.params.push_back(params[0].dif(nomerconst, 1));
                    }
                    expr sin;
                    sin.type = 6;
                    sin.name = op[sin.type];
                    sin.params = params;
                    sin.kolvoparam = kolvoparam;
                    sin.uniqconst = uniqconst--;
                    mul.params.push_back(sin);
                    mul.kolvoparam = mul.params.size();
                    proizvodnay.params.push_back(mul);
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
            }
        }
        if (type == 6)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }

            proizvodnay.type = 2;
            proizvodnay.name = op[proizvodnay.type];
            if (glag == 1)
            {
                proizvodnay.type = 5;
                proizvodnay.name = op[proizvodnay.type];
                proizvodnay.uniqconst = 0;
                proizvodnay.params.push_back(params[0]);
                proizvodnay.kolvoparam = 1;
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
            else
            {
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                    if (params[0].dif(nomerconst, 1).type == 2)
                    {
                        proizvodnay.params = params[0].dif(nomerconst, 1).params;
                    }
                    else
                    {
                        proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    }
                    expr cos;
                    cos.type = 5;
                    cos.name = op[cos.type];
                    cos.params = params;
                    cos.kolvoparam = kolvoparam;
                    cos.uniqconst = uniqconst--;
                    proizvodnay.params.push_back(cos);
                    proizvodnay.kolvoparam = proizvodnay.params.size();
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
            }
        }
        if (type == 7)
        {
            
            proizvodnay.type = 4;
            proizvodnay.name = op[proizvodnay.type];
            proizvodnay.kolvoparam = 2;
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                    expr mal;
                    mal.type = 11;
                    mal.name = op[mal.type];
                    mal.uniqconst = uniqconst--;
                    expr cos;
                    cos.type = 5;
                    cos.name = op[cos.type];
                    cos.kolvoparam = 1;
                    cos.params.push_back(params[0]);
                    cos.uniqconst = params[0].uniqconst--;
                    mal.params.push_back(cos);
                    expr two;
                    two.type = 12;
                    two.name = "2";
                    two.kolvoparam = 0;
                    two.uniqconst = 0;
                    mal.params.push_back(two);
                    mal.kolvoparam = 2;
                    proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    proizvodnay.params.push_back(mal);
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
            
        }
        if (type == 8)
        {
            proizvodnay.type = 4;
            proizvodnay.name = op[proizvodnay.type];
            proizvodnay.kolvoparam = 2;
            
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                    
                    proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    proizvodnay.params.push_back(params[0]);
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
            
        }
        if (type == 9)
        {
            proizvodnay.type = 2;
            proizvodnay.name = op[proizvodnay.type];
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                    expr e;
                    e.type = type;
                    e.name = name;
                    e.kolvoparam = 1;
                    e.params = params; 
                    if (params[0].dif(nomerconst, 1).type == 2)
                    {
                        proizvodnay.params = params[0].dif(nomerconst, 1).params;
                        proizvodnay.params.push_back(e);
                        proizvodnay.kolvoparam = params.size();
                        if (time == 0)
                        {
                            d.params.push_back(proizvodnay);
                            return d;
                        }
                        return proizvodnay;
                    }
                    proizvodnay.params.push_back(params[0].dif(nomerconst, 1));
                    proizvodnay.params.push_back(e);
                    proizvodnay.kolvoparam = proizvodnay.params.size();
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
        }
        if (type == 10)
        {
            int glag = 0;
            auto it = params.begin();
            while (it != params.end())
            {
                if ((*it).name == nomer)
                {
                    glag = 1;
                    break;
                }
                else
                    it++;
            }
            proizvodnay.type = 4;
            proizvodnay.name = op[proizvodnay.type];
            proizvodnay.kolvoparam = 2;
            expr one;
            one.type = 12;
            one.name = "1";
            one.kolvoparam = 0;
            one.uniqconst = 0;
            expr two;
            two.type = 12;
            two.name = "2";
            two.kolvoparam = 0;
            two.uniqconst = 0;
            expr iznach;
            iznach.type = type;
            iznach.name = name;
            iznach.params = params;
            iznach.kolvoparam = kolvoparam;
            if (glag == 1)
            {
               
                proizvodnay.params.push_back(one);
                expr proiz;
                proiz.type = 2;
                proiz.name = op[proiz.type];
                proiz.kolvoparam=2;
                proiz.params.push_back(two);
                proiz.params.push_back(iznach);
                proizvodnay.params.push_back(proiz);
                if (time == 0)
                {
                    d.params.push_back(proizvodnay);
                    return d;
                }
                return proizvodnay;
            }
            else
            {
                if (params[0].dif(nomerconst, 1).name != "0")
                {
                        proizvodnay.params.push_back((params[0].dif(nomerconst, 1)));
                        expr pr;
                        pr.type = 2;
                        pr.name = op[pr.type];
                        pr.kolvoparam = 2;
                        pr.params.push_back(two);
                        pr.params.push_back(iznach);
                        proizvodnay.params.push_back(pr);
                        if (time == 0)
                        {
                            d.params.push_back(proizvodnay);
                            return d;
                        }
                        return proizvodnay;
                }
                else
                {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
            }
        }
        if (type == 11)
        {
            proizvodnay.type = 2;
            proizvodnay.name = op[proizvodnay.type];
            expr iznach;
            iznach.type = type;
            iznach.name = name;
            iznach.params = params;
            iznach.kolvoparam = kolvoparam;
            expr ln;
            ln.type = 8;
            ln.name = op[ln.type];
            ln.kolvoparam = 1;
            ln.params.push_back(params[0]);
            if (params[0].dif(nomerconst, 1).name != "0")
               {
                 proizvodnay.params.push_back((params[0].dif(nomerconst, 1)));
                 expr nov_mnog;
                 nov_mnog.type = 1;
                 nov_mnog.name = op[nov_mnog.type];
                 expr one;
                 one.type = 12;
                 one.name = "1";
                 one.kolvoparam = 0;
                 one.uniqconst = 0;
                 expr minus;
                 minus.type = 3;
                 minus.name = op[minus.type];
                 minus.kolvoparam = 1;
                 minus.params.push_back(one);
                 nov_mnog.params.push_back(params[1]);
                 nov_mnog.params.push_back(minus);
                 proizvodnay.params.push_back(params[1]);
                 nov_mnog.kolvoparam = nov_mnog.params.size();
                 iznach.params[1] = nov_mnog;
                 proizvodnay.params.push_back(iznach);
                 proizvodnay.kolvoparam = proizvodnay.params.size();
                 if (time == 0)
                 {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;
                }
                else
                {
                if (params[1].dif(nomerconst, 1).name != "0")
                {
                    proizvodnay.params.push_back(iznach);
                    proizvodnay.params.push_back(ln);
                    if(params[1].dif(nomerconst, 1).name!="1")
                    { 
                    proizvodnay.params.push_back(params[1].dif(nomerconst, 1));
                    }
                    proizvodnay.kolvoparam = proizvodnay.params.size();
                    if (time == 0)
                    {
                        d.params.push_back(proizvodnay);
                        return d;
                    }
                    return proizvodnay;

                }
                else {
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.kolvoparam = 0;
                    zero.uniqconst = 0;
                    if (time == 0)
                    {
                        d.params.push_back(zero);
                        return d;
                    }
                    return zero;
                }
                }
        }
    }

    void optimization()
    {
        
        int flag = 0;
        while(flag==0)
        {
            int typecon=0;
            flag = 1;
            if (type == 1)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                int size1 = params.size();
                int i=0;
                
                for (int i = 0; i < params.size(); i++)
                {
                    vector<int> xcount;
                    vector<int> ycount;
                    vector<int> zcount;
                    vector<int> concount;
                    vector<int> mincount;
                    if (params[i].type == 1)
                    {
                        for (int j = 0; j < params[i].params.size(); j++)
                        {
                            params.push_back(params[i].params[j]);
                        }
                        params.erase(params.begin() + i);
                        flag = 0;
                    }
                    if (params[i].type == 3)
                    {
                        
                        mincount.push_back(i);
                        if (i != params.size() - 1)
                        {
                            for (int j = i + 1; j < params.size(); j++)
                            {
                                if (params[j].type == 3)
                                {
                                    
                                    mincount.push_back(j);
                                }
                            }
                        }
                        if(mincount.size()>1)
                        { 
                            expr sum;
                            sum.type = 1;
                            sum.name = op[sum.type];
                            sum.kolvoparam = mincount.size();
                            expr slag;
                            for (int k = mincount.size()-1; k >=0; k--)
                            {
                                slag = params[mincount[k]].params[0];
                                sum.params.push_back(slag);
                                
                                params.erase(params.begin() + mincount[k]);
                            }
                            
                            expr min;
                            min.type = 3;
                            min.name = op[min.type];
                            min.kolvoparam = 1;
                            vector<int> con1;
                            min.uniqconst = sum.kolvouniqconsts(con1);
                            min.params.push_back(sum);
                           
                            params.push_back(min);
                            expr temp;
                            if (params.size() == 1)
                            {
                                expr temp = params[0];
                                (*this) = temp;
                            }
                            vector<int> con;
                            uniqconst = (*this).kolvouniqconsts(con);
                            flag = 0;
                            min.kolvoparam = 1;
                        }
                    }
                    if (params[i].type == 13)
                    {
                         xcount.push_back(i);
                         if (i != params.size() - 1)
                         {
                             for (int j = i + 1; j < params.size(); j++)
                             {
                                 if (params[j].type == 13)
                                 {

                                     xcount.push_back(j);
                                 }
                                 if (params[j].type == 3)
                                 {
                                     if (params[j].params[0].type == 13)
                                     {
                                         params.erase(params.begin() + j);
                                         params.erase(params.begin() + i);
                                         break;
                                     }
                                 }
                                 if (params[j].type == 2)
                                 {
                                     int fl = 0;
                                     for (int k = 0; k < params[j].params.size(); k++)
                                     {
                                         if ((params[j].params[k].type != 12) || (params[j].params[k].type != 13))
                                         {
                                             fl = 1;
                                             break;
                                         } 
                                        
                                     }
                                     if (fl == 0)
                                     {
                                         xcount.push_back(j);
                                     }
                                 }
                             }
                             if (xcount.size() > 1)
                             {
                                 for (int k = xcount.size()-1; k >= 0; k--)
                                 {
                                     params.erase(params.begin() + xcount[k]);
                                 }
                                 vector<int> con;
                                 uniqconst = (*this).kolvouniqconsts(con);
                                 expr var;
                                 var.type = 13;
                                 var.name = op[var.type];
                                 expr mul;
                                 mul.type = 2;
                                 mul.name = op[mul.type];
                                 mul.kolvoparam = 2;
                                 mul.params.push_back(createconst((*this).uniqconst+1));
                                 mul.params.push_back(var);
                                 mul.uniqconst = 1;
                                 params.push_back(mul);
                             }
                         }
                    }
                    if (params[i].type == 14)
                    {
                        ycount.push_back(i);
                        if (i != params.size() - 1)
                        {
                            for (int j = i + 1; j < params.size(); j++)
                            {
                                if (params[j].type == 14)
                                {

                                    ycount.push_back(j);
                                }
                                if (params[j].type == 3)
                                {
                                    if (params[j].params[0].type == 14)
                                    {
                                        params.erase(params.begin() + j);
                                        params.erase(params.begin() + i);
                                        break;
                                    }
                                }
                                if (params[j].type == 2)
                                {
                                    int fl = 0;
                                    for (int k = 0; k < params[j].params.size(); k)
                                    {
                                        if ((params[j].params[k].type != 12) || (params[j].params[k].type != 14))
                                        {
                                            fl = 1;
                                            break;
                                        }

                                    }
                                    if (fl == 0)
                                    {
                                        ycount.push_back(j);
                                    }
                                }
                            }
                            if (ycount.size() > 1)
                            {
                                for (int k = ycount.size() - 1; k >= 0; k++)
                                {
                                    params.erase(params.begin() + ycount[k]);

                                }
                                vector<int> con;
                                uniqconst = (*this).kolvouniqconsts(con);
                                expr var;
                                var.type = 14;
                                var.name = op[var.type];
                                expr mul;
                                mul.type = 2;
                                mul.name = op[mul.type];
                                mul.kolvoparam = 2;
                                mul.params.push_back(createconst((*this).uniqconst + 1));
                                mul.params.push_back(var);
                                mul.uniqconst = 1;
                                params.push_back(mul);
                            }
                        }
                    }
                    if (params[i].type == 15)
                    {
                        zcount.push_back(i);
                        if (i != params.size() - 1)
                        {
                            for (int j = i + 1; j < params.size(); j++)
                            {
                                if (params[j].type == 15)
                                {

                                    zcount.push_back(j);
                                }
                                if (params[j].type == 3)
                                {
                                    if (params[j].params[0].type == 15)
                                    {
                                        params.erase(params.begin() + j);
                                        params.erase(params.begin() + i);
                                        break;
                                    }
                                }
                                if (params[j].type == 2)
                                {
                                    int fl = 0;
                                    for (int k = 0; k < params[j].params.size(); k++)
                                    {
                                        if ((params[j].params[k].type != 12) || (params[j].params[k].type != 15))
                                        {
                                            fl = 1;
                                            break;
                                        }

                                    }
                                    if (fl == 0)
                                    {
                                        zcount.push_back(j);
                                    }
                                }
                            }
                            if (zcount.size() > 1)
                            {
                                for (int k = zcount.size() - 1; k >= 0; k++)
                                {
                                    params.erase(params.begin() + zcount[k]);
                                }
                                vector<int> con;
                                uniqconst = (*this).kolvouniqconsts(con);
                                expr var;
                                var.type = 15;
                                var.name = op[var.type];
                                expr mul;
                                mul.type = 2;
                                mul.name = op[mul.type];
                                mul.kolvoparam = 2;
                                mul.params.push_back(createconst((*this).uniqconst + 1));
                                mul.params.push_back(var);
                                mul.uniqconst = 1;
                                params.push_back(mul);
                            }
                        }
                    }
                }
                while (i < size1 - 1)
                {
                    expr osnova = params[i];
                    vector<int> new_mul;
                    vector<int> num1;
                    params[i].makeint(num1);
                    new_mul.push_back(i);

                    int j = i + 1;

                    while (j < size1)
                    {
                        params[j].kolvoparam = params[j].params.size();
                        vector <int> num2;
                        params[j].makeint(num2);

                        if (num1 == num2)
                        {
                            new_mul.push_back(j);
                        }
                        j++;
                    }

                    if (new_mul.size() > 1)
                    {
                        expr mul;
                        mul.type = 2;
                        mul.kolvoparam = 2;
                        mul.name = op[mul.type];
                        for (int k = new_mul.size() - 1; k >= 0; k--)
                        {
                            params.erase(params.begin() + new_mul[k]);
                            size1--;
                        }
                        mul.params.push_back(createconst(0));
                        mul.params.push_back(osnova);
                        if (params.size() < 1)
                        {
                            (*this) = mul;
                        }
                        else {
                            params.insert(params.begin() + i, mul);
                        }
                        (*this).rasstanovka_nomerov_consts(0);
                        size1++;
                        i--;
                        flag = 1;
                    }
                    i++;
                }
            }
            if (type == 2)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                


                int size1 = params.size();
                int i = 0;
                while (i < size1 - 1)
                {
                    expr osnova = params[i];
                    vector<int> new_step;
                    vector<int> num1;
                    params[i].makeint(num1);
                    new_step.push_back(i);
                    int j = i + 1;
                    while (j < size1)
                    {
                        vector <int> num2;
                        params[j].makeint(num2);
                        if (num1 == num2)
                        {
                            new_step.push_back(j);
                        }
                        j++;
                    }
                    if (new_step.size() > 1)
                    {
                        expr step;
                        step.type = 11;
                        step.kolvoparam = 2;
                        step.name = op[step.type];
                        for (int k = new_step.size() - 1; k >= 0; k--)
                        {
                            params.erase(params.begin() + new_step[k]);
                            size1--;
                        }
                        step.params.push_back(osnova);
                        step.params.push_back(createconst(0));
                        if (params.size() < 1)
                        {
                            (*this) = step;
                        }
                        else{
                        params.insert(params.begin() + i, step);
                        }
                        (*this).rasstanovka_nomerov_consts(0);
                        size1++;
                        i--;
                        flag = 1;
                    }
                    i++;
                }
            }
            if (type == 3)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 3)
                {
                    (*this) = params[0].params[0];
                    flag = 0;
                }
            }
            if (type == 4)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if ((params[0].type == 3) && (params[1].type == 3))
                {
                    expr temp1;
                    temp1= params[0].params[0];
                    params[0] = temp1;
                    expr temp2;
                    temp2 = params[1].params[0];
                    params[1] = temp2;
                }
                if ((params[0].type == 12) && (params[1].type == 12))
                {
                    (*this) = createconst(0);
                }
                if (((params[0].type == 13) && (params[1].type == 13))|| ((params[0].type == 14) && (params[1].type == 14))||((params[0].type == 15) && (params[1].type == 15)))
                {
                    
                    (*this) = createconst(uniqconst + 1);
                }
            }
            if (type == 5)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 12)
                {
                    
                    (*this) = params[0];
                }
            }
            if (type == 6)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 12)
                {
                    (*this) = params[0];
                }
            }
            if (type == 7)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 12)
                {
                    (*this) = params[0];
                }
            }
            if (type == 8)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 9)
                {
                    (*this) = params[0].params[0];
                    flag = 0;
                }
            }
            if (type == 9)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 8)
                {
                    (*this) = params[0].params[0];
                    flag = 0;
                }

            }
            if (type == 10)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 11)
                {
                    type = 11;
                    name = op[type];
                    kolvoparam = 2;
                    expr two;
                    two.type = 12;
                    two.name = "2";
                    two.kolvoparam = 0;
                    two.uniqconst = 0;
                    expr stepen;
                    stepen.type = 4;
                    stepen.name = op[stepen.type];
                    stepen.kolvoparam = 2;
                    stepen.uniqconst = params[0].params[1].uniqconst;
                    stepen.params.push_back ( params[0].params[1]);
                    stepen.params.push_back(two);
                    params.push_back(stepen);
                    params[0] = params[0].params[0];
                    flag = 0;
                }
            }
            if (type == 11)
            {
                for (int i = 0; i < params.size(); i++)
                {
                    params[i].optimization();
                }
                if (params[0].type == 11)
                {
                    expr proizv;
                    proizv.type = 2;
                    proizv.name = op[proizv.type];
                    proizv.params.push_back(params[0].params[1]);
                    proizv.params.push_back(params[1]);
                    proizv.kolvoparam = proizv.params.size();
                    params[1] = proizv;
                    flag = 0;
                }
                if ((params[0].type == 12)&&(params[1].type==12))
                {
                    uniqconst--;
                    type = 12;
                    (*this) = createconst(uniqconst);
                    flag = 0;
                }
            }
            kolvoparam = params.size();
        }
        
    }
    int ext(vector<expr>& exprs)
    {
        if (type == 12)
        {
            return 0;
        }
        if (type != 1)
        {
            for (int i = 0; i < 2; i++)
            {
                expr var;
                var.type = 13 + i;
                var.name = op[var.type];
                expr mul;
                mul.type = 2;
                mul.name = op[mul.type];
                mul.params.push_back(createconst(uniqconst));
                mul.params.push_back(var);
                for (int j = 2; j < 12; j++)
                {
                    expr sum;
                    sum.type = 1;
                    sum.name = op[sum.type];
                    sum.params.push_back((*this));
                    expr ex;
                    switch (j)
                    {
                    case 2:
                        sum.params.push_back(mul);
                        break;
                    case 3:
                        ex.uniqconst = 0;
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        for (int i = 0; i < ex.params.size(); i++)
                            ex.uniqconst += ex.params[i].uniqconst;
                        sum.params.push_back(ex);
                        break;
                    case 4:
                        ex.uniqconst = 0;
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.type = j;
                        ex.kolvoparam = 2;
                        ex.params.push_back(createconst(ex.uniqconst));
                        ex.params.push_back(var);
                        for (int i = 0; i < ex.params.size(); i++)
                            ex.uniqconst += ex.params[i].uniqconst;
                        sum.params.push_back(ex);
                        break;
                    case 5:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        sum.params.push_back(ex);
                        break;
                    case 6:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 7:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 8:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 9:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 10:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 11:
                        ex.type = 2;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 2;
                        ex.params.push_back(createconst(0));
                        ex.uniqconst = 2;
                        expr ex1;
                        ex1.type = j;
                        ex1.name = op[ex1.type];
                        ex1.uniqconst = 1;
                        ex1.kolvoparam = 2;
                        ex1.params.push_back(var);
                        ex1.params.push_back(createconst(ex1.uniqconst + 1));
                        ex.params.push_back(ex1);
                        sum.params.push_back(ex);
                        break;
                    }
                    exprs.push_back(sum);
                }
            }
            for (int i = 0; i < params.size(); i++)
            {
                expr exxx;
                exxx.type = type;
                exxx.name = name;
                exxx.params = params;
                vector<expr> exx;
                params[i].ext(exx);
                for (int j = 0; j < exx.size(); j++)
                {
                    exxx.params[i] = exx[j];
                    expr temp = exxx;
                    exprs.push_back(temp);
                }
            }
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                expr var;
                var.type = 13 + i;
                var.name = op[var.type];
                expr mul;
                mul.type = 2;
                mul.name = op[mul.type];
                mul.params.push_back(createconst(uniqconst));
                mul.params.push_back(var);
                for (int j = 2; j < 12; j++)
                {
                    expr sum;
                    sum.type = type;
                    sum.name = op[sum.type];
                    sum.params = params;
                    expr ex;
                    switch (j)
                    {
                    case 2:
                        sum.params.push_back(mul);
                        break;
                    case 3:
                        ex.uniqconst = 0;
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        for (int i = 0; i < ex.params.size(); i++)
                            ex.uniqconst += ex.params[i].uniqconst;
                        sum.params.push_back(ex);
                        break;
                    case 4:
                        ex.uniqconst = 0;
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.type = j;
                        ex.kolvoparam = 2;
                        ex.params.push_back(createconst(ex.uniqconst));
                        ex.params.push_back(var);
                        for (int i = 0; i < ex.params.size(); i++)
                            ex.uniqconst += ex.params[i].uniqconst;
                        sum.params.push_back(ex);
                        break;
                    case 5:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        sum.params.push_back(ex);
                        break;
                    case 6:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 7:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 8:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 9:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 10:
                        ex.type = j;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 1;
                        ex.params.push_back(mul);
                        ex.uniqconst = 1;
                        sum.params.push_back(ex);
                        break;
                    case 11:
                        ex.type = 2;
                        ex.name = op[ex.type];
                        ex.kolvoparam = 2;
                        ex.params.push_back(createconst(0));
                        ex.uniqconst = 2;
                        expr ex1;
                        ex1.type = j;
                        ex1.name = op[ex1.type];
                        ex1.uniqconst = 1;
                        ex1.kolvoparam = 2;
                        ex1.params.push_back(var);
                        ex1.params.push_back(createconst(ex1.uniqconst + 1));
                        ex.params.push_back(ex1);
                        sum.params.push_back(ex);
                        break;
                    }
                    exprs.push_back(sum);
                }
            }
            for (int i = 0; i < params.size(); i++)
            {
                expr exxx;
                exxx.type = type;
                exxx.name = name;
                exxx.params = params;
                vector<expr> exx;
                params[i].ext(exx);
                for (int j = 0; j < exx.size(); j++)
                {
                    exxx.params[i] = exx[j];
                    expr temp = exxx;
                    exprs.push_back(temp);
                }
            }
        }
        return 0;
    }
    vector<expr> exten(Node& root)
    { 
        int t = 0;
        vector <expr> potomki;
        while (t < 3)
        {
            if (t == 0)
            {
                for (int i = 0; i < 2; i++)
                {
                    expr var;
                    var.uniqconst = 0;
                    var.kolvoparam = 0;
                    if (i == 0)
                    {
                        var.type = 13;
                        var.name = op[var.type];
                    }
                    if (i == 1)
                    {
                        var.type = 14;
                        var.name = op[var.type];
                    }
                    expr mul;
                    mul.type = 2;
                    mul.name = op[mul.type];
                    mul.kolvoparam = 2;
                    mul.params.push_back(createconst(uniqconst));
                    mul.params.push_back(var);
                    mul.uniqconst = 1;
                    
                    if((type!=2)&&(type!=3))
                    { 
                        if (type == 1)
                        {
                            for (int j = 1; j < 12; j++)
                            {
                                vector <int> kol;
                                expr exdop;
                                expr ex;
                                ex.type = type;
                                ex.name = name;
                                ex.params = params;
                                ex.uniqconst = uniqconst;
                                switch (j)
                                {
                                case 1:
                                    ex.params.push_back(mul);
                                    ex.params.push_back(createconst(ex.uniqconst + 1));
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                  
                                    potomki.push_back(ex);
                                    break;
                                case 3:
                                    exdop.uniqconst = 0;
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    for (int i = 0; i < exdop.params.size(); i++)
                                        exdop.uniqconst += exdop.params[i].uniqconst;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    
                                    potomki.push_back(ex);
                                    break;
                                case 4:
                                    exdop.uniqconst = 0;
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 2;
                                    exdop.params.push_back(createconst(ex.uniqconst));
                                    exdop.params.push_back(var);
                                    for (int i = 0; i < exdop.params.size(); i++)
                                        exdop.uniqconst += exdop.params[i].uniqconst;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                   
                                    potomki.push_back(ex);
                                    break;
                                case 5:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    
                                    potomki.push_back(ex);
                                    break;
                                case 6:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    
                                    potomki.push_back(ex);
                                    break;
                                case 7:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    
                                    potomki.push_back(ex);
                                    break;
                                case 8:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 9:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                   potomki.push_back(ex);
                                    break;
                                case 10:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 11:
                                    exdop.type = 2;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 2;
                                    exdop.params.push_back(createconst(uniqconst));
                                    exdop.uniqconst = 2;
                                    expr ex1;
                                    ex1.type = j;
                                    ex1.name = op[ex1.type];
                                    ex1.uniqconst = 1;
                                    ex1.kolvoparam = 2;
                                    ex1.params.push_back(var);
                                    ex1.params.push_back(createconst(uniqconst + 1));
                                    exdop.params.push_back(ex1);
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                }
                            }
                        }
                        else {
                            for (int j = 1; j < 12; j++)
                            {
                                expr ex;
                                ex.type = 1;
                                ex.name = op[ex.type];
                                expr u = *this;
                                ex.params.push_back(u);
                                ex.uniqconst = uniqconst;
                                vector <int> kol;
                                expr exdop;
                                switch (j)
                                {
                                case 1:
                                    ex.params.push_back(createconst(ex.uniqconst + 1));
                                    ex.params.push_back(mul);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                   potomki.push_back(ex);
                                    break;
                                case 2:
                                    ex.params.push_back(mul);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 3:
                                    exdop.uniqconst = 0;
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    for (int i = 0; i < exdop.params.size(); i++)
                                        exdop.uniqconst += exdop.params[i].uniqconst;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 4:
                                    exdop.uniqconst = 0;
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 2;
                                    exdop.params.push_back(createconst(ex.uniqconst));
                                    exdop.params.push_back(var);
                                    for (int i = 0; i < ex.params.size(); i++)
                                        exdop.uniqconst += exdop.params[i].uniqconst;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 5:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 6:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 7:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 8:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 9:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 10:
                                    exdop.type = j;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 1;
                                    exdop.params.push_back(mul);
                                    exdop.uniqconst = 1;
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                case 11:
                                    exdop.type = 2;
                                    exdop.name = op[exdop.type];
                                    exdop.kolvoparam = 2;
                                    exdop.params.push_back(createconst(uniqconst));
                                    exdop.uniqconst = 2;
                                    expr ex1;
                                    ex1.type = j;
                                    ex1.name = op[ex1.type];
                                    ex1.uniqconst = 1;
                                    ex1.kolvoparam = 2;
                                    ex1.params.push_back(var);
                                    ex1.params.push_back(createconst(uniqconst + 1));
                                    exdop.params.push_back(ex1);
                                    ex.params.push_back(exdop);
                                    ex.kolvoparam = ex.params.size();
                                    ex.uniqconst = ex.kolvouniqconsts(kol);
                                    potomki.push_back(ex);
                                    break;
                                }
                            }
                        }
                    }
                    else {
                        
                        for (int j = 1; j < 12; j++)
                        {
                            expr ex;
                            ex.type = 1;
                            ex.name = op[ex.type];
                            expr u = *this;
                            ex.params.push_back(u);
                            ex.uniqconst = uniqconst;
                            vector <int> kol;
                            expr exdop;
                            switch (j)
                            {
                            case 1:
                                ex.params.push_back(createconst(ex.uniqconst + 1));
                                ex.params.push_back(mul);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                potomki.push_back(ex);
                                break;
                            case 3:
                                exdop.uniqconst = 0;
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                for (int i = 0; i < exdop.params.size(); i++)
                                    exdop.uniqconst += exdop.params[i].uniqconst;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                potomki.push_back(ex);
                                break;
                            case 4:
                                exdop.uniqconst = 0;
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.type = j;
                                exdop.kolvoparam = 2;
                                exdop.params.push_back(createconst(ex.uniqconst));
                                exdop.params.push_back(var);
                                for (int i = 0; i < ex.params.size(); i++)
                                    exdop.uniqconst += exdop.params[i].uniqconst;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                               
                                potomki.push_back(ex);
                                break;
                            case 5:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                kol.clear();
                                
                                potomki.push_back(ex);
                                break;
                            case 6:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                exdop.uniqconst = 1;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            case 7:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                exdop.uniqconst = 1;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            case 8:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                exdop.uniqconst = 1;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            case 9:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                exdop.uniqconst = 1;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            case 10:
                                exdop.type = j;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 1;
                                exdop.params.push_back(mul);
                                exdop.uniqconst = 1;
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            case 11:
                                exdop.type = 2;
                                exdop.name = op[exdop.type];
                                exdop.kolvoparam = 2;
                                exdop.params.push_back(createconst(uniqconst ));
                                exdop.uniqconst = 2;
                                expr ex1;
                                ex1.type = j;
                                ex1.name = op[ex1.type];
                                ex1.uniqconst = 1;
                                ex1.kolvoparam = 2;
                                ex1.params.push_back(var);
                                ex1.params.push_back(createconst(uniqconst + 1));
                                exdop.params.push_back(ex1);
                                ex.params.push_back(exdop);
                                ex.kolvoparam = ex.params.size();
                                ex.uniqconst = ex.kolvouniqconsts(kol);
                                
                                
                                potomki.push_back(ex);
                                break;
                            }
                        }
                    }
                }
            }
            if (t == 1)
            {
                for (int i = 0; i < 2; i++)
                {
                    expr var;
                    var.uniqconst = 0;
                    var.kolvoparam = 0;
                    if (i == 0)
                    {
                        var.type = 13;
                        var.name = op[var.type];
                    }
                    if (i == 1)
                    {
                        var.type = 14;
                        var.name = op[var.type];
                    }
                    expr mul;
                    mul.type = 2;
                    mul.name = op[mul.type];
                    mul.kolvoparam = 2;
                    mul.params.push_back(createconst(uniqconst));
                    mul.params.push_back(var);
                    mul.uniqconst = 1;
                    for (int k = 0; k < 2; k++)
                    {
                        if (k == 0)
                        {
                            if (type == 3)
                            {
                                for (int j = 1; j < 12; j++)
                                {
                                    vector <int> kol;
                                    expr exdop;
                                    expr ex;
                                    ex.type = type;
                                    ex.name = name;
                                    ex.params = params;
                                    ex.uniqconst = uniqconst;
                                    switch (j)
                                    {
                                    case 1:
                                        ex.params[0].params.push_back(mul);
                                        ex.params[0].params.push_back(createconst(ex.uniqconst + 1));
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 3:
                                        exdop.uniqconst = 0;
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        for (int i = 0; i < exdop.params.size(); i++)
                                            exdop.uniqconst += exdop.params[i].uniqconst;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 4:
                                        exdop.uniqconst = 0;
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.type = j;
                                        exdop.kolvoparam = 2;
                                        exdop.params.push_back(createconst(ex.uniqconst));
                                        exdop.params.push_back(var);
                                        for (int i = 0; i < ex.params.size(); i++)
                                            exdop.uniqconst += exdop.params[i].uniqconst;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 5:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 6:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 7:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 8:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 9:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 10:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 11:
                                        exdop.type = 2;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 2;
                                        exdop.params.push_back(createconst(uniqconst));
                                        exdop.uniqconst = 2;
                                        expr ex1;
                                        ex1.type = j;
                                        ex1.name = op[ex1.type];
                                        ex1.uniqconst = 1;
                                        ex1.kolvoparam = 2;
                                        ex1.params.push_back(var);

                                        ex1.params.push_back(createconst(uniqconst + 1));
                                        exdop.params.push_back(ex1);
                                        ex.params[0].params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    }
                                }
                            }
                            else {
                                if (type == 2)
                                {

                                    for (int j = 1; j < 12; j++)
                                    {
                                        vector <int> kol;
                                        expr exdop;
                                        expr ex;
                                        ex.type = type;
                                        ex.name = name;
                                        ex.params = params;
                                        ex.uniqconst = uniqconst;
                                        switch (j)
                                        {
                                        case 1:
                                            ex.params.push_back(mul);
                                            ex.params.push_back(createconst(ex.uniqconst + 1));
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 3:
                                            exdop.uniqconst = 0;
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            for (int i = 0; i < exdop.params.size(); i++)
                                                exdop.uniqconst += exdop.params[i].uniqconst;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 4:
                                            exdop.uniqconst = 0;
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.type = j;
                                            exdop.kolvoparam = 2;
                                            exdop.params.push_back(createconst(ex.uniqconst));
                                            exdop.params.push_back(var);
                                            for (int i = 0; i < ex.params.size(); i++)
                                                exdop.uniqconst += exdop.params[i].uniqconst;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 5:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 6:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 7:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 8:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 9:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 10:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 11:
                                            exdop.type = 2;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 2;
                                            exdop.params.push_back(createconst(uniqconst));
                                            exdop.uniqconst = 2;
                                            expr ex1;
                                            ex1.type = j;
                                            ex1.name = op[ex1.type];
                                            ex1.uniqconst = 1;
                                            ex1.kolvoparam = 2;
                                            ex1.params.push_back(var);

                                            ex1.params.push_back(createconst(uniqconst + 1));
                                            exdop.params.push_back(ex1);
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        }
                                    }

                                }
                                else {
                                    for (int j = 1; j < 12; j++)
                                    {
                                        expr ex;
                                        ex.type = 2;
                                        ex.name = op[ex.type];
                                        expr temp = (*this);
                                        ex.params.push_back(temp);
                                        ex.uniqconst = uniqconst;
                                        vector <int> kol;
                                        expr exdop;
                                        expr s;
                                        switch (j)
                                        {
                                        case 1:
                                            s.type = 1;
                                            s.name = op[s.type];
                                            s.params.push_back(mul);
                                            s.params.push_back(createconst(ex.uniqconst + 1));
                                            ex.params.push_back(s);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 3:
                                            exdop.uniqconst = 0;
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            for (int i = 0; i < exdop.params.size(); i++)
                                                exdop.uniqconst += exdop.params[i].uniqconst;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 4:
                                            exdop.uniqconst = 0;
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 2;
                                            exdop.params.push_back(createconst(ex.uniqconst));
                                            exdop.params.push_back(var);
                                            for (int i = 0; i < ex.params.size(); i++)
                                                exdop.uniqconst += exdop.params[i].uniqconst;

                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 5:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 6:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 7:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 8:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 9:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 10:
                                            exdop.type = j;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 1;
                                            exdop.params.push_back(mul);
                                            exdop.uniqconst = 1;
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        case 11:
                                            exdop.type = 2;
                                            exdop.name = op[exdop.type];
                                            exdop.kolvoparam = 2;
                                            exdop.params.push_back(createconst(uniqconst));
                                            exdop.uniqconst = 2;
                                            expr ex1;
                                            ex1.type = j;
                                            ex1.name = op[ex1.type];
                                            ex1.uniqconst = 1;
                                            ex1.kolvoparam = 2;

                                            ex1.params.push_back(var);
                                            ex1.params.push_back(createconst(uniqconst + 1));
                                            exdop.params.push_back(ex1);
                                            ex.params.push_back(exdop);
                                            ex.kolvoparam = ex.params.size();
                                            ex.uniqconst = ex.kolvouniqconsts(kol);
                                            
                                            
                                            potomki.push_back(ex);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if (k == 1)
                        {
                            if (type == 4) 
                            {

                                for (int j = 1; j < 12; j++)
                                {
                                    vector <int> kol;
                                    expr exdop;
                                    expr ex;
                                    ex.type = type;
                                    ex.name = name;
                                    ex.params = params;
                                    ex.kolvoparam = params.size();
                                    ex.uniqconst = uniqconst;
                                    expr s;
                                    expr m;
                                    expr temp1;
                                    expr temp2;
                                    switch (j)
                                    {
                                    case 1:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        s.type = 1;
                                        s.name = op[s.type];
                                        s.params.push_back(mul);
                                        s.params.push_back(createconst(ex.uniqconst + 1));
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        m.params.push_back(s);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1]=temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 2:
                                        m = mul;
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 3:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.uniqconst = 0;
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = mul.uniqconst;
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 5:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1]=temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 6:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 7:
                                        m.type = 2;
                                        m.name = op[m.type]; 
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 8:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 9:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 10:
                                        m.type = 2;
                                        m.name = op[m.type];
                                        temp1 = ex.params[1];
                                        m.params.push_back(temp1);
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        m.params.push_back(exdop);
                                        m.kolvoparam = m.params.size();
                                        temp2 = m;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 11:
                                        exdop.type = 2;
                                        exdop.name = op[exdop.type];
                                        temp1 = ex.params[1];
                                        exdop.params.push_back(temp1);
                                        exdop.params.push_back(createconst(uniqconst));
                                        exdop.kolvoparam = exdop.params.size();
                                        exdop.uniqconst = 2+ temp1.uniqconst;
                                        expr ex1;
                                        ex1.type = j;
                                        ex1.name = op[ex1.type];
                                        ex1.uniqconst = 1;
                                        ex1.kolvoparam = 2;
                                        ex1.params.push_back(var);
                                        ex1.params.push_back(createconst(uniqconst + 1));
                                        exdop.params.push_back(ex1);
                                        temp2 = exdop;
                                        ex.params[1] = temp2;
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    }
                                }

                            }
                            else {
                                
                                for (int j = 1; j < 12; j++)
                                {
                                    expr ex;
                                    ex.type = 4;
                                    ex.name = op[ex.type];
                                    ex.params.push_back((*this));
                                    ex.uniqconst = uniqconst;
                                    vector <int> kol;
                                    expr exdop;
                                    expr s;
                                    switch (j)
                                    {
                                    case 1:
                                        s.type = 1;
                                        s.name = op[s.type];
                                        s.params.push_back(createconst(ex.uniqconst + 1));
                                        s.params.push_back(mul);
                                        ex.params.push_back(s);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 3:
                                        exdop.uniqconst = 0;
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        for (int i = 0; i < exdop.params.size(); i++)
                                            exdop.uniqconst += exdop.params[i].uniqconst;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 5:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 6:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 7:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 8:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 9:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 10:
                                        exdop.type = j;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 1;
                                        exdop.params.push_back(mul);
                                        exdop.uniqconst = 1;
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    case 11:
                                        exdop.type = 2;
                                        exdop.name = op[exdop.type];
                                        exdop.kolvoparam = 2;
                                        exdop.params.push_back(createconst(uniqconst));
                                        exdop.uniqconst = 2;
                                        expr ex1;
                                        ex1.type = j;
                                        ex1.name = op[ex1.type];
                                        ex1.uniqconst = 1;
                                        ex1.kolvoparam = 2;
                                        ex1.params.push_back(var);

                                        ex1.params.push_back(createconst(uniqconst + 1));
                                        exdop.params.push_back(ex1);
                                        ex.params.push_back(exdop);
                                        ex.kolvoparam = ex.params.size();
                                        ex.uniqconst = ex.kolvouniqconsts(kol);
                                        
                                        
                                        potomki.push_back(ex);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (t == 2)
            {
                if (type == 1) 
                {
                    for (int i = 0; i < params.size(); i++)
                    {
                        if (params[i].type < 12)
                        {
                            for (int j = 0; j < params[i].params.size(); j++)
                            {
                                vector<expr> ex;
                                params[i].params[j].ext(ex);
                                for (int k = 0; k < ex.size(); k++)
                                {
                                    expr exprn;
                                    exprn.type = (*this).type;
                                    exprn.name = op[exprn.type];
                                    exprn.kolvoparam = (*this).kolvoparam;
                                    exprn.params = (*this).params;
                                    exprn.params[i].params[j] = ex[k];
                                    potomki.push_back(exprn);
                                }
                            }
                        }
                   }
                }
                else
                {
                    for (int i = 0; i < params.size(); i++)
                    {
                        if (params[i].type < 12)
                        {
                            vector<expr> ex;
                            params[i].ext(ex);
                            for (int k = 0; k < ex.size(); k++)
                            {
                                expr exprn;
                                exprn.type = (*this).type;
                                exprn.name = op[exprn.type];
                                exprn.kolvoparam = (*this).kolvoparam;
                                exprn.params = (*this).params;
                                exprn.params[i] = ex[k];
                                potomki.push_back(exprn);
                            }
                        }
                    }
                }
            }
            t++;
        }
       
        return potomki;
    }

    
    int kolvouniqconsts(vector<int> &consts)
    {
        if (type == 12)
        {
            if (name.find("CONST") != std::string::npos)
            {
                int est=0;
                string nn = name.substr(5, (*name.end() - 1) - 5);
                int n1 = stoi(nn);
                for (int i = 0; i < consts.size(); i++)
                {
                    if (consts[i] == n1)
                    {
                        est = 1;
                        break;
                    }
                }
                if (est == 0)
                {
                    consts.push_back(n1);
                }
            }
            return consts.size();
        }
        else
        {
            
            for (int i = 0; i < params.size(); i++)
            {
                params[i].kolvouniqconsts(consts);
            }
            return consts.size();
        }
    }
};

vector<expr> temporal;

struct thread_inform
{
    double* fitx;
    double* fity;
    double** fitz;
    int fitconstssize;
    expr fitexpr;
    cl_command_queue command_queue;
    int vhod;
    int kolvo_na_1;
    int nomer;
};
thread_inform info1, info2, info3, info4, info5, info6, info7, info8;
expr createconst(int unic)
{
    expr con;
    con.type = 12;
    con.kolvoparam = 0;
    con.uniqconst = 1;
    con.name = "CONST" + to_string(unic + 1);
    return con;
}



vector<double> solve_lm(expr f, int nomer_thread, double* consts, int consts_size)
{
    real_1d_array cons;
    real_1d_array s;
    
    double* ss = new double[n * m];
    double* constsx = new double[consts_size];
    for (int i = 0; i < consts_size; i++)
    {
        constsx[i] = consts[i];
    }
    for (int i = 0; i < n * m; i++)
    {
        ss[i] = 1;
    }
    s.setlength(n*m);
    cons.setlength(consts_size);
    s.setcontent(n * m, ss);
    cons.setcontent(consts_size, constsx);
    minlmstate state;
    double epsx = 0.0000001;
    int maxits = 10;
    minlmcreatev(n * m, cons, 0.0001, state);
    minlmsetcond(state, epsx, maxits);
    minlmsetscale(state, s);
    minlmreport rep;
    vector <double> consst;
    switch (nomer_thread)
    {
    case 1:
        info1.fitexpr = f;
        info1.fitconstssize = consts_size;
        minlmoptimize(state, funclm1);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 2:

        info2.fitexpr = f;
        info2.fitconstssize = consts_size;
        minlmoptimize(state, funclm2);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 3:

        info3.fitexpr = f;
        info3.fitconstssize = consts_size;
        minlmoptimize(state, funclm3);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 4:

        info4.fitexpr = f;
        info4.fitconstssize = consts_size;
        minlmoptimize(state, funclm4);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 5:

        info5.fitexpr = f;
        info5.fitconstssize = consts_size;
        minlmoptimize(state, funclm5);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 6:

        info6.fitexpr = f;
        info6.fitconstssize = consts_size;
        minlmoptimize(state, funclm6);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 7:

        info7.fitexpr = f;
        info7.fitconstssize = consts_size;
        minlmoptimize(state, funclm7);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    case 8:

        info8.fitexpr = f;
        info8.fitconstssize = consts_size;
        minlmoptimize(state, funclm8);
        minlmresults(state, cons, rep);
        //printf("%s\n", cons.tostring(consts.size()).c_str());
        for (int i = 0; i < consts_size; i++)
        {
            consst.push_back(cons[i]);
        }
        delete[] ss;
        delete[] constsx;
        return consst;
        break;
    }
    
    
    
}
/*void funclmjac(const real_1d_array& x, real_1d_array& fi, real_2d_array& jac, void* ptr)
{
    vector<double> consts;
    for (int i = 0; i < fitconstssize; i++)
    {
        consts.push_back(x[i]);
    }
    int it = m;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            
                fi[i * it + j] = fitexpr.res_in_point(fitx[i], fity[j], fitz[i][j], consts);
                for (int k = 0; k < consts.size(); k++)
                {
                    jac[i * it + j][k] = fitexpr.dif(k+1, 1).res_in_point(fitx[i], fity[j], fitz[i][j], consts);
                }
               
        }
    }
}*/
void funclm1(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts=new double[info1.fitconstssize];
    for (int i = 0; i < info1.fitconstssize; i++)
    {
        consts[i]=x[i];
    }
    double toch = 0;
    int it = m;
    double* znach=new double[n*m];
    info1.fitexpr.res(info1.fitexpr, info1.fitx, info1.fity, info1.fitz, consts, znach, info1.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm2(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info2.fitconstssize];
    for (int i = 0; i < info2.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info2.fitexpr.res(info2.fitexpr, info2.fitx, info2.fity, info2.fitz, consts, znach, info2.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm3(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info3.fitconstssize];
    for (int i = 0; i < info3.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info3.fitexpr.res(info3.fitexpr, info3.fitx, info3.fity, info3.fitz, consts, znach, info3.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm4(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info4.fitconstssize];
    for (int i = 0; i < info4.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info4.fitexpr.res(info4.fitexpr, info4.fitx, info4.fity, info4.fitz, consts, znach, info4.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm5(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info5.fitconstssize];
    for (int i = 0; i < info5.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info5.fitexpr.res(info5.fitexpr, info5.fitx, info5.fity, info5.fitz, consts, znach, info5.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm6(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info6.fitconstssize];
    for (int i = 0; i < info6.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info6.fitexpr.res(info6.fitexpr, info6.fitx, info6.fity, info6.fitz, consts, znach, info6.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm7(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info7.fitconstssize];
    for (int i = 0; i < info7.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info7.fitexpr.res(info7.fitexpr, info7.fitx, info7.fity, info7.fitz, consts, znach, info7.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void funclm8(const real_1d_array& x, real_1d_array& fi, void* ptr)
{
    double* consts = new double[info8.fitconstssize];
    for (int i = 0; i < info8.fitconstssize; i++)
    {
        consts[i] = x[i];
    }
    double toch = 0;
    int it = m;
    double* znach = new double[n * m];
    info8.fitexpr.res(info8.fitexpr, info8.fitx, info8.fity, info8.fitz, consts, znach, info8.command_queue);
    for (int i = 0; i < n * m; i++)
    {
        double temp = znach[i];
        fi[i] = temp;
    }

    delete[] consts;
    delete[] znach;
}
void sort(vector<expr>& exprs, int size)
{
    
        //Указатели в начало и в конец массива
        int i = 0;
        int j = size - 1;

        //Центральный элемент массива
        expr mid = exprs[size / 2];

        //Делим массив
        do {
            //Пробегаем элементы, ищем те, которые нужно перекинуть в другую часть
            //В левой части массива пропускаем(оставляем на месте) элементы, которые меньше центрального
            while (exprs[i].tochnost < mid.tochnost) {
                i++;
            }
            //В правой части пропускаем элементы, которые больше центрального
            while (exprs[j].tochnost > mid.tochnost) {
                j--;
            }

            //Меняем элементы местами
            if (i <= j) {
                expr tmp1 = exprs[i];
                expr tmp2 = exprs[j];
                exprs[i] = tmp2;
                exprs[j] = tmp1;
                i++;
                j--;
            }
        } while (i <= j);


        //Рекурсивные вызовы, если осталось, что сортировать
        if (j > 0) {
            //"Левый кусок"
            sort(exprs, j + 1);
        }
        if (i < size) {
            //"Прaвый кусок"
            vector<expr> newexpr;
            for (int k = i; k < size;k++)
            {
                newexpr.push_back(exprs[k]);
            }
            sort(newexpr, size - i);
            for (int k = i; k < size; k++)
            {
                expr temp = newexpr[k - i];
                exprs[k] = temp;
            }
        }
    
}
expr maxtochnost(expr* exprs, int size)
{ 
    int p=0;
    expr max;
    for (int i = 0; i < size; i++)
    {
        if (exprs[i].name != "0")
        {
            max = exprs[i];
            p = i;
            break;
        }
    }
    for (int i = p; i < size; i++)
    {
        if ((exprs[i].tochnost < max.tochnost)&& (exprs[i].name != "0"))
        {
            max = exprs[i];
            p = i;
        }
    }
    expr zero;
    zero.type = 12;
    zero.name = "0";
    zero.tochnost = 1000000000;
    zero.dlin = 1000000000;
    exprs[p] = zero;
    return max;
}
void pareto_sort(vector<expr>& exprs)
{
    int size1 = exprs.size();

    expr* copia = new expr[exprs.size()];
    for (int i = exprs.size()-1; i >= 0; i--)
    {
        copia[i] = exprs[i];
        exprs.erase(exprs.begin() + i);
    }
    int point=0;
    int flag = 1;
    while (flag==1)
    {
        flag = 0;
        expr max = maxtochnost(copia, size1);
        
        if(max.name=="0")
        {
            break;
        }
        vector<expr> front;
        if (max.name != "0")
        {
            front.push_back(max);
        }
        for (int i = 0; i < size1; i++)
        {
            if (copia[i].name != "0")
            {
                
                flag = 1;
                int j = 0;
                int flagcycle = 0;
                int front_size = front.size();
                while (j < front.size())
                {
                    if (front[j].name == "0")
                    {
                        front.erase(front.begin() + j);
                        break;
                    }
                    if ((copia[i].dlin == front[j].dlin) && (copia[i].tochnost == front[j].tochnost)&& (front[j].name != "0")&& (copia[i].name != "0"))
                    {
                        if ((copia[i].name == "0") && (front[j].name == "0"))
                        {
                            front.erase(front.begin() + j);
                            break;
                        }
                        front.insert(front.begin() + j, copia[i]);
                        expr zero;
                        zero.type = 12;
                        zero.name = "0";
                        zero.tochnost = 1000000000;
                        zero.dlin = 1000000000;
                        copia[i] = zero;
                    }
                    if ((((copia[i].dlin <= front[j].dlin) && (copia[i].tochnost < front[j].tochnost)) || ((copia[i].dlin < front[j].dlin) && (copia[i].tochnost <= front[j].tochnost)))&&(copia[i].name!="0"))
                    {
                        expr temp = copia[i];
                        if(temp.name!="0"){
                        front.push_back(temp);
                        copia[i] = front[j];
                        front.erase(front.begin() + j);
                        flagcycle = -1;
                        }
                    }
                    if (((copia[i].dlin >= front[j].dlin) && (copia[i].tochnost > front[j].tochnost)) || ((copia[i].dlin > front[j].dlin) && (copia[i].tochnost >= front[j].tochnost)))
                    {
                        flagcycle = 1;
                    }
                    j++;
                    front_size = front.size();
                }
                if (flagcycle == 0)
                {
                    if(copia[i].name != "0")
                    { 
                    expr temp = copia[i];
                    front.push_back(temp);
                    expr zero;
                    zero.type = 12;
                    zero.name = "0";
                    zero.tochnost = 1000000000;
                    zero.dlin = 1000000000;
                    copia[i] = zero;
                    }
                }
            }
        }
        int flag1 = 1;
        while(flag1)
        {
            flag1 = 0;
        for (int i = 0; i < front.size()-1; i++)
        {
            if (front[i].tochnost > front[i + 1].tochnost)
            {
                flag1 = 1;
                expr temp = front[i];
                front[i] = front[i + 1];
                front[i + 1] = temp;
            }
        }
        }
        
        for (int i = 0; i < front.size(); i++)
        {
            exprs.push_back(front[i]);
        }
        point += front.size(); 
        vector<expr> ().swap(front);
    }
    exprs.pop_back();
    delete[] copia;
}
double dpp = 225;
expr otvet[3];
vector<vector<double>> constss;
void render()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    double r;
    double g;
    double b;
    double tochkax;
    double tochkay;
    double nn = double(n) / 2;
    double mm = double(m) / 2;
    double rx = 1 / nn;
    double ry = 1 / mm;
    int flagx = 1;
    int flagy = 1;
    for (int i = 0; i < n; i++)
    {
        tochkax = i +1 ;

        for (int j = 0; j < m; j++)
        {
            tochkay = j+1 ;
            r = otvet[0].res_in_point(tochkax, tochkay, 1, otvet[0].consts)/dpp;
            g = otvet[1].res_in_point(tochkax, tochkay, 1, otvet[1].consts)/dpp;
            b = otvet[2].res_in_point(tochkax, tochkay, 1, otvet[2].consts)/dpp;
            
            tochkay -= 1;
            if (r < 0)
                r = 0;
            if (r > 1)
                r = 1;
            if (g < 0)
                g = 0;
            if (g > 1)
                g = 1;
            if (b < 0)
                b = 0;
            if (b > 1)
                b = 1;
            colors[i][j].r = r;
            colors[i][j].g = g;
            colors[i][j].b = b;
            glColor3f(r, g, b);
            glBegin(GL_QUADS);
            tochkax -= 1;
            //cout << tochkax << " " << tochkay << " ";
            //cout << r << " " << g << " " << b << " " << endl;
            glVertex2f((tochkax)/ nn - 1, (tochkay) / mm - 1);
            glVertex2f((tochkax) / nn - 1, (tochkay) / mm - 1 + ry * flagy);
            glVertex2f((tochkax) / nn - 1 + rx * flagx, (tochkay) / mm - 1 + ry * flagy);
            glVertex2f((tochkax) / nn - 1 + rx * flagx, (tochkay) / mm - 1);
            glEnd();
            r = 0.0;
            g = 0.0;
            b = 0.0;
            tochkax +=1;
            flagy = 1;
        }
        flagx = 1;
    }
    glutSwapBuffers();
    glFinish();
}
void render1()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    double r;
    double g;
    double b;
    double tochkax;
    double tochkay;
    double nn = double(n) / 2;
    double mm = double(m) / 2;
    double rx = 1 / nn;
    double ry = 1 / mm;
    int flagx = 1;
    int flagy = 1;
    for (int i = 0; i < n; i++)
    {
        tochkax = i / nn;

        for (int j = 0; j < m; j++)
        {
            tochkay = j / mm;
            r = orig[i][j].r;
            g = orig[i][j].g;
            b = orig[i][j].b;
            //cout << r << " " << g << " " << b << endl;
            tochkay -= 1;
            glColor3f(r, g, b);
            glBegin(GL_QUADS);
            glVertex2f(tochkax-1, tochkay);
            glVertex2f(tochkax-1, tochkay + ry * flagy);
            glVertex2f(tochkax-1 + rx * flagx, tochkay + ry * flagy);
            glVertex2f(tochkax-1 + rx * flagx, tochkay);
            glEnd();
            r = 0.0;
            g = 0.0;
            b = 0.0;

            flagy = 1;
            
        }
        flagx = 1;
    }
    glutSwapBuffers();
    glFinish();
}

void render2()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glRotatef(5.0, 1.0, 1.0, 0.0);
    glLineWidth(10.0);
    glPushMatrix();
    glRotatef(180.0, 1.0, 1.0, 0.0);
    glScalef(0.5, 0.5, 0.5);
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 1; j < m - 1; j++)
        {

            glBegin(GL_LINES);
            glBegin(GL_LINES);
            glColor3f(0.0f, 0.0f, 0.0f);//black
            glVertex3f(bx[i], by[j], orig[i][j].r);
            glVertex3f(bx[i], by[j + 1], orig[i][j + 1].r);
            glVertex3f(bx[i], by[j], orig[i][j].r);
            glVertex3f(bx[i + 1], by[j], orig[i + 1][j].r);
            glVertex3f(bx[i], by[j], orig[i][j].r);
            glVertex3f(bx[i], by[j - 1], orig[i][j - 1].r );
            glVertex3f(bx[i], by[j], orig[i][j].r );
            glVertex3f(bx[i - 1], by[j], orig[i - 1][j].r );
            glEnd();
            glBegin(GL_LINES);
            glColor3f(1.0f, 0.0f, 0.0f);

            glVertex3f(bx[i], by[j], colors[i][j].r );
            glVertex3f(bx[i], by[j + 1], colors[i][j + 1].r );
            glVertex3f(bx[i], by[j], colors[i][j].r );
            glVertex3f(bx[i + 1], by[j], colors[i + 1][j].r );

            glVertex3f(bx[i], by[j], colors[i][j].r );
            glVertex3f(bx[i], by[j - 1], colors[i][j - 1].r );
            glVertex3f(bx[i], by[j], colors[i][j].r );
            glVertex3f(bx[i - 1], by[j], colors[i - 1][j].r );
            glEnd();
        }
    }
    for (int i = 1; i < n - 1; i++)
    {

        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(bx[i], by[0], orig[i][0].r );
        glVertex3f(bx[i - 1], by[0], orig[i - 1][0].r );
        glVertex3f(bx[i], by[0], orig[i][0].r );
        glVertex3f(bx[i + 1], by[0], orig[i + 1][0].r );
        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].r );
        glVertex3f(bx[i + 1], by[n - 1], orig[i + 1][n - 1].r );
        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].r );
        glVertex3f(bx[i - 1], by[n - 1], orig[i - 1][n - 1].r );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(bx[i], by[0], colors[i][0].r );
        glVertex3f(bx[i - 1], by[0], colors[i - 1][0].r );
        glVertex3f(bx[i], by[0], colors[i][0].r );
        glVertex3f(bx[i + 1], by[0], colors[i + 1][0].r );

        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].r );
        glVertex3f(bx[i + 1], by[n - 1], colors[i + 1][n - 1].r );
        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].r );
        glVertex3f(bx[i - 1], by[n - 1], colors[i - 1][n - 1].r );

        glEnd();
    }
    for (int i = 1; i < m - 1; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(bx[0], by[i], orig[0][i].r );
        glVertex3f(bx[0], by[i - 1], orig[0][i - 1].r );
        glVertex3f(bx[0], by[i], orig[0][i].r );
        glVertex3f(bx[0], by[i + 1], orig[0][i + 1].r );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].r );
        glVertex3f(bx[n - 1], by[i - 1], orig[n - 1][i - 1].r );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].r );
        glVertex3f(bx[n - 1], by[i + 1], orig[n - 1][i + 1].r );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);

        glVertex3f(bx[0], by[i], colors[0][i].r );
        glVertex3f(bx[0], by[i - 1], colors[0][i - 1].r );
        glVertex3f(bx[0], by[i], colors[0][i].r );
        glVertex3f(bx[0], by[i + 1], colors[0][i + 1].r );

        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].r );
        glVertex3f(bx[n - 1], by[i - 1], colors[n - 1][i - 1].r );
        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].r );
        glVertex3f(bx[n - 1], by[i + 1], colors[n - 1][i + 1].r );
        glEnd();
    }
    glPopMatrix();
    glutSwapBuffers();
    glFinish();
}

void render3()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glRotatef(5.0, 1.0, 1.0, 0.0);
    glPushMatrix();
    glRotatef(180.0, 1.0, 1.0, 0.0);
    glScalef(0.5, 0.5, 0.5);
    glLineWidth(10.0);
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 1; j < m - 1; j++)
        {
            glBegin(GL_LINES);
            glColor3f(0.0f, 0.0f, 0.0f);//black
            glVertex3f(bx[i], by[j], orig[i][j].g );
            glVertex3f(bx[i], by[j + 1], orig[i][j + 1].g );
            glVertex3f(bx[i], by[j], orig[i][j].g );
            glVertex3f(bx[i + 1], by[j], orig[i + 1][j].g );
            glVertex3f(bx[i], by[j], orig[i][j].g );
            glVertex3f(bx[i], by[j - 1], orig[i][j - 1].g );
            glVertex3f(bx[i], by[j], orig[i][j].g );
            glVertex3f(bx[i - 1], by[j], orig[i - 1][j].g );
            glEnd();
            glBegin(GL_LINES);
            glColor3f(0.0f, 1.0f, 0.0f);

            glVertex3f(bx[i], by[j], colors[i][j].g );
            glVertex3f(bx[i], by[j + 1], colors[i][j + 1].g );
            glVertex3f(bx[i], by[j], colors[i][j].g );
            glVertex3f(bx[i + 1], by[j], colors[i + 1][j].g );

            glVertex3f(bx[i], by[j], colors[i][j].g );
            glVertex3f(bx[i], by[j - 1], colors[i][j - 1].g );
            glVertex3f(bx[i], by[j], colors[i][j].g );
            glVertex3f(bx[i - 1], by[j], colors[i - 1][j].g );
            glEnd();
        }
    }
    for (int i = 1; i < n - 1; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(bx[i], by[0], orig[i][0].g );
        glVertex3f(bx[i - 1], by[0], orig[i - 1][0].g );
        glVertex3f(bx[i], by[0], orig[i][0].g );
        glVertex3f(bx[i + 1], by[0], orig[i + 1][0].g );
        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].g );
        glVertex3f(bx[i + 1], by[n - 1], orig[i + 1][n - 1].g );
        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].g );
        glVertex3f(bx[i - 1], by[n - 1], orig[i - 1][n - 1].g );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(bx[i], by[0], colors[i][0].g );
        glVertex3f(bx[i - 1], by[0], colors[i - 1][0].g );
        glVertex3f(bx[i], by[0], colors[i][0].g );
        glVertex3f(bx[i + 1], by[0], colors[i + 1][0].g );

        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].g );
        glVertex3f(bx[i + 1], by[n - 1], colors[i + 1][n - 1].g );
        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].g );
        glVertex3f(bx[i - 1], by[n - 1], colors[i - 1][n - 1].g );
        glEnd();
    }
    for (int i = 1; i < m - 1; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);

        glVertex3f(bx[0], by[i], orig[0][i].g );
        glVertex3f(bx[0], by[i - 1], orig[0][i - 1].g );
        glVertex3f(bx[0], by[i], orig[0][i].g );
        glVertex3f(bx[0], by[i + 1], orig[0][i + 1].g );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].g );
        glVertex3f(bx[n - 1], by[i - 1], orig[n - 1][i - 1].g );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].g );
        glVertex3f(bx[n - 1], by[i + 1], orig[n - 1][i + 1].g );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);

        glVertex3f(bx[0], by[i], colors[0][i].g );
        glVertex3f(bx[0], by[i - 1], colors[0][i - 1].g );
        glVertex3f(bx[0], by[i], colors[0][i].g );
        glVertex3f(bx[0], by[i + 1], colors[0][i + 1].g );

        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].g );
        glVertex3f(bx[n - 1], by[i - 1], colors[n - 1][i - 1].g );
        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].g );
        glVertex3f(bx[n - 1], by[i + 1], colors[n - 1][i + 1].g );
        glEnd();
    }
    glPopMatrix();
    glutSwapBuffers();
    glFinish();
}

void render4()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glRotatef(5.0, 1.0, 1.0, 0.0);
    glPushMatrix();
    glRotatef(180.0, 1.0, 1.0, 0.0);
    glScalef(0.5, 0.5, 0.5);
    glLineWidth(10.0);
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 1; j < m - 1; j++)
        {
            glBegin(GL_LINES);
            glColor3f(0.0f, 0.0f, 0.0f);//black
            glVertex3f(bx[i], by[j], orig[i][j].b / dpp);
            glVertex3f(bx[i], by[j + 1], orig[i][j + 1].b );
            glVertex3f(bx[i], by[j], orig[i][j].b );
            glVertex3f(bx[i + 1], by[j], orig[i + 1][j].b );
            glVertex3f(bx[i], by[j], orig[i][j].b );
            glVertex3f(bx[i], by[j - 1], orig[i][j - 1].b );
            glVertex3f(bx[i], by[j], orig[i][j].b );
            glVertex3f(bx[i - 1], by[j], orig[i - 1][j].b );
            glEnd();
            glBegin(GL_LINES);
            glColor3f(0.0f, 0.0f, 1.0f);

            glVertex3f(bx[i], by[j], colors[i][j].b );
            glVertex3f(bx[i], by[j + 1], colors[i][j + 1].b );
            glVertex3f(bx[i], by[j], colors[i][j].b );
            glVertex3f(bx[i + 1], by[j], colors[i + 1][j].b );

            glVertex3f(bx[i], by[j], colors[i][j].b );
            glVertex3f(bx[i], by[j - 1], colors[i][j - 1].b );
            glVertex3f(bx[i], by[j], colors[i][j].b );
            glVertex3f(bx[i - 1], by[j], colors[i - 1][j].b );
            glEnd();
        }
    }
    for (int i = 1; i < n - 1; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(bx[i], by[0], orig[i][0].b );
        glVertex3f(bx[i - 1], by[0], orig[i - 1][0].b );
        glVertex3f(bx[i], by[0], orig[i][0].b );
        glVertex3f(bx[i + 1], by[0], orig[i + 1][0].b );

        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].b );
        glVertex3f(bx[i + 1], by[n - 1], orig[i + 1][n - 1].b );
        glVertex3f(bx[i], by[n - 1], orig[i][n - 1].b );
        glVertex3f(bx[i - 1], by[n - 1], orig[i - 1][n - 1].b );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(bx[i], by[0], colors[i][0].b );
        glVertex3f(bx[i - 1], by[0], colors[i - 1][0].b );
        glVertex3f(bx[i], by[0], colors[i][0].b );
        glVertex3f(bx[i + 1], by[0], colors[i + 1][0].b );

        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].b );
        glVertex3f(bx[i + 1], by[n - 1], colors[i + 1][n - 1].b );
        glVertex3f(bx[i], by[n - 1], colors[i][n - 1].b );
        glVertex3f(bx[i - 1], by[n - 1], colors[i - 1][n - 1].b );

        glEnd();
    }
    for (int i = 1; i < m - 1; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(bx[0], by[i], orig[0][i].b );
        glVertex3f(bx[0], by[i - 1], orig[0][i - 1].b );
        glVertex3f(bx[0], by[i], orig[0][i].b );
        glVertex3f(bx[0], by[i + 1], orig[0][i + 1].b );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].b );
        glVertex3f(bx[n - 1], by[i - 1], orig[n - 1][i - 1].b );
        glVertex3f(bx[n - 1], by[i], orig[n - 1][i].b );
        glVertex3f(bx[n - 1], by[i + 1], orig[n - 1][i + 1].b );
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);

        glVertex3f(bx[0], by[i], colors[0][i].b );
        glVertex3f(bx[0], by[i - 1], colors[0][i - 1].b );
        glVertex3f(bx[0], by[i], colors[0][i].b );
        glVertex3f(bx[0], by[i + 1], colors[0][i + 1].b );

        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].b );
        glVertex3f(bx[n - 1], by[i - 1], colors[n - 1][i - 1].b );
        glVertex3f(bx[n - 1], by[i], colors[n - 1][i].b );
        glVertex3f(bx[n - 1], by[i + 1], colors[n - 1][i + 1].b );
        glEnd();
    }
    glPopMatrix();
    glutSwapBuffers();
    glFinish();
}





void createbasis(vector<expr>& exprs, Node & root)
{
    for (int i = 0; i < 2; i++)
    {
        expr var;
        var.uniqconst = 0;
        var.kolvoparam = 0;
        if (i == 0)
        {
            var.type = 13;
            var.name = op[var.type];
        }
        if (i == 1)
        {
            var.type = 14;
            var.name = op[var.type];
        }
        
        expr mul;
        mul.type = 2;
        mul.name = op[mul.type];
        mul.kolvoparam = 2;
        mul.params.push_back(createconst(0));
        mul.params.push_back(var);
        mul.uniqconst = 1;
        
        for (int j = 1; j < 12; j++)
        {
            expr ex; 
            expr sum;
            sum.type = 1;
            sum.name = op[sum.type];
            sum.kolvoparam = 2;
            sum.uniqconst = 2;
            int ok = 0;
            switch (j)
            { 
                case 1:
                    ex.type = j;
                    ex.uniqconst = 0;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 2;
                    ex.uniqconst += mul.uniqconst;
                    ex.params.push_back(createconst(ex.uniqconst));
                    ex.params.push_back(mul);
                    ex.uniqconst++;
                    ok = 1;
                    exprs.push_back(ex);
                    break;
                case 2:
                    ok = 1;
                    exprs.push_back(mul);
                    break;
                case 3:
                    ok = 1;
                    ex.uniqconst = 0;
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    for (int i = 0; i < ex.params.size(); i++)
                        ex.uniqconst += ex.params[i].uniqconst;
                    exprs.push_back(ex);
                    break;
                case 4:
                    ex.uniqconst = 0;
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.type = j;
                    ex.kolvoparam = 2;
                    ex.params.push_back(createconst(ex.uniqconst));
                    ex.params.push_back(var);
                    for (int i = 0; i < ex.params.size(); i++)
                        ex.uniqconst += ex.params[i].uniqconst;
                    sum.params.push_back(ex);
                    break;
                case 5:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    sum.params.push_back(ex);
                    break;
                case 6:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    ex.uniqconst = 1;
                    sum.params.push_back(ex);
                    break;
                case 7:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    ex.uniqconst = 1; 
                    sum.params.push_back(ex);
                    break;
                case 8:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    ex.uniqconst = 1;
                    sum.params.push_back(ex);
                    break;
                case 9:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    ex.uniqconst = 1;
                    sum.params.push_back(ex);
                    break;
                case 10:
                    ex.type = j;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 1;
                    ex.params.push_back(mul);
                    ex.uniqconst = 1;
                    sum.params.push_back(ex);
                    break;
                case 11:
                    ex.type = 2;
                    ex.name = op[ex.type];
                    ex.kolvoparam = 2;
                    ex.params.push_back(createconst(0));
                    ex.uniqconst = 2;
                    expr ex1;
                    ex1.type=j;
                    ex1.name = op[ex1.type];
                    ex1.uniqconst = 1;
                    ex1.kolvoparam = 2;
                    ex1.params.push_back(var);
                    ex1.params.push_back(createconst(ex1.uniqconst+1));
                    ex.params.push_back(ex1);
                    sum.params.push_back(ex);
                    break;
            }
            if (ok == 0)
            {
                sum.params.insert(sum.params.cbegin(),createconst(1));
                vector<int> co;
                sum.uniqconst = sum.kolvouniqconsts(co);

                exprs.push_back(sum);
            }

        }
    }
        for (int i = 0; i < exprs.size(); i++)
        {
            vector<int> ex;
            exprs[i].makeint(ex);
            root.insert(ex);
        }
}
void func_thread(thread_inform info, int i,int j)
{
    info.fitx = x;
    info.fity = y;
    if(j==1)
    info.fitz = z1;
    if (j == 2)
        info.fitz = z2;
    if (j == 3)
        info.fitz = z3;
    info.nomer = i;
    for (int st = 0; st < info.kolvo_na_1; st++)
    {
        vector<double> cons;
        cons=temporal[st + info.vhod].fit(i, info.command_queue, info.fitz);
        
    }
}

// bit extract
unsigned char bitextract(const unsigned int byte, const unsigned int mask);
int main(int argc, char* argv[]) 
{
    string fileName3;
    cout << "Input image's address:";
    cin >> fileName3;
    const char* fileName = fileName3.c_str();
    // открываем файл
    std::ifstream fileStream(fileName, std::ifstream::binary);
    if (!fileStream) {
        std::cout << "Error opening file '" << fileName << "'." << std::endl;
        return 0;
    }

    // заголовк изображения
    BITMAPFILEHEADER fileHeader;
    read(fileStream, fileHeader.bfType, sizeof(fileHeader.bfType));
    read(fileStream, fileHeader.bfSize, sizeof(fileHeader.bfSize));
    read(fileStream, fileHeader.bfReserved1, sizeof(fileHeader.bfReserved1));
    read(fileStream, fileHeader.bfReserved2, sizeof(fileHeader.bfReserved2));
    read(fileStream, fileHeader.bfOffBits, sizeof(fileHeader.bfOffBits));

    if (fileHeader.bfType != 0x4D42) {
        std::cout << "Error: '" << fileName << "' is not BMP file." << std::endl;
        return 0;
    }

    // информация изображения
    BITMAPINFOHEADER1 fileInfoHeader;
    read(fileStream, fileInfoHeader.biSize, sizeof(fileInfoHeader.biSize));

    // bmp core
    if (fileInfoHeader.biSize >= 12) {
        read(fileStream, fileInfoHeader.biWidth, sizeof(fileInfoHeader.biWidth));
        read(fileStream, fileInfoHeader.biHeight, sizeof(fileInfoHeader.biHeight));
        read(fileStream, fileInfoHeader.biPlanes, sizeof(fileInfoHeader.biPlanes));
        read(fileStream, fileInfoHeader.biBitCount, sizeof(fileInfoHeader.biBitCount));
    }

    // получаем информацию о битности
    int colorsCount = fileInfoHeader.biBitCount >> 3;
    if (colorsCount < 3) {
        colorsCount = 3;
    }

    int bitsOnColor = fileInfoHeader.biBitCount / colorsCount;
    int maskValue = (1 << bitsOnColor) - 1;

    // bmp v1
    if (fileInfoHeader.biSize >= 40) {
        read(fileStream, fileInfoHeader.biCompression, sizeof(fileInfoHeader.biCompression));
        read(fileStream, fileInfoHeader.biSizeImage, sizeof(fileInfoHeader.biSizeImage));
        read(fileStream, fileInfoHeader.biXPelsPerMeter, sizeof(fileInfoHeader.biXPelsPerMeter));
        read(fileStream, fileInfoHeader.biYPelsPerMeter, sizeof(fileInfoHeader.biYPelsPerMeter));
        read(fileStream, fileInfoHeader.biClrUsed, sizeof(fileInfoHeader.biClrUsed));
        read(fileStream, fileInfoHeader.biClrImportant, sizeof(fileInfoHeader.biClrImportant));
    }

    // bmp v2
    fileInfoHeader.biRedMask = 0;
    fileInfoHeader.biGreenMask = 0;
    fileInfoHeader.biBlueMask = 0;

    if (fileInfoHeader.biSize >= 52) {
        read(fileStream, fileInfoHeader.biRedMask, sizeof(fileInfoHeader.biRedMask));
        read(fileStream, fileInfoHeader.biGreenMask, sizeof(fileInfoHeader.biGreenMask));
        read(fileStream, fileInfoHeader.biBlueMask, sizeof(fileInfoHeader.biBlueMask));
    }

    // если маска не задана, то ставим маску по умолчанию
    if (fileInfoHeader.biRedMask == 0 || fileInfoHeader.biGreenMask == 0 || fileInfoHeader.biBlueMask == 0) {
        fileInfoHeader.biRedMask = maskValue << (bitsOnColor * 2);
        fileInfoHeader.biGreenMask = maskValue << bitsOnColor;
        fileInfoHeader.biBlueMask = maskValue;
    }

    // bmp v3
    if (fileInfoHeader.biSize >= 56) {
        read(fileStream, fileInfoHeader.biAlphaMask, sizeof(fileInfoHeader.biAlphaMask));
    }
    else {
        fileInfoHeader.biAlphaMask = maskValue << (bitsOnColor * 3);
    }

    // bmp v4
    if (fileInfoHeader.biSize >= 108) {
        read(fileStream, fileInfoHeader.biCSType, sizeof(fileInfoHeader.biCSType));
        read(fileStream, fileInfoHeader.biEndpoints, sizeof(fileInfoHeader.biEndpoints));
        read(fileStream, fileInfoHeader.biGammaRed, sizeof(fileInfoHeader.biGammaRed));
        read(fileStream, fileInfoHeader.biGammaGreen, sizeof(fileInfoHeader.biGammaGreen));
        read(fileStream, fileInfoHeader.biGammaBlue, sizeof(fileInfoHeader.biGammaBlue));
    }

    // bmp v5
    if (fileInfoHeader.biSize >= 124) {
        read(fileStream, fileInfoHeader.biIntent, sizeof(fileInfoHeader.biIntent));
        read(fileStream, fileInfoHeader.biProfileData, sizeof(fileInfoHeader.biProfileData));
        read(fileStream, fileInfoHeader.biProfileSize, sizeof(fileInfoHeader.biProfileSize));
        read(fileStream, fileInfoHeader.biReserved, sizeof(fileInfoHeader.biReserved));
    }

    // проверка на поддерку этой версии формата
    if (fileInfoHeader.biSize != 12 && fileInfoHeader.biSize != 40 && fileInfoHeader.biSize != 52 &&
        fileInfoHeader.biSize != 56 && fileInfoHeader.biSize != 108 && fileInfoHeader.biSize != 124) {
        std::cout << "Error: Unsupported BMP format." << std::endl;
        return 0;
    }

    if (fileInfoHeader.biBitCount != 16 && fileInfoHeader.biBitCount != 24 && fileInfoHeader.biBitCount != 32) {
        std::cout << "Error: Unsupported BMP bit count." << std::endl;
        return 0;
    }

    if (fileInfoHeader.biCompression != 0 && fileInfoHeader.biCompression != 3) {
        std::cout << "Error: Unsupported BMP compression." << std::endl;
        return 0;
    }

    // rgb info
    RGBQUAD** rgbInfo = new RGBQUAD * [fileInfoHeader.biHeight];

    for (unsigned int i = 0; i < fileInfoHeader.biHeight; i++) {
        rgbInfo[i] = new RGBQUAD[fileInfoHeader.biWidth];
    }

    // определение размера отступа в конце каждой строки
    int linePadding = ((fileInfoHeader.biWidth * (fileInfoHeader.biBitCount / 8)) % 4) & 3;

    // чтение
    unsigned int bufer;

    for (unsigned int i = 0; i < fileInfoHeader.biHeight; i++) {
        for (unsigned int j = 0; j < fileInfoHeader.biWidth; j++) {
            read(fileStream, bufer, fileInfoHeader.biBitCount / 8);

            rgbInfo[i][j].rgbRed = bitextract(bufer, fileInfoHeader.biRedMask);
            rgbInfo[i][j].rgbGreen = bitextract(bufer, fileInfoHeader.biGreenMask);
            rgbInfo[i][j].rgbBlue = bitextract(bufer, fileInfoHeader.biBlueMask);
            rgbInfo[i][j].rgbReserved = bitextract(bufer, fileInfoHeader.biAlphaMask);
        }
        fileStream.seekg(linePadding, std::ios_base::cur);
    }
    n = fileInfoHeader.biHeight;
    m = fileInfoHeader.biWidth;
    n = 5;
    m = 5;
    Node root1;
    Node root2;
    Node root3;
    int vz = 5;
    root1.type = -1;
    root2.type = -1;
    root3.type = -1;
    orig = new pixel * [n];
    colors = new pixel * [n];
    bx = new double[n];
    by = new double[m];
    for (int i = 0; i < n; i++)
    {
        orig[i] = new pixel[m];
        colors[i] = new pixel[m];
    }
    createbasis(expressions1, root1);
    createbasis(expressions2, root2);
    createbasis(expressions3, root3);
     x = new double[n];
     y = new double[n];
     for (double y1 = 1; y1 <= m; y1++)
     {
         y[(int)y1 - 1] = y1;

     }
     for (double x1 = 1; x1 <= n; x1++)
     {
         x[(int)x1 - 1] = x1;
     }
     z1 = new double* [n];
     z2 = new double* [n];
     z3 = new double* [n];
    double shagx = (double)2 / (n);
    double shagy = (double)2 / (m);
    double** qwerty;
    qwerty = new double* [n];
    for (unsigned int i = 0; i < n; i++) {
        z1[i] = new double[m];
        z2[i] = new double[m];
        z3[i] = new double[m];
        for (unsigned int j = 0; j < m; j++) 
        {
            z1[i][j] = +rgbInfo[i][j].rgbRed;
            z2[i][j] = +rgbInfo[i][j].rgbGreen;
            z3[i][j] = +rgbInfo[i][j].rgbBlue;
            z1[i][j] = exp(x[i]*y[j]);
            if (z1[i][j] > 255)
                z1[i][j] = 255;
            if (z1[i][j] < 0)
                z1[i][j] = 0;
            z2[i][j] = x[i] * y[j];
            if (z2[i][j] > 255)
                z2[i][j] = 255;
            if (z2[i][j] < 0)
                z2[i][j] = 0;
            z3[i][j] = sin(x[i]*y[j]) ;
            if (z3[i][j] > 255)
                z3[i][j] = 255;
            if (z3[i][j] < 0)
                z3[i][j] = 0;
        }
    }
    

    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    
    /* получить доступные устройства */
    
    FILE* fp;
    const char fileName1[] = "./func.cl";
    size_t source_size;
    char* source_str;
    int i;

    fp = fopen(fileName1, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);


    /* создать бинарник из кода программы */
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
   
    /* скомпилировать программу */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
   
    expr otv;
    int flag = 0;
    int flagkolvo = 0;
    expr best=expressions1[0];
    temporal = expressions1;
    best = temporal[0];
    while ((flag == 0) && (flagkolvo < 11))
    {
        cout << temporal.size() << endl;
        int kolvo_na_1[8];
        int vhod[8];
        for (int i = 0; i < 8; i++)
        {
            kolvo_na_1[i] = temporal.size() / 8;
        }
        vhod[0] = 0;
        for (int i = (temporal.size() / 8) * 8; i < temporal.size(); i++)
        {
            kolvo_na_1[i - (temporal.size() / 8) * 8] += 1;
        }
        for (int i = 1; i < 8; i++)
        {
            vhod[i] = vhod[i - 1] + kolvo_na_1[i - 1];
        }
        info1.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info1.kolvo_na_1 = kolvo_na_1[0];
        info1.vhod = vhod[0];
        info1.fitx = x;
        info1.fity = y;
        info1.fitz = z1;

        info2.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info2.kolvo_na_1 = kolvo_na_1[1];
        info2.vhod = vhod[1];
        info2.fitx = x;
        info2.fity = y;
        info2.fitz = z1;

        info3.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info3.kolvo_na_1 = kolvo_na_1[2];
        info3.vhod = vhod[2];
        info3.fitx = x;
        info3.fity = y;
        info3.fitz = z1;

        info4.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info4.kolvo_na_1 = kolvo_na_1[3];
        info4.vhod = vhod[3];
        info4.fitx = x;
        info4.fity = y;
        info4.fitz = z1;

        info5.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info5.kolvo_na_1 = kolvo_na_1[4];
        info5.vhod = vhod[4];
        info5.fitx = x;
        info5.fity = y;
        info5.fitz = z1;

        info6.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info6.kolvo_na_1 = kolvo_na_1[5];
        info6.vhod = vhod[5];
        info6.fitx = x;
        info6.fity = y;
        info6.fitz = z1;

        info7.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info7.kolvo_na_1 = kolvo_na_1[6];
        info7.vhod = vhod[6];
        info7.fitx = x;
        info7.fity = y;
        info7.fitz = z1;

        info8.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info8.kolvo_na_1 = kolvo_na_1[7];
        info8.vhod = vhod[7];
        info8.fitx = x;
        info8.fity = y;
        info8.fitz = z1;

        thread th1(func_thread, info1, 1, 1);
        thread th2(func_thread, info2, 2, 1);
        thread th3(func_thread, info3, 3, 1);
        thread th4(func_thread, info4, 4, 1);
        thread th5(func_thread, info5, 5, 1);
        thread th6(func_thread, info6, 6, 1);
        thread th7(func_thread, info7, 7, 1);
        thread th8(func_thread, info8, 8, 1);
        th1.join();
        th2.join();
        th3.join();
        th4.join();
        th5.join();
        th6.join();
        th7.join();
        th8.join();
        if(flagkolvo!=0){
        for (int t = 0; t < temporal.size(); t++)
        {
            expr temp = temporal[t];
            expressions1.push_back(temp);
        }
        }
        else
        {
            for (int t = 0; t < temporal.size(); t++)
            {
                expr temp = temporal[t];
                expressions1[t]=temp;
            }
        }
        clReleaseCommandQueue(info1.command_queue);
        clReleaseCommandQueue(info2.command_queue);
        clReleaseCommandQueue(info3.command_queue);
        clReleaseCommandQueue(info4.command_queue);
        clReleaseCommandQueue(info5.command_queue);
        clReleaseCommandQueue(info6.command_queue);
        clReleaseCommandQueue(info7.command_queue);
        clReleaseCommandQueue(info8.command_queue);
        temporal.clear();
        temporal.resize(0);
        sort(expressions1, expressions1.size());
        if (expressions1[0].tochnost < tochn)
        {
            best = expressions1[0];
            flag = 1;
            break;
        }
        cout << "expressions1 tochnost na pokolenii " << flagkolvo << "=";
        cout << expressions1[0].tochnost << endl;
        expressions1[0].print();
        cout << endl;
        cout << "best=";
        cout << best.tochnost<<endl;
        best.print();
        cout << endl;
        for (int i = 0; i < population; i++)
        {
            expressions1[i].print();
            cout << endl;
            cout << expressions1[i].tochnost << endl;
        }
        if (expressions1[0].tochnost < best.tochnost)
            best = expressions1[0];
        vector<expr> pop;
        for (int i = population-1; i >= 0; i--)
        {
            pop=expressions1[i].exten(root1);
            expressions1.erase(expressions1.begin() + i);
            for (int j = 0; j < pop.size(); j++)
            {
                pop[j].optimization();
                
                vector<int> num;
                pop[j].makeint(num);
                    if (root1.insert(num)) {
                        temporal.push_back(pop[j]);
                    }
                
            }
        }
        flagkolvo++;
    }
    cout << "vot ono 1" << endl;
    cout << best.tochnost << endl;
    best.print();
    cout << endl;
    for (int q = 0; q <best.consts.size(); q++)
        cout << best.consts[q] << " ";
    cout << endl;
    //cout << flagkolvo << endl;
    otv = best;
    constss.push_back(best.consts);
    
    otvet[0] = otv;
    flag = 0;
    flagkolvo = 0;
    tochn = 2;
    temporal = expressions2;
    best = expressions2[0];
    while ((flag == 0) && (flagkolvo < 11))
    {
        cout << temporal.size() << endl;
        int kolvo_na_1[8];
        int vhod[8];
        for (int i = 0; i < 8; i++)
        {
            kolvo_na_1[i] = temporal.size() / 8;
        }
        vhod[0] = 0;
        for (int i = (temporal.size() / 8) * 8; i < temporal.size(); i++)
        {
            kolvo_na_1[i - (temporal.size() / 8) * 8] += 1;
        }
        for (int i = 1; i < 8; i++)
        {
            vhod[i] = vhod[i - 1] + kolvo_na_1[i - 1];
        }
        info1.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info1.kolvo_na_1 = kolvo_na_1[0];
        info1.vhod = vhod[0];
        info1.fitx = x;
        info1.fity = y;
        info1.fitz = z2;

        info2.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info2.kolvo_na_1 = kolvo_na_1[1];
        info2.vhod = vhod[1];
        info2.fitx = x;
        info2.fity = y;
        info2.fitz = z2;

        info3.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info3.kolvo_na_1 = kolvo_na_1[2];
        info3.vhod = vhod[2];
        info3.fitx = x;
        info3.fity = y;
        info3.fitz = z2;

        info4.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info4.kolvo_na_1 = kolvo_na_1[3];
        info4.vhod = vhod[3];
        info4.fitx = x;
        info4.fity = y;
        info4.fitz = z2;

        info5.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info5.kolvo_na_1 = kolvo_na_1[4];
        info5.vhod = vhod[4];
        info5.fitx = x;
        info5.fity = y;
        info5.fitz = z2;

        info6.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info6.kolvo_na_1 = kolvo_na_1[5];
        info6.vhod = vhod[5];
        info6.fitx = x;
        info6.fity = y;
        info6.fitz = z2;

        info7.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info7.kolvo_na_1 = kolvo_na_1[6];
        info7.vhod = vhod[6];
        info7.fitx = x;
        info7.fity = y;
        info7.fitz = z2;

        info8.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info8.kolvo_na_1 = kolvo_na_1[7];
        info8.vhod = vhod[7];
        info8.fitx = x;
        info8.fity = y;
        info8.fitz = z2;

        thread th1(func_thread, info1, 1,2);
        thread th2(func_thread, info2, 2,2);
        thread th3(func_thread, info3, 3,2);
        thread th4(func_thread, info4, 4,2);
        thread th5(func_thread, info5, 5,2);
        thread th6(func_thread, info6, 6,2);
        thread th7(func_thread, info7, 7,2);
        thread th8(func_thread, info8, 8,2);
        th1.join();
        th2.join();
        th3.join();
        th4.join();
        th5.join();
        th6.join();
        th7.join();
        th8.join();
        if (flagkolvo != 0) {
            for (int t = 0; t < temporal.size(); t++)
            {
                expr temp = temporal[t];
                expressions2.push_back(temp);
            }
        }
        else
        {
            for (int t = 0; t < temporal.size(); t++)
            {
                expr temp = temporal[t];
                expressions2[t]=temp;
            }
        }
        clReleaseCommandQueue(info1.command_queue);
        clReleaseCommandQueue(info2.command_queue);
        clReleaseCommandQueue(info3.command_queue);
        clReleaseCommandQueue(info4.command_queue);
        clReleaseCommandQueue(info5.command_queue);
        clReleaseCommandQueue(info6.command_queue);
        clReleaseCommandQueue(info7.command_queue);
        clReleaseCommandQueue(info8.command_queue);
        pareto_sort(expressions2);
        temporal.clear();
        temporal.resize(0);
        if (expressions2[0].tochnost < tochn)
        {
                best = expressions2[0];
            flag = 1;
            break;
        }
        if (expressions2[0].tochnost < best.tochnost)
            best = expressions2[0];
        cout << "expressions2 tochnost na pokolenii " << flagkolvo << "=";
        cout << expressions2[0].tochnost << endl;
        expressions2[0].print();
        cout << endl;
        cout << "best2=";
        cout << best.tochnost << endl;
        best.print();
        cout << endl;
        vector<expr> pop;
        for (int i = population-1; i >=0; i--)
        {
            pop = expressions2[i].exten(root2);
            expressions2.erase(expressions2.begin() + i);
            for (int j = 0; j < pop.size(); j++)
            {
                pop[j].optimization();
                vector<int> num;
                pop[j].makeint(num);
                if (root2.insert(num)) {
                    temporal.push_back(pop[j]);
                }
            }
        }
        flagkolvo++;
    }
    cout << "vot ono 2" << endl;
    best.print();
    cout << endl;
    cout << best.dlin << " " << best.tochnost << endl;
    for (int q = 0; q < best.consts.size(); q++)
        cout << best.consts[q] << " ";
    cout << endl;
    cout << flagkolvo << endl;
    
    /*for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << z2[i][j] << endl;
            cout << best.res_in_point(x[i], y[j], 1, best.consts) << endl;
            system("pause");
        }
    }*/
    otv = best;
    constss.push_back(best.consts);
    otvet[1] = otv;
    flag = 0;
    flagkolvo = 0;
    temporal = expressions3;
    tochn = 2;
    best = expressions3[0];
    while ((flag == 0) && (flagkolvo < 11))
    {
        cout << temporal.size() << endl;
        int kolvo_na_1[8];
        int vhod[8];
        for (int i = 0; i < 8; i++)
        {
            kolvo_na_1[i] = temporal.size() / 8;
        }
        vhod[0] = 0;
        for (int i = (temporal.size() / 8) * 8; i < temporal.size(); i++)
        {
            kolvo_na_1[i - (temporal.size() / 8) * 8] += 1;
        }
        for (int i = 1; i < 8; i++)
        {
            vhod[i] = vhod[i - 1] + kolvo_na_1[i - 1];
        }
        info1.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info1.kolvo_na_1 = kolvo_na_1[0];
        info1.vhod = vhod[0];
        info1.fitx = x;
        info1.fity = y;
        info1.fitz = z3;

        info2.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info2.kolvo_na_1 = kolvo_na_1[1];
        info2.vhod = vhod[1];
        info2.fitx = x;
        info2.fity = y;
        info2.fitz = z3;

        info3.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info3.kolvo_na_1 = kolvo_na_1[2];
        info3.vhod = vhod[2];
        info3.fitx = x;
        info3.fity = y;
        info3.fitz = z3;

        info4.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info4.kolvo_na_1 = kolvo_na_1[3];
        info4.vhod = vhod[3];
        info4.fitx = x;
        info4.fity = y;
        info4.fitz = z3;

        info5.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info5.kolvo_na_1 = kolvo_na_1[4];
        info5.vhod = vhod[4];
        info5.fitx = x;
        info5.fity = y;
        info5.fitz = z3;

        info6.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info6.kolvo_na_1 = kolvo_na_1[5];
        info6.vhod = vhod[5];
        info6.fitx = x;
        info6.fity = y;
        info6.fitz = z3;

        info7.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info7.kolvo_na_1 = kolvo_na_1[6];
        info7.vhod = vhod[6];
        info7.fitx = x;
        info7.fity = y;
        info7.fitz = z3;

        info8.command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
        info8.kolvo_na_1 = kolvo_na_1[7];
        info8.vhod = vhod[7];
        info8.fitx = x;
        info8.fity = y;
        info8.fitz = z3;

        thread th1(func_thread, info1, 1,3);
        thread th2(func_thread, info2, 2,3);
        thread th3(func_thread, info3, 3,3);
        thread th4(func_thread, info4, 4,3);
        thread th5(func_thread, info5, 5,3);
        thread th6(func_thread, info6, 6,3);
        thread th7(func_thread, info7, 7,3);
        thread th8(func_thread, info8, 8,3);
        th1.join();
        th2.join();
        th3.join();
        th4.join();
        th5.join();
        th6.join();
        th7.join();
        th8.join();
        if(flagkolvo!=0){
        for (int t = 0; t < temporal.size(); t++)
        {
            expr temp = temporal[t];
            expressions3.push_back(temp);
        }
        }
        else {
            for (int t = 0; t < temporal.size(); t++)
            {
                expr temp = temporal[t];
                expressions2[t]=temp;
            }
        }
        clReleaseCommandQueue(info1.command_queue);
        clReleaseCommandQueue(info2.command_queue);
        clReleaseCommandQueue(info3.command_queue);
        clReleaseCommandQueue(info4.command_queue);
        clReleaseCommandQueue(info5.command_queue);
        clReleaseCommandQueue(info6.command_queue);
        clReleaseCommandQueue(info7.command_queue);
        clReleaseCommandQueue(info8.command_queue);
        pareto_sort(expressions3);
        temporal.clear();
        temporal.resize(0);
        if (expressions3[0].tochnost < tochn)
        {
            best = expressions3[0];
            flag = 1;
            break;
        }
        if (expressions3[0].tochnost < best.tochnost)
            best = expressions3[0];
        cout << "expressions3 tochnost na pokolenii " << flagkolvo << "=";
        cout << expressions3[0].tochnost << endl;
        expressions3[0].print();
        cout << endl;
        cout << "best3=";
        cout << best.tochnost << endl;
        best.print();
        cout << endl;
        vector<expr> pop;
        for (int i = population-1; i >=0; i--)
        {
            pop = expressions3[i].exten(root3);
            expressions3.erase(expressions3.begin() + i);
            for (int j = 0; j < pop.size(); j++)
            {
                pop[j].optimization();
                vector<int> num;
                pop[j].makeint(num);
                if (root3.insert(num)) {
                    temporal.push_back(pop[j]);
                }
            }
        }
        flagkolvo++;
    }
    cout << "vot ono 3" << endl;
    best.print();
    cout << endl;
    cout << best.dlin << " " << best.tochnost << endl;
    for (int q = 0; q < best.consts.size(); q++)
        cout << best.consts[q] << " ";
    cout << endl;
    cout << flagkolvo << endl;
    otv = best;
    constss.push_back(best.consts);
    otvet[2] = otv;

    for (int i = 0; i < 3; i++)
    {
        otvet[i].print();
        cout << endl;
        cout << otvet[i].tochnost << endl;
    }
    double st = -1;
    double shag1 = 2 / double(n);
    double shag2 = 2 / double(m);
    for (int nom1 = 0; nom1 < n; nom1++)
    {
        for (int nom2 = 0; nom2 < m; nom2++)
        {
            //cout << "input color in " << nom1 << " " << nom2 << endl;
            //cin >> orig[nom1][nom2].r;
           // cin >> orig[nom1][nom2].g;
            //cin >> orig[nom1][nom2].b;
            orig[nom1][nom2].r = z1[nom1][nom2]/255;
            orig[nom1][nom2].g = z2[nom1][nom2]/255;
            orig[nom1][nom2].b = z3[nom1][nom2]/255;
            // orig[nom1][nom2].r = pow((-1 + nom1 * shag1)*(-1 + nom2 * shag2), 40);
             //orig[nom1][nom2].g = pow((-1 + nom1 * shag1)*(-1 + nom2 * shag2), 30);
             //orig[nom1][nom2].b = pow((-1 + nom1 * shag1)*(-1 + nom2 * shag2), 35);
            if (orig[nom1][nom2].r < 0)
                orig[nom1][nom2].r = 0;
            if (orig[nom1][nom2].g < 0)
                orig[nom1][nom2].g = 0;
            if (orig[nom1][nom2].b < 0)
                orig[nom1][nom2].b = 0;
            if (orig[nom1][nom2].r > 1)
                orig[nom1][nom2].r = 1;
            if (orig[nom1][nom2].g > 1)
                orig[nom1][nom2].g = 1;
            if (orig[nom1][nom2].b > 1)
                orig[nom1][nom2].b = 1;
            // cout << orig[nom1][nom2].r << endl;
             //cout << orig[nom1][nom2].g << endl;
             //cout << orig[nom1][nom2].b << endl;
        }
        bx[nom1] = -1 + nom1 * shag1;
    }

    for (int nom = 0; nom < m; nom++)
    {
        by[nom] = -1 + nom * shag2;
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    GLint win1 = glutCreateWindow("image1");
    glutReshapeFunc(changeViewPort1);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(render);
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(550, 0);
    GLint win2 = glutCreateWindow("image2");
    glutReshapeFunc(changeViewPort2);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(render1);
    GLenum err2 = glewInit();
    if (GLEW_OK != err2) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(1100, 0);
    GLint win3 = glutCreateWindow("graphic1");
    glutReshapeFunc(changeViewPort3);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(render2);
    GLenum err3 = glewInit();
    if (GLEW_OK != err3) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 550);
    GLint win4 = glutCreateWindow("graphic2");
    glutReshapeFunc(changeViewPort2);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(render3);
    GLenum err4 = glewInit();
    if (GLEW_OK != err4) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(550, 550);
    GLint win5 = glutCreateWindow("graphic3");
    glutReshapeFunc(changeViewPort2);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(render4);
    GLenum err5 = glewInit();
    if (GLEW_OK != err5) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutMainLoop();
    return 0;
}
