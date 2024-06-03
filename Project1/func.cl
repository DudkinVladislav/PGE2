
__kernel void sinsqrt(__global __write_only float* mes)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    float x = xid;
    float y = yid;
    if (xid < 10 && yid < 10)
    {
        mes[xid * 10 * 2 + yid * 2 + 1] = native_sqrt(x);
        mes[xid * 10 * 2 + yid * 2] = native_sin(y);
    }
}
__kernel void test(__global int* sum, __global int* a, __global int* b)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    float x = xid;
    float y = yid;
    if (xid < 10 && yid < 10)
    {
       sum[xid * 10  + yid ] = a[xid*10+yid]+b[xid*10+yid];
    }
}

__kernel void sumz(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y, int params_count)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        double sum=0;
        if (input[xid * max_y + yid + max_x * max_y] > 255)
        {
            sum = 255;
        }
        else
        {
            if (input[xid * max_y + yid + max_x * max_y] > 0)
            {
                sum = input[xid * max_y + yid + max_x * max_y];
            }
        }
        sum = sum + input[xid * max_y + yid];
        output[xid * max_y + yid] = sum;
    }
}

__kernel void minz(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    double m = 0;
    if (xid < max_x && yid < max_y)
    {
        if (input[xid * max_y + yid] > 255)
        {
            m = 255;
        }
        else
        {
            if (input[xid * max_y+yid] > 0)
            {
                m = input[xid * max_y + yid];
            }
        }
        output[xid * max_y + yid] = m*(-1);
    }
}

__kernel void summ(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y, int params_count)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        double sum = 0;
        for (int i = 0; i < params_count; i++)
        {
            sum += input[xid * max_y + yid + i * max_x * max_y];
        }
        output[xid * max_y + yid] = sum;
    }
}
 

__kernel void prr(__global  double* output, __global  double* input, int max_x, int max_y, int params_count)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    double x = yid;
    double y = xid; 
    double pr = 1;
    if (xid < max_x && yid < max_y)
    {
        for (int i = 0; i < params_count; i++)
        {
            pr *= input[xid * max_y + yid + i * max_x * max_y];
        }
        output[xid * max_y + yid] = pr;
    }
}

__kernel void minn(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = (-1)*input[xid*max_y+yid];
    }
}

__kernel void divv(__global __write_only double* output, __global __read_only double* input1, __global __read_only double* input2, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = input1[xid*max_y+yid]/input2[xid*max_y+yid];
    }
}

__kernel void coss(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_cos(input[xid * max_y + yid]);
    }
}

__kernel void sinn(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_sin(input[xid * max_y + yid]);
    }
}

__kernel void tann(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_tan(input[xid * max_y + yid]);
    }
}

__kernel void logg(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_log(input[xid * max_y + yid]);
    }
}

__kernel void expp(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_exp(input[xid * max_y + yid]);
    }
}

__kernel void sqrtt(__global __write_only double* output, __global __read_only double* input, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = native_sqrt(input[xid * max_y + yid]);
    }
}

__kernel void poww(__global __write_only double* output, __global __read_only double* input1, __global __read_only double* input2, int max_x, int max_y)
{
    // получаем текущий id.
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    if (xid < max_x && yid < max_y)
    {
        output[xid * max_y + yid] = pow(input1[xid * max_y + yid] , input2[xid * max_y + yid]);
    }
}