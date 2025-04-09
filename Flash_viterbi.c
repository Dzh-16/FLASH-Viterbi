#define _CRT_SECURE_NO_WARNINGS //FOPEN

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>



#define K_STATE 3965 // 状态空间大小为10                         修改为K相同的值
#define T_STATE 50 // 观测状态大小为10

#define obserRouteLEN 256 //观测路径长度 �?10                  修改为T相同的值

#define MAX_THREADS 2
clock_t start_time;
clock_t end_time;//COUNT TIME
LARGE_INTEGER t1, t2, tc;

typedef float ElementType;    //MAX HEAP 堆排序元素类型为浮点型，即转换概率的单位为double

ElementType A[K_STATE][K_STATE];//当数组维数过大无法申请大小过大的内存,用malloc
ElementType B[K_STATE][T_STATE];
ElementType pi[K_STATE];
int Obroute[obserRouteLEN];

int snvOutPutPath[obserRouteLEN]; // output

typedef struct Queue
{
    int* preNode; //问题路径的前�?
    int* sucNode;//问题路径的后�?
    int  front; //头指�?
    int  rear; //尾指�?
}Queue;

// 线程池结构
typedef struct {
    HANDLE threads[MAX_THREADS]; // 线程句柄数组
    CRITICAL_SECTION lock;       // 临界区对象，用于线程安全操作
    CONDITION_VARIABLE pool_wake;  // 事件对象数组，用于线程间的同步
    Queue Q;
    int active_threads;          // 活动线程的数量
    int task_count;
} ThreadPool;
int TotalTask;
// typedef struct TreeNode // BST TREE NODE        �?
// {
//     float count_score; // 记录对应节点概率�?
//     int node_no;       // 记录当前节点对应隐状�?
//     struct TreeNode* lchild;
//     struct TreeNode* rchild;
// } TreeNode;

void InitOutputpath()
{

    int i = 0;

        for (i = 0; i < obserRouteLEN+1; i++)
        {
        
            snvOutPutPath[i] =  -1;
    
        }

}

void InitQueue(Queue* Q)        //初始化队�?
{
    //为队列分配存储空间，为前驱后继队�?
    Q->preNode = (int*)malloc(sizeof(int) * obserRouteLEN);//给指针变量分配长度为观测路径长的整型存储空间�?
    Q->sucNode = (int*)malloc(sizeof(int) * obserRouteLEN);
    if ((Q->preNode != NULL) && (Q->sucNode != NULL))
        Q->front = Q->rear = 0;
    //初始时队列为空，头指针和尾指针都指向0位置
    else
    {
        printf("allocation is failed !!");
        return;
    }
}


void EnQueue(Queue* Q, int preside, int sucside)        //进队
{
    //判断循环队列是否已满
    if (((Q->rear + 1) % obserRouteLEN) == Q->front)   //判满条件是尾指针的下一个位置（通过取模运算 (Q->rear + 1) % obserRouteLEN）等于头指针的位置（Q->front）�?
        return;
    //队列未满，将数据入队
    Q->preNode[Q->rear] = preside;//前驱边界入队
    Q->sucNode[Q->rear] = sucside;//后继边界入队
    //更改尾指针的指向
    Q->rear = (Q->rear + 1) % obserRouteLEN;
}

int DeQueue(Queue* Q, int* preside, int* sucside)       //出队
{
    //判断循环队列是否为空
    if (Q->front == Q->rear)
        return 0;
    //出队前将前驱和后继节点的值保�?
    *preside = Q->preNode[Q->front];
    *sucside = Q->sucNode[Q->front];
    //如果非空，实现可循环出队
    Q->front = (Q->front + 1) % obserRouteLEN;
    return 1;
}
void ShowQueue(Queue* Q)                
{
    //遍历循环队列中的元素，并将数据打�?
    for (int i = Q->front; i != Q->rear; i = (i + 1) % obserRouteLEN)
    {
        printf("(%d->%d), ", Q->preNode[i], Q->sucNode[i]);
        //此操作是为了实现循环遍历

    }
    printf("\n");
}


int Length(Queue* Q)            
{
    //计算尾指针位置与头指针位置的差距
    int len = Q->rear - Q->front;
    //如果为正数，那么len就是队列的长度；如果为负数，那么obserRouteLEN+len才是队列的长�?
    len = (len > 0) ? len : obserRouteLEN + len;
    return len;
}


void max_score(ElementType arr[], int arr_count, int* max_temp_no, ElementType* max_temp)       //求求队列概率最大值和下标
{

    if (NULL == arr)
        return;
    int max_no = 0;                 //概率值的No�?
    ElementType max = arr[0];       //概率值大�?
    for (int i = 0; i < arr_count; i++)
    {
        if (max < arr[i])
        {
            max = arr[i];
            max_no = i;
        }
    }
    *max_temp_no = max_no;
    *max_temp = max;
}


void InitMatrixA(char* str_MatrixA)
{

    int i = 0, j = 0;
    // test=malloc(sizeof(ElementType) * K_STATE * K_STATE);
    ElementType *tmpt = (ElementType *)malloc(sizeof(ElementType) * K_STATE * K_STATE);

    if (tmpt)
    {
        FILE *fpRead = fopen(str_MatrixA, "rb+"); // matrixA or matrixB 程序成功打开文件后，fopen()将返回文件指针。文件指针的类型是FILE

        if (fpRead == NULL)
        {
            printf("open fail errno = %d reason = %s \n", errno, strerror(errno));
            return;
        }
        int retnum = 0;
        for (i = 0; i < K_STATE * K_STATE; i++)
        {
            retnum = fscanf(fpRead, "%f", &tmpt[i]); // 使用fscanf函数从文件中读取一个double类型的数据，并将其存储在tmpt[i]中。
            // printf("%f  ",tmpt[i]);
        }
        fclose(fpRead);

        for (i = 0; i < K_STATE; i++)
        {
            for (j = 0; j < K_STATE; j++)
            {
                A[i][j] = tmpt[i * K_STATE + j];
            }
        }
    }
    free(tmpt);
}

void InitMatrixB(char* str_MatrixB )
{

    int i = 0, j = 0;
    ElementType *tmpt = (ElementType *)malloc(sizeof(ElementType) * K_STATE * T_STATE);
    if (tmpt)
    {
        FILE *fpRead = fopen(str_MatrixB, "rb+"); // matrixA or matrixB

        if (fpRead == NULL)
        {
            return;
        }
        int retnum = 0;
        for (i = 0; i < K_STATE * T_STATE; i++)
        {
            retnum = fscanf(fpRead, "%f", &tmpt[i]);
        }
        fclose(fpRead);

        for (i = 0; i < K_STATE; i++)
        {
            for (j = 0; j < T_STATE; j++)
            {
                B[i][j] = tmpt[i * T_STATE + j];
            }
        }
    }

    free(tmpt);
}

void InitMatrixPI(char* str_MatrixPI)
{
    int i, j;
    ElementType *tmpt = (ElementType *)malloc(sizeof(ElementType) * K_STATE);
    if (tmpt == NULL)
    {
        printf("malloc failed!!");
        return;
    }
    FILE *fpRead = fopen(str_MatrixPI, "rb+");


    if (fpRead == NULL)
    {
        return;
    }
    int retnum = 0;
    for (i = 0; i < K_STATE; i++)
    {
        retnum = fscanf(fpRead, "%f", &tmpt[i]);
    }
    fclose(fpRead);
    for (j = 0; j < K_STATE; j++)
    {
        pi[j] = tmpt[j];
    }
    printf("\n");
    free(tmpt);

    //
}


void InitObRoute(char* str_ObRoute)
{
    int i, j;
    int tmpt[obserRouteLEN] = { 0 };
    FILE *fpRead3 = fopen(str_ObRoute, "rb+");

    
    if (fpRead3 == NULL)
    {
        return;
    }
    int retnum = 0;
    for (i = 0; i < obserRouteLEN; i++)
    {
        retnum = fscanf(fpRead3, "%d", &tmpt[i]);

    }
    fclose(fpRead3);
    for (j = 0; j < obserRouteLEN; j++)
    {
        Obroute[j] = tmpt[j];
    }
    //
}

void ViterbiNDivide(int qpreNode, int qsucNode, int N, Queue *Q)
{
    int TobsLenth = obserRouteLEN;
    if (qsucNode - qpreNode < 2*N)  //小于6 无法四分
        return;
    // (TobsLenth <= 1)
      //  return;
    int i, j;
    ElementType scorearr[K_STATE] = { 0 }; 

    int T_obs = Obroute[qpreNode]; 
    int midpoint[N+1], gap_length = (qsucNode - qpreNode) / N, gap_extra = (qsucNode - qpreNode) % N;
    midpoint[0] = qpreNode, midpoint[N] = qsucNode; //0无意义
    for(int i = 1; i < N; ++i)
    {
        midpoint[i] = midpoint[i-1]+gap_length;
        if(gap_extra > 0) gap_extra--, midpoint[i]++;
    }
    ElementType snvT1[K_STATE];
    ElementType snvTtmp[K_STATE];
    int** snv = (int**)malloc(sizeof(int*)*K_STATE);
    int** snv_tmp = (int**)malloc(sizeof(int*)*K_STATE);
    for (i = 0; i < K_STATE; i++) // �?
    {
        snv[i] = (int*)malloc(sizeof(int)*N);
        snv_tmp[i] = (int*)malloc(sizeof(int)*N);
        snvTtmp[i] = (qpreNode == 0 ? log(pi[i]) : log(A[snvOutPutPath[qpreNode-1]][i])) + log(B[i][T_obs]);
        snvT1[i] = snvTtmp[i];
        for(int j = 1; j < N; ++j)
        {
            snv_tmp[i][j] = (qpreNode == 0 ? -1 : snvOutPutPath[qpreNode-1]);
            snv[i][j] = snv_tmp[i][j];
        }
    }
    for (j = qpreNode+1; j < qsucNode+1; j++)      
    {
        T_obs = Obroute[j];           // 更新观测序列当前�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            for (int k = 0; k < K_STATE; k++)
            {
                scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
            }
            int scoreNo = 0;
            ElementType scoreMax = 0;
            int* p = &scoreNo;
            ElementType* q = &scoreMax;
            max_score(scorearr, K_STATE, p, q);
            snvTtmp[i] = scoreMax;

            for(int p = 1; p < N; ++p)
            {
                if(j > midpoint[p]+1) snv_tmp[i][p] = snv[scoreNo][p];
                else snv_tmp[i][p] = snv[i][p] = scoreNo;
            }
        }
        for (i = 0; i < K_STATE; i++) 
        {
            snvT1[i] = snvTtmp[i];
            for(int p = 1; p < N; ++p)
            {
                if(j > midpoint[p]+1) snv[i][p] = snv_tmp[i][p];
            }
        }
    }
    

    if (qpreNode == 0 && qsucNode == obserRouteLEN-1) 
    {
        ElementType last_max = snvT1[0];
        int last_max_no = 0;
        for (int i = 0; i < K_STATE; i++)
        {
            if (last_max < snvT1[i])
            {
                last_max = snvT1[i];
                last_max_no = i;
            }
        }

        snvOutPutPath[qsucNode] = last_max_no;        
        for(int p = 1; p < N; ++p)
        {
            snvOutPutPath[midpoint[p]] = snv[last_max_no][p];
        }

        printf("总占用的内存大小（不算输出）: %zu 字节\n", (sizeof(int)*K_STATE*2+sizeof(snvT1)+sizeof(snvTtmp))*N+(sizeof(scorearr)+sizeof(last_max)+sizeof(last_max_no)+sizeof(T_obs))*MAX_THREADS+sizeof(midpoint));
        printf("总占用的内存大小: %zu 字节\n", (sizeof(int)*K_STATE*2+sizeof(snvT1)+sizeof(snvTtmp))*N+sizeof(snvOutPutPath)+(sizeof(scorearr)+sizeof(last_max)+sizeof(last_max_no)+sizeof(T_obs))*MAX_THREADS+sizeof(midpoint));
        printf("\n log lastmax%f", last_max);//log值测�?
        printf("\n lastmax%f", exp(last_max));//最大路径概率�?
    }
    else//其余都执�?
    {
        int connt = snvOutPutPath[qsucNode];//找到当前最末点的状态�?
        for(int p = 1; p < N; ++p)
        {
            snvOutPutPath[midpoint[p]] = snv[connt][p];
        }
    }

    for(int i = 0; i < K_STATE; ++i)
    {
        free(snv[i]);
        free(snv_tmp[i]);
    }
    free(snv);
    free(snv_tmp);

    EnQueue(Q,qpreNode,midpoint[1]);
    for(int p = 1; p < N; ++p)
    {
        EnQueue(Q,midpoint[p]+1,midpoint[p+1]);
    }
}

void Viterbifirst(int qpreNode, int qsucNode)       //大于3时的情况
{

    int TobsLenth = obserRouteLEN;
    if (qsucNode - qpreNode < 0)        //后继-前驱小于0，结�?
        return;
    // (TobsLenth <= 1)
      //  return;
    double test_pi,test_B,or_B,test_A;
    int i, j;
    ElementType scorearr[K_STATE] = { 0 }; // 存储所有概率值，用于求最大概�?

    int T_obs = Obroute[qpreNode ]; // 观测序列初值，第一列观测�?   *当前序列起始的观测�?
    int midpoint = (qpreNode + qsucNode) / 2;  //path的中间节�?,运行结束后得到当前Path[midpoint]
    //snvOutPutPath[midpoint - 1] = -1; // init path[midpoint-1]
    ElementType snvT1[K_STATE];       // T1_table,prob
    ElementType snvTtmp[K_STATE];       // T1_table,save for last prob
    int snvT2[K_STATE];               // T2_table,path
    int snvT3[K_STATE];               // Mid Table,midpath
    if (qpreNode == 0) //仅当第一列的T表格初始化，当头节点�?0�?
    {
        for (i = 0; i < K_STATE; i++)
        {
            test_pi = log(pi[i]);
            or_B   = B[i][T_obs];
            test_B = log(B[i][T_obs]);
            snvT1[i] = log(pi[i]) + log(B[i][T_obs]); // 初始化T1,用log避免数据下溢�?
            snvT2[i] = -1;                            // 初始化T2
            snvT3[i] = -1;                            // 初始化T3
        }
    }
    else//其他情况初始一�?,若不是初始的列，说明此处是之前的mid列，因此直接找到该列的对应状态，其他初始�?0
    {
        T_obs = Obroute[qpreNode];           // 更新观测序列当前�?
        int State_temp=snvOutPutPath[qpreNode-1]; //  当前起始对应的状态�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            test_A = log(A[State_temp][i]);
            test_B = log(B[i][T_obs]);
            snvT2[i] =  State_temp;
            snvTtmp[i]= log(A[State_temp][i]) + log(B[i][T_obs]);
        }
        for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
        {
            snvT1[i] = snvTtmp[i];
        }
    }
    // 第pre+1列到第midpoint�?
    for (j = qpreNode+1; j < midpoint+1; j++)      
    {
        T_obs = Obroute[j];           // 更新观测序列当前�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            for (int k = 0; k < K_STATE; k++)
            {
                double test_A,test_B;
                test_A = log(A[k][i]);
                test_B = log(B[i][T_obs]);
                scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
            }
            int scoreNo = 0;
            ElementType scoreMax = 0;
            int* p = &scoreNo;
            ElementType* q = &scoreMax;
            max_score(scorearr, K_STATE, p, q);
            snvTtmp[i] = scoreMax; // 下溢�?
            snvT2[i] = scoreNo;//前半不需要存路径
        }
        int lastno=snvT2[K_STATE-1];
        ElementType lastmax=snvTtmp[K_STATE-1];
        for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
        {
            snvT1[i] = snvTtmp[i];
        }
    }

    // 第midpoint+1�?

    T_obs = Obroute[midpoint+1];           // 更新观测序列当前�?,T2[midpoint+1]
    for (i = 0; i < K_STATE; i++) // �?
    {
        for (int k = 0; k < K_STATE; k++)
        {
            double test_A,test_B,test_T1;
            test_A = log(A[k][i]);
            test_B = log(B[i][T_obs]);
            test_T1 = snvT1[k];
            scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
        }
        int scoreNo = 0;
        ElementType scoreMax = 0;
        int* p = &scoreNo;
        ElementType* q = &scoreMax;
        max_score(scorearr, K_STATE, p, q);
        snvTtmp[i] = scoreMax; // 下溢�?
        snvT2[i] = scoreNo;//后半路径开始保存T2
        snvT3[i] = scoreNo;//第midpoint节点的前驱路�?
    }
    for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
    {
        snvT1[i] = snvTtmp[i];
    }

    // 第midpoint+2列到第qsucNode�?
    for (j = midpoint + 2; j < qsucNode + 1; j++)      
    {
        T_obs = Obroute[j];           // 更新观测序列当前�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            for (int k = 0; k < K_STATE; k++)
            {
                double test_A,test_B,test_T1;
                test_A = log(A[k][i]);
                test_B = log(B[i][T_obs]);
                test_T1 = snvT1[k];
                scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
            }
            int scoreNo = 0;
            ElementType scoreMax = 0;
            int* p = &scoreNo;
            ElementType* q = &scoreMax;
            max_score(scorearr, K_STATE, p, q);
            snvTtmp[i] = scoreMax; // 下溢�?
            //snvT2[i] = scoreNo;//当前前驱节点为scoreno
            // snvT2[i] = scoreNo;
            snvT2[i] = snvT3[scoreNo];//取T3对应前驱节点下标节点内的值即为当前列到mid列路�?
            
        }
        // for (i = 0; i < K_STATE; i++)
        // {
        //     snvT2[i] = snvT3[snvT2[i]];//取T3对应前驱节点下标节点内的值即为当前列到mid列路�?
        // }
    
        for (i = 0; i < K_STATE; i++) // 将暂存概率值拷贝，并更新T3列，T2和T3的迭代更�?
        {
            snvT1[i] = snvTtmp[i];
            snvT3[i] = snvT2[i];
        }
}
    
    //此时保存了中间节点路径和最后节点的概率
    //求最后一个节点的路径,并根据此回溯路径中点，输�?
    int mi = (obserRouteLEN - 1) / 2;
    if (midpoint == mi ) //仅第一次执�?
    {
        ElementType last_max = snvT1[0];//回溯处本身为负�?,不要�?0比较
        int last_max_no = 0;
        for (int i = 0; i < K_STATE; i++)
        {
            if (last_max < snvT1[i])
            {
                last_max = snvT1[i];
                last_max_no = i;
            }
        }


        snvOutPutPath[qsucNode] = last_max_no;        // 倒数第一个节点�? //?存入最大�?
        snvOutPutPath[midpoint] = snvT2[last_max_no]; // 第midpoint个节点值，最后T2与T3是一样的

        printf("\n log lastmax%f", last_max);//log值测�?
        printf("\n lastmax%f", exp(last_max));//最大路径概率�?
    }
    else
    {
        int connt = snvOutPutPath[qsucNode];//找到当前最末点的状态�?
        snvOutPutPath[midpoint] = snvT3[connt]; //当前路径范围内输出的，将此时T3到该状态点的值进行输�?
    }

}

void viter1(int qpreNode, int qsucNode)//pre + 1 = suc,mid = pre  最末端区间�?2的加�?
{

    int TobsLenth = obserRouteLEN;
    if (qsucNode - qpreNode <= 0)
        return;
    // (TobsLenth <= 1)
      //  return;
    int i, j;
    ElementType scorearr[K_STATE] = { 0 }; // 存储所有概率值，用于求最大概�?
    int T_obs = Obroute[qpreNode]; // 观测序列初�?

    ElementType snvT1[K_STATE];       // T1_table,prob
    ElementType snvTtmp[K_STATE];       // T1_table,save for last prob
    int snvT2[K_STATE];               // T2_table,path
    int snvT3[K_STATE];               // Mid Table,midpath
    
    //snvOutPutPath[midpoint - 1] = -1; // init path[midpoint-1]
    if (qpreNode == 0) //仅当第一列的T表格初始�?
    {
        for (i = 0; i < K_STATE; i++)
        {
            snvT1[i] = log(pi[i]) + log(B[i][T_obs]); // 初始化T1,用log避免数据下溢�?
            snvT2[i] = -1;                            // 初始化T2
            snvT3[i] = -1;                            // 初始化T3
        }
    }
    else//其他情况初始一�?,若不是初始的列，说明此处是之前的mid列，因此直接找到该列的对应状态，其他初始�?0
    {
        T_obs = Obroute[qpreNode];           // 更新观测序列当前�?
        int State_temp=snvOutPutPath[qpreNode-1]; //  当前起始对应的状态�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            snvT2[i] =  State_temp;
            snvTtmp[i]= log(A[State_temp][i]) + log(B[i][T_obs]);
        }
        for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
        {
            snvT1[i] = snvTtmp[i];
        }
    }

    // 第midpoint�?

    T_obs = Obroute[qsucNode];           // 更新观测序列当前�?,T2[midpoint+1]
    for (i = 0; i < K_STATE; i++) // �?
    {
        for (int k = 0; k < K_STATE; k++)
        {
            scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
        }
        int scoreNo = 0;
        ElementType scoreMax = 0;
        int* p = &scoreNo;
        ElementType* q = &scoreMax;
        max_score(scorearr, K_STATE, p, q);
        snvTtmp[i] = scoreMax; // 下溢�?
        snvT2[i] = scoreNo;//后半路径开始保存T2
        //snvT3[i] = scoreNo;//第midpoint节点的前驱路�?
    }
    for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
    {
        snvT1[i] = snvTtmp[i];
    }



        int connt = snvOutPutPath[qsucNode];
        snvOutPutPath[qpreNode] = snvT2[snvOutPutPath[qsucNode]]; //当前路径范围内输出的
   
}


void viter2(int qpreNode, int qsucNode)//pre+1 = mid,mid+1 = suc   最末端区间�?3的加�?
{

    int TobsLenth = obserRouteLEN;
    if (qsucNode - qpreNode < 0)
        return;
    // (TobsLenth <= 1)
      //  return;
    int i, j;
    ElementType scorearr[K_STATE] = { 0 }; // 存储所有概率值，用于求最大概�?
    int T_obs = Obroute[qpreNode]; // 观测序列初�?
    int midpoint = qpreNode +1;  //path的中间节�?,运行结束后得到当前Path[midpoint]
    ElementType snvT1[K_STATE];       // T1_table,prob
    ElementType snvTtmp[K_STATE];       // T1_table,save for last prob
    int snvT2[K_STATE];               // T2_table,path
    int snvT3[K_STATE];               // Mid Table,midpath
    if (qpreNode == 0) //仅当第一列的T表格初始�?
    {
        for (i = 0; i < K_STATE; i++)
        {
            snvT1[i] = log(pi[i]) + log(B[i][T_obs]); // 初始化T1,用log避免数据下溢�?
            snvT2[i] = -1;                            // 初始化T2
            snvT3[i] = -1;                            // 初始化T3
        }
    }
    else//其他情况初始一�?
    {
        T_obs = Obroute[qpreNode];           // 更新观测序列当前�?
        int State_temp=snvOutPutPath[qpreNode-1]; //  当前起始对应的状态�?
        for (i = 0; i < K_STATE; i++) // �?
        {
            snvT2[i] =  State_temp;
            snvTtmp[i]= log(A[State_temp][i]) + log(B[i][T_obs]);
        }
        for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
        {
            snvT1[i] = snvTtmp[i];
        }
    }


    T_obs = Obroute[midpoint];           // 更新观测序列当前�?
    for (i = 0; i < K_STATE; i++) // �?
    {
        for (int k = 0; k < K_STATE; k++)
        {
            scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
        }
        int scoreNo = 0;
        ElementType scoreMax = 0;
        int* p = &scoreNo;
        ElementType* q = &scoreMax;
        max_score(scorearr, K_STATE, p, q);
        snvTtmp[i] = scoreMax; // 下溢�?
        //snvT2[i] = scoreNo;//前半不需要存路径
    }
    for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
    {
        snvT1[i] = snvTtmp[i];
    }

    // 第midpoint�?

    T_obs = Obroute[midpoint + 1];           // 更新观测序列当前�?,T2[midpoint+1]
    for (i = 0; i < K_STATE; i++) // �?
    {
        for (int k = 0; k < K_STATE; k++)
        {
            scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率�?,用log避免数值下溢出,其中T1已经为log 
        }
        int scoreNo = 0;
        ElementType scoreMax = 0;
        int* p = &scoreNo;
        ElementType* q = &scoreMax;
        max_score(scorearr, K_STATE, p, q);
        snvTtmp[i] = scoreMax; // 下溢�?
        snvT2[i] = scoreNo;//后半路径开始保存T2
        //snvT3[i] = scoreNo;//第midpoint节点的前驱路�?
    }
    for (i = 0; i < K_STATE; i++) // 将暂存概率值拷�?
    {
        snvT1[i] = snvTtmp[i];
    }

    
    snvOutPutPath[midpoint] = snvT2[snvOutPutPath[qsucNode]]; //当前路径范围内输出的
    

}

void printRoute()//输出路径
{

    printf("\npathlenth%d,路径为：\n", obserRouteLEN);

    for (int i = 0; i < obserRouteLEN; i++)
    {
        printf(" %d,", snvOutPutPath[i]);

    }
    printf("\n");
}
int maxQueueSize = 1; // 记录队列的最大长度
// 工作线程函数
DWORD WINAPI Worker(LPVOID arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    while (1) {
        int pre = 0,suc = obserRouteLEN-1, midpoint;
        int *qpre = &pre,*qsuc = &suc;
        EnterCriticalSection(&pool->lock); // 进入临界区
        int currentQueueSize;
        while((currentQueueSize = ((pool->Q).rear - (pool->Q).front + obserRouteLEN) % obserRouteLEN) == 0 
                && pool->task_count < TotalTask)
        {
            SleepConditionVariableCS(&pool->pool_wake, &pool->lock, INFINITE);
        }
        if (currentQueueSize > maxQueueSize)
        {
            maxQueueSize = currentQueueSize;
        }
        if (DeQueue(&(pool->Q),qpre,qsuc) == 0) {       // 检查任务队列是否为空
            LeaveCriticalSection(&pool->lock);
            break;
        }
        pool->task_count++;
        LeaveCriticalSection(&pool->lock);                  // 离开临界区
        midpoint = (pre + suc) / 2;
        if (suc - pre > 2)
        {
            Viterbifirst(pre, suc);
            EnterCriticalSection(&pool->lock); // 进入临界区
            EnQueue(&(pool->Q), pre, midpoint);
            EnQueue(&(pool->Q), midpoint + 1, suc);
            LeaveCriticalSection(&pool->lock); // 离开临界区
        }
        else if (suc == pre + 2)        
        {
            viter2(pre, suc);
            EnterCriticalSection(&pool->lock); // 进入临界区
            EnQueue(&(pool->Q), pre, midpoint);
            LeaveCriticalSection(&pool->lock); // 离开临界区
        }           
        else //pre=suc+1,pre = suc
        {
            viter1(pre, suc);
        }
        WakeAllConditionVariable(&pool->pool_wake);
    }
    return 0;
}
// 初始化线程池
void ThreadPoolInit(ThreadPool* pool) {
    InitQueue(&(pool->Q));
    pool->active_threads = MAX_THREADS;
    InitializeCriticalSection(&pool->lock); // 初始化临界区对象
    InitializeConditionVariable(&pool->pool_wake);
    for (int i = 0; i < MAX_THREADS; ++i) {
        pool->threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)pool, 0, NULL); // 创建工作线程
    }
}
// 销毁线程池
void DestroyThreadPool(ThreadPool* pool) {
    for (int i = 0; i < MAX_THREADS; ++i) {
        WaitForSingleObject(pool->threads[i], INFINITE); // 等待工作线程结束
        CloseHandle(pool->threads[i]); // 关闭线程句柄
    }
    DeleteCriticalSection(&pool->lock); // 删除临界区对象
}
void tracebackroute()//space主体        调用刚刚主体的代�?
{
    int TobsLenth = obserRouteLEN;      //观察路径�?
    int pre = 0;                        //当前处理的路径的起始
    int suc = TobsLenth - 1;            //结束�?
    int* qpre = &pre;                   //声明指针qpre和qsuc分别指向pre和suc，用于在函数中更新起始和结束点�?
    int* qsuc = &suc;
    ThreadPool pool;
    ThreadPoolInit(&pool);
    printf("Qmemory: %zu 字节\n", sizeof(pool.Q));
    int Ndivide = MAX_THREADS;
    if(suc - pre >= 2*Ndivide)
    {
        TotalTask = obserRouteLEN-Ndivide;
        ViterbiNDivide(pre,suc,Ndivide,&(pool.Q));
        WakeAllConditionVariable(&pool.pool_wake);
    }
    else
    {
        TotalTask = obserRouteLEN-1;
        EnQueue(&(pool.Q), pre, suc);
        WakeConditionVariable(&pool.pool_wake);
    }

    DestroyThreadPool(&pool);
    ShowQueue(&(pool.Q));
    printf("队列Q的最大数据量: %d\n", maxQueueSize);

}

int exmpleSNVviter()
{

    int excu_i = 0;
    char* str_MatrixA="";
    char* str_MatrixB="";
    char* str_MatrixPI="";
    char* str_ObRoute="";
    InitOutputpath();
    InitMatrixA(str_MatrixA);
    InitMatrixB(str_MatrixB);
    InitMatrixPI(str_MatrixPI);
    InitObRoute(str_ObRoute);
    QueryPerformanceFrequency(&tc); // time 算执行时�?
    QueryPerformanceCounter(&t1);   // time
    tracebackroute();
    QueryPerformanceCounter(&t2);                                                           // time
    printf("\nFLASHVITERBI_time:%lf s", (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart); // time
    printRoute();//输出路径
    printf("\nthank\n");
    return 1;
}

int main()
{

    if (!exmpleSNVviter()) // exmplesnv

    {
        printf("error");
        return 0;
    }

    printf("success");
    return 1;
}
