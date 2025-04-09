#define _CRT_SECURE_NO_WARNINGS // FOPEN

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

// #define STR_LEN 5   // 输入状态序列字符长
// #define K_STATE 128 // 状态空间大小为10
// #define T_STATE 100 // 观测状态大小为10
#define K_STATE 3965 // 状态空间大小为10                         修改为K相同的值
#define T_STATE 50 // 观测状态大小为10
// #define SUPER_K_STATE     20      //状态空间大小为200
// #define SUPER_T_STATE     20
// #define FST_SIZE    5       //firstN 大小
// #define BST_SIZE    5       //bestN 大小

#define obserRouteLEN 256 //观测路径长度 �?10                  修改为T相同的值

#define MAX_ELEMENTS 64// 状态约束大小                          修改为K相同的值

#define MAX_THREADS 16
clock_t start_time;
clock_t end_time; // COUNT TIME
LARGE_INTEGER t1, t2, tc;

typedef float ElementType; // MAX HEAP 堆排序元素类型为浮点型，即转换概率的单位为double
typedef int Status;         /* Status是函数的类型,其值是函数结果状态代码，如OK?? */

ElementType A[K_STATE][K_STATE]; // 当数组维数过大无法申请大小过大的内存,用malloc
ElementType B[K_STATE][T_STATE];
ElementType pi[K_STATE];
int Obroute[obserRouteLEN]; // 观测序列

// int spaceNaiveViterbiPath[obserRouteLEN];//space naive viterbi path
//  int snvT2[K_STATE];               // T2_table,path
//  int snvT3[K_STATE];               // Mid Table,midpath
int snvOutPutPath[obserRouteLEN]; // output
/// 存储队列相关///////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct Queue
{
    int *preNode; // 问题路径的前驱
    int *sucNode; // 问题路径的后继
    int front;    // 头指针
    int rear;     // 尾指针
} Queue;

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

/// 最大堆相关///////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct element
{
    ElementType Value;
    int State;
    int T3_State;
} element;
// typedef struct TreeNode // BST TREE NODE        树
// {
//     float count_score; // 记录对应节点概率值
//     int node_no;       // 记录当前节点对应隐状态
//     struct TreeNode* lchild;
//     struct TreeNode* rchild;
// } TreeNode;

void insert_binary_tree(element *heap, ElementType Value, int State, int T3_State)
{
    (*heap).Value = Value;
    (*heap).State = State;
    (*heap).T3_State = T3_State;
}

void initial_heap_element(element *heap)
{
    (*heap).Value = 0;
    (*heap).State = -1;
    (*heap).T3_State = -1;
    // for (int i = 1; i < MAX_ELEMENTS + 1; i++)
    // {
    //     (*(heap + i)).Value = 0;
    //     (*(heap + i)).State = -1;
    //     (*(heap + i)).T3_State = -1;
    // }
}


int Find_T3_State(element **heap, int state)
{
    int total = (*heap)[0].Value; // 获取堆中元素的总数

    for (int i = 1; i <= total; i++)
    {
        if ((*heap)[i].State == state)
        {
            return (*heap)[i].T3_State; // 找到匹配的state值，返回对应的T3_State值
        }
    }

    return -1; // 如果未找到匹配的state值，返回-1表示未找到
}

void print_heap_element(element **heap)
{
    for (int i = 1; i < MAX_ELEMENTS + 1; i++)
    {
        printf("heap element is %lf,state is %d,T3 state is %d\n", (*((*heap) + i)).Value, (*((*heap) + i)).State, (*((*heap) + i)).T3_State);
    }
}

void create_min_heap(element **heap)
{
    int total = (**heap).Value;
    /// 求倒数第一个非叶子结点
    int child = 2, parent = 1;
    for (int node = total / 2; node > 0; node--)
    {
        parent = node;
        child = 2 * node;
        // int max_node = 2 * node + 1;
        element temp = *((*heap) + parent);     // 保存当前父节点
        // for (; child <= total; child *= 2, max_node = 2 * parent + 1)
        for (; child <= total; child *= 2)
        {
            if (child + 1 <= total && (*((*heap) + child)).Value > (*((*heap) + child + 1)).Value)       //找到子节点中较小的节点
            {
                child++;
            }
            if (temp.Value <= (*((*heap) + child)).Value)       //如果父节点小于等于最小的子节点，堆已经满足性质，跳出循环
            { // 这里由<修改为<=,对和父节点相等的子节点不进行交换
                break;
            }
            *((*heap) + parent) = *((*heap) + child);
            parent = child;
        }
        *((*heap) + parent) = temp;     //给父节点位置赋值，*heap为堆的首地址，再加上父节点的位置，即为父节点的地址，之后*取值，即为父节点的值
    }
}

// **
//  * 替换堆中的最小元素并维护最小堆性质
//  * @param heap 最小堆
//  * @param newValue 替换的新值
//  * @param newState 新状态
//  */
void replace_min_heap_element(element **heap, ElementType newValue, int newState, int newT3_State)
{
    // 替换堆顶元素为新值
    (*heap)[1].Value = newValue;
    (*heap)[1].State = newState;
    (*heap)[1].T3_State = newT3_State;

    int total = (*heap)[0].Value; // 总元素个数
    int parent = 1;
    int child = 2;

    // 自顶向下调整堆
    while (child <= total)
    {
        // 找到左右子节点中较小的节点
        if (child + 1 <= total && (*heap)[child].Value > (*heap)[child + 1].Value)
        {
            child++;
        }

        // 如果父节点小于等于最小的子节点，堆已经满足性质
        if ((*heap)[parent].Value <= (*heap)[child].Value)
        {
            break;
        }

        // 交换父节点和子节点
        element temp = (*heap)[parent];
        (*heap)[parent] = (*heap)[child];
        (*heap)[child] = temp;

        parent = child;
        child *= 2;
    }
}

Status generate_state_heap(ElementType probability_i, int i, element **heap_total, int *changestate, int T3_State)
{
    element *num = *heap_total;      // 存储树个数的位置
    element *position = *heap_total; // 当前节点到树的那个位置
    position = position + i + 1;     // positi每次向后移i+1个位置，之前已经计算了i个点
    if (i < MAX_ELEMENTS - 1)
    { // 树未满时
        insert_binary_tree(position, probability_i, i, T3_State);
        (*num).Value++;
        // 输出插入树后的结果,测试用
        //  printf("total number is %d ,new element is %lf  ,state is %d  ,T3 State is%d\n",(int)((*num).Value),(*position).Value,(*position).State,(*position).T3_State);
        return 0;
    }
    else if (i == MAX_ELEMENTS - 1)
    { // 树刚满了
        insert_binary_tree(position, probability_i, i, T3_State);
        (*num).Value++;
        // 输出插入树后的结果,测试用
        //  printf("\ntotal number is %d  ,new element is %lf  ,state is %d  \n",(int)((*num).Value),(*position).Value,(*position).State);
        create_min_heap(heap_total);
        // 输出排序后的结果,测试用
        //  print_heap_element(heap_total);
        return 0;
    }
    else
    { // 树满，且下个元素大于最小值时，进行替换，并输出1
        if (probability_i > (*heap_total)[1].Value)
        {
            // printf("\nChange:new element value is %lf state is %d T3 State is%d,Total min element is %lf  state is %d T3 State is%d \n",probability_i,i,T3_State,(*heap_total)[1].Value,(*heap_total)[1].State,(*heap_total)[1].T3_State);
            *changestate = (*heap_total)[1].State; // 被替换的state
            // (*heap_total)[1].Value=probability_i;
            // (*heap_total)[1].State=i;
            // create_max_heap(heap_total);
            replace_min_heap_element(heap_total, probability_i, i, T3_State);
            // 输出排序后的结果,测试用
            //  print_heap_element(heap_total);
            return 1;
        }
        // 树满，且下个元素不大于最小值时，不进行替换，并输出2
        else
        {
            // printf("Keep:new element value is %lf  state is %d  ,Total min element is %lf  state is %d  \n",probability_i,i,(*heap_total)[1].Value,(*heap_total)[1].State);
            return 2;
        }
    }
}

// void chaneg_T2_i(int changestate,int changgelabel,int i,int statei_value){

//     if (!changgelabel){         //=0时说明还没满，尚可以直接给T2赋值
//         snvT2[i] =  statei_value;
//         // printf("input state : %d , Arc: %d -> %d \n",insert,statei_value,i);
//         }
//     else if(changgelabel==1){   //=1时说明已经满了，且需要替换状态i，其对应的状态为statei_value，changgelabel是被替换的状态
//         snvT2[i] =  statei_value;
//         snvT2[changestate] =  -1;
//         printf("change stae %d value to %d ,input state %d value to %d\n",snvT2[changestate],changestate,snvT2[i],i);
//         }
//     else{                       //=2时说明已经满了，且不需要替换
//         snvT2[i] =  -1;
//         printf("input stae %d value to %d\n",snvT2[i],i);
//         }
// }

// void  Change_T3_element(element **heap,int snvT3[]){
//     for (int i = 1; i<MAX_ELEMENTS+1; i++) {
//         snvT3[(*((*heap) + i)).State]=(*((*heap) + i)).T3_State;
//         // printf("heap element is %lf,state is %d,T3 state is %d\n",(*((*heap) + i)).Value,(*((*heap) + i)).State,(*((*heap) + i)).T3_State);}
// }
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void InitOutputpath()
{

    int i = 0;

    for (i = 0; i < obserRouteLEN; i++)
    {

        snvOutPutPath[i] = -1;
    }
}

void InitQueue(Queue *Q) // 初始化队列
{
    // 为队列分配存储空间，为前驱后继队列
    Q->preNode = (int *)malloc(sizeof(int) * obserRouteLEN); // 给指针变量分配长度为观测路径长的整型存储空间。
    Q->sucNode = (int *)malloc(sizeof(int) * obserRouteLEN);
    if ((Q->preNode != NULL) && (Q->sucNode != NULL))
        Q->front = Q->rear = 0;
    // 初始时队列为空，头指针和尾指针都指向0位置
    else
    {
        printf("allocation is failed !!");
        return;
    }
}

void EnQueue(Queue *Q, int preside, int sucside) // 进队
{
    // 判断循环队列是否已满
    if (((Q->rear + 1) % obserRouteLEN) == Q->front) // 判满条件是尾指针的下一个位置（通过取模运算 (Q->rear + 1) % obserRouteLEN）等于头指针的位置（Q->front）。
        return;
    // 队列未满，将数据入队
    Q->preNode[Q->rear] = preside; // 前驱边界入队
    Q->sucNode[Q->rear] = sucside; // 后继边界入队
    // 更改尾指针的指向
    Q->rear = (Q->rear + 1) % obserRouteLEN;
}

Status DeQueue(Queue *Q, int *preside, int *sucside) // 出队
{
    // 判断循环队列是否为空
    if (Q->front == Q->rear)
        return 0;
    // 出队前将前驱和后继节点的值保存
    *preside = Q->preNode[Q->front];
    *sucside = Q->sucNode[Q->front];
    // 如果非空，实现可循环出队
    Q->front = (Q->front + 1) % obserRouteLEN;
    return 1;
}
void ShowQueue(Queue *Q)
{
    // 遍历循环队列中的元素，并将数据打印
    for (int i = Q->front; i != Q->rear; i = (i + 1) % obserRouteLEN)
    {
        printf("(%d->%d), ", Q->preNode[i], Q->sucNode[i]);
        // 此操作是为了实现循环遍历
    }
    printf("\n");
}

int Length(Queue *Q)
{
    // 计算尾指针位置与头指针位置的差距
    int len = Q->rear - Q->front;
    // 如果为正数，那么len就是队列的长度；如果为负数，那么obserRouteLEN+len才是队列的长度
    len = (len > 0) ? len : obserRouteLEN + len;
    return len;
}

void max_score(ElementType arr[], int arr_count, int *max_temp_no, ElementType *max_temp) // 求求队列概率最大值和下标
{

    if (NULL == arr)
        return;
    int max_no = 0;           // 概率值的No号
    ElementType max = arr[0]; // 概率值大小
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
    ElementType *tmpt = (ElementType *)malloc(sizeof(ElementType) * K_STATE * K_STATE);

    if (tmpt)
    {
        // ElementType tmpt[K_STATE*K_STATE]= {0};//大数组下溢出
        //  FILE* fpRead = fopen("C:/Users/DELL/Documents/work/source/Nvvviter/dataset/matrixA_500plus500_fixed.txt", "rb+");//matrixA or matrixB
        // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\A.txt", "rb+"); // matrixA or matrixB 程序成功打开文件后，fopen()将返回文件指针。文件指针的类型是FILE
        // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\A_512.txt", "rb+"); // matrixA or matrixB 程序成功打开文件后，fopen()将返回文件指针。文件指针的类型是FILE
        // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\new512data\\A_512.txt", "rb+"); // matrixA or matrixB 程序成功打开文件后，fopen()将返回文件指针。文件指针的类型是FILE
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
        // static ElementType tmpt[K_STATE*T_STATE]= {0};
        //  FILE* fpRead = fopen("C:/Users/DELL/Documents/work/source/Nvvviter/dataset/matrixB_500plus500_fixed.txt", "rb+");//matrixA or matrixB
        // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\B_512.txt", "rb+"); // matrixA or matrixB
        // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\new512data\\B_512.txt", "rb+"); // matrixA or matrixB
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
    // ElementType tmpt[K_STATE]= {0};
    //  FILE* fpRead = fopen("C:/Users/DELL/Documents/work/source/Nvvviter/dataset/matrixPi500_fixed.txt", "rb+");
    // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\Pi_512.txt", "rb+");
    // FILE *fpRead = fopen("E:\\Code\\Ccode\\test\\Nvviter\\new512data\\Pi_512.txt", "rb+");
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
    int tmpt[obserRouteLEN] = {0};
    // FILE* fpRead3 = fopen("C:/Users/DELL/Documents/work/source/Nvvviter/dataset/obRoute5000K500.txt", "rb+");
    // FILE *fpRead3 = fopen("E:\\Code\\Ccode\\test\\Nvviter\\ob_512.txt", "rb+");
    // FILE *fpRead3 = fopen("E:\\Code\\Ccode\\test\\Nvviter\\new512data\\ob_512.txt", "rb+");
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
    if (qsucNode - qpreNode < 2*N) // 后继-前驱小于6，结束
        return;
    int i;
    ElementType scorearr[MAX_ELEMENTS] = {0}; // 存储所有概率值，用于求最大概率
    int T_obs = Obroute[qpreNode];                                // 观测序列初值，第一列观测值   *当前序列起始的观测值
    int midpoint[N+1], gap_length = (qsucNode - qpreNode) / N, gap_extra = (qsucNode - qpreNode) % N;
    midpoint[0] = qpreNode, midpoint[N] = qsucNode; //0无意义
    for(int i = 1; i < N; ++i)
    {
        midpoint[i] = midpoint[i-1]+gap_length;
        if(gap_extra > 0) gap_extra--, midpoint[i]++;
    }
    element** heap_pre_matrix = (element**)malloc(sizeof(element*)*N);
    element** heap_total_matrix = (element**)malloc(sizeof(element*)*N);
    element *heap_pre[N];
    element *heap_total[N];
    for(int i = 0; i < N; ++i)
    {
        heap_pre_matrix[i] = (element*)malloc(sizeof(element)*(MAX_ELEMENTS + 1));
        heap_total_matrix[i] = (element*)malloc(sizeof(element)*(MAX_ELEMENTS + 1));
        heap_pre[i] = heap_pre_matrix[i];
        heap_total[i] = heap_total_matrix[i];
        initial_heap_element(heap_total[i]);
        initial_heap_element(heap_pre[i]);
    }
    for (i = 0; i < K_STATE; i++)
    {
        ElementType probability_i = (qpreNode == 0 ? log(pi[i]) : log(A[snvOutPutPath[qpreNode - 1]][i]))+ log(B[i][T_obs]); // 初始化T1,用log避免数据下溢出
        for(int j = 0; j < N; ++j)
        {
            int changestate;
            generate_state_heap(probability_i, i, &(heap_total[j]), &changestate, -1);
        }
    }
    for(int i = 0; i < N; ++i)
    {
        element *heap_tem = heap_pre[i];
        heap_pre[i] = heap_total[i];
        heap_total[i] = heap_tem;
        initial_heap_element(heap_total[i]);
    }

    for (int j = qpreNode + 1; j < qsucNode + 1; j++)
    {
        T_obs = Obroute[j];           // 更新观测序列当前值
        for (i = 0; i < K_STATE; i++) // 行
        {
            for (int k = 0; k < MAX_ELEMENTS; k++)
            {
                int preState = heap_pre[1][k + 1].State;
                ElementType preValue = heap_pre[1][k + 1].Value;
                scorearr[k] = preValue + log(A[preState][i]) + log(B[i][T_obs]); // 求出所有概率值,用log避免数值下溢出,其中T1已经为log
            }
            int scoreNo = 0;
            ElementType scoreMax = 0;
            int *p = &scoreNo;
            ElementType *q = &scoreMax;
            max_score(scorearr, MAX_ELEMENTS, p, q);
            ElementType probability_i = scoreMax; // 下溢出
            for(int p = 1; p < N; ++p)
            {
                int changestate;
                generate_state_heap(probability_i, i, &(heap_total[p]), &changestate,
                    j > midpoint[p]+1 ? heap_pre[p][scoreNo + 1].T3_State : heap_pre[p][scoreNo + 1].State);
            }
        }
        for(int i = 0; i < N; ++i)
        {
            element *heap_tem = heap_pre[i];
            heap_pre[i] = heap_total[i];
            heap_total[i] = heap_tem;
            initial_heap_element(heap_total[i]);
        }
    }
    // 此时保存了中间节点路径和最后节点的概率
    // 求最后一个节点的路径,并根据此回溯路径中点，输出
    if (qpreNode == 0 && qsucNode == obserRouteLEN-1)
    {
        ElementType last_max = heap_pre[1][1].Value; // 回溯处本身为负值,不要用0比较
        int last_max_no = 0;
        // int last_max_state = heap_pre[1].State;
        for (int i = MAX_ELEMENTS / 2+1; i < MAX_ELEMENTS; i++)
        {
            if (last_max < heap_pre[1][i + 1].Value)
            {
                last_max_no=i;
                last_max = heap_pre[1][i + 1].Value;
            }
        }
        ElementType value_test=heap_pre[1][last_max_no+1].Value;
        snvOutPutPath[qsucNode] = heap_pre[1][last_max_no+1].State; // 倒数第一个节点值 //?存入最大值
        for(int p = 1; p < N; ++p)
        {
            snvOutPutPath[midpoint[p]] = heap_pre[p][last_max_no+1].T3_State;
        }
        
        printf("总内存占用大小(不算输出)：%zu 字节\n", (sizeof(ElementType)+2*sizeof(int))*(MAX_ELEMENTS + 1)*N*2+sizeof(midpoint)+(sizeof(scorearr)+sizeof(last_max)+sizeof(last_max_no)+sizeof(T_obs))*MAX_THREADS);
        printf("总内存占用大小：%zu 字节\n", (sizeof(ElementType)+2*sizeof(int))*(MAX_ELEMENTS + 1)*N*2+sizeof(midpoint)+sizeof(snvOutPutPath)+(sizeof(scorearr)+sizeof(last_max)+sizeof(last_max_no)+sizeof(T_obs))*MAX_THREADS);
    }
    else // 其余都执行
    {
        int connt = snvOutPutPath[qsucNode]; // 找到当前最末点的状态值
        for(int p = 1; p < N; ++p)
        {
            snvOutPutPath[midpoint[p]] = Find_T3_State(&(heap_pre[p]), connt);
        }
    }

    for(int i = 0; i < N; ++i)
    {
        free(heap_pre_matrix[i]);
        free(heap_total_matrix[i]);
    }
    free(heap_pre_matrix);
    free(heap_total_matrix);

    EnQueue(Q,qpreNode,midpoint[1]);
    for(int p = 1; p < N; ++p)
    {
        EnQueue(Q,midpoint[p]+1,midpoint[p+1]);
    }
}

void Viterbifirst(int qpreNode, int qsucNode) // 大于3时的情况
{

    int TobsLenth = obserRouteLEN;
    if (qsucNode - qpreNode < 0) // 后继-前驱小于0，结束
        return;
    // (TobsLenth <= 1)
    //  return;
    int i;
    ElementType scorearr[MAX_ELEMENTS] = {0}; // 存储所有概率值，用于求最大概率
    int T_obs;                                // 观测序列初值，第一列观测值   *当前序列起始的观测值
    int midpoint = (qpreNode + qsucNode) / 2; // path的中间节点,运行结束后得到当前Path[midpoint]
    // snvOutPutPath[midpoint - 1] = -1; // init path[midpoint-1]
    element heap_1[MAX_ELEMENTS + 1];
    element heap_2[MAX_ELEMENTS + 1];
    element *heap_pre = heap_1;
    element *heap_total = heap_2;
    for (int j = qpreNode; j < qsucNode + 1; j++)
    {
        if (j == qpreNode)
        {
            if (qpreNode == 0) // 仅当第一列的T表格初始化，当头节点为0时
            {
                T_obs = Obroute[qpreNode];

                initial_heap_element(heap_total);
                initial_heap_element(heap_pre);

                for (i = 0; i < K_STATE; i++)
                {
                    double B_data=log(B[i][T_obs]);
                    double pi_data=log(pi[i]);
                    ElementType probability_i = log(pi[i]) + log(B[i][T_obs]); // 初始化T1,用log避免数据下溢出
                    // if (i<MAX_ELEMENTS-1){
                    //     insert_binary_tree(position,probability_i ,i);
                    //     num->Value++;
                    //     printf("total number is %d ,new element is %lf  ,state is %d  \n",num->Value,position->Value,position->State);
                    //     position++;
                    //     }
                    // else if (i==MAX_ELEMENTS-1){
                    //     insert_binary_tree(position,probability_i ,i);
                    //     num->Value++;
                    //     printf("total number is %d  ,new element is %lf  ,state is %d  \n",num->Value,position->Value,position->State);
                    //     create_max_heap(heap_total);
                    //     //输出排序后的结果
                    //     print_heap_element(heap_total);
                    //     }
                    // else{
                    //     if (probability_i>heap_total[1].Value){
                    //         printf("\nChange:new element value is %lf state is %d  ,Total min element is %lf  state is %d  \n",probability_i,i,heap_total[1].Value,heap_total[1].State);
                    //         heap_total[1].Value=probability_i;
                    //         heap_total[1].State=i;
                    //         create_max_heap(heap_total);
                    //         print_heap_element(heap_total);
                    //         }
                    //         // 隐去
                    //     else{
                    //         printf("Keep:new element value is %lf  state is %d  ,Total min element is %lf  state is %d  \n",probability_i,i,heap_total[1].Value,heap_total[1].State);
                    //         }
                    //     }
                    int changestate;
                    int changgelabel = generate_state_heap(probability_i, i, &heap_total, &changestate, -1);
                    // chaneg_T2_i(changestate,changgelabel,i,-1);
                    // snvT3[i] = -1;                            // 初始化T3
                }

                element *heap_tem = heap_pre;
                heap_pre = heap_total;
                heap_total = heap_tem;
                initial_heap_element(heap_total);


                // 输出排序后的结果,测试用
                //  print_heap_element(&heap_pre);
                //  snvT1[i] = log(pi[i]) + log(B[i][T_obs]); // 初始化T1,用log避免数据下溢出
                // printf("over:\n");
            }
            else // 其他情况初始一列,若不是初始的列，说明此处是之前的mid列，因此直接找到该列的对应状态，其他初始为0
            {
                initial_heap_element(heap_total);
                initial_heap_element(heap_pre);

                T_obs = Obroute[qpreNode];                    // 更新观测序列当前值
                int State_temp = snvOutPutPath[qpreNode - 1]; //  当前起始对应的状态值
                for (i = 0; i < K_STATE; i++)                 // 行，对于已经知道的起始元素的起始点（i）时，直接使用第i对应的概率进行后续状态的赋值
                {
                    // for (int k = 0; k < K_STATE; k++)
                    // {
                    //     scorearr[k] = snvT1[k] + log(A[k][i]) + log(B[i][T_obs]); // 求出所有概率值,用log避免数值下溢出,其中T1已经为log
                    // }
                    // int scoreNo = 0;
                    // ElementType scoreMax = 0;
                    // int* p = &scoreNo;
                    // ElementType* q = &scoreMax;
                    // max_score(scorearr, K_STATE, p, q); //T2存下标，T1是概率
                    // snvTtmp[i] = scoreMax; // 下溢出
                    // snvT2[i] = scoreNo;//前半不需要存路径
                    ElementType probability_i = log(A[State_temp][i]) + log(B[i][T_obs]);
                    int changestate;
                    int changgelabel = generate_state_heap(probability_i, i, &heap_total, &changestate, -1);
                    // chaneg_T2_i(changestate,changgelabel,i,State_temp);

                    // if (!changgelabel){
                    //     snvT2[i] =  State_temp;}
                    // else if(changgelabel==1)
                    // {
                    //     snvT2[i] =  State_temp;
                    //     snvT2[changestate] =  -1;
                    //     print("change stae %d value to %d ,input state %d value to %d\n",changestate,snvT2[changestate],i,snvT2[i]);
                    // }
                    // else{
                    //     snvT2[i] =  -1;
                    //     print("input stae %d value to %d\n",i,snvT2[i]);
                    // }
                    // snvT3[i] = -1;                            // 初始化T3
                }
                element *heap_tem = heap_pre;
                heap_pre = heap_total;
                heap_total = heap_tem;
                initial_heap_element(heap_total);


                // 输出排序后的结果,测试用
                //  print_heap_element(&heap_pre);
                //  printf("over:\n");
            }
        }
        // 第pre+1列到第midpoint列              0-4
        else if (j < midpoint + 1)
        {
            T_obs = Obroute[j];           // 更新观测序列当前值
            for (i = 0; i < K_STATE; i++) // 行
            {
                for (int k = 0; k < MAX_ELEMENTS; k++)
                {
                    int preState = heap_pre[k + 1].State;
                    ElementType preValue = heap_pre[k + 1].Value;
                    ElementType A_value = log(A[preState][i]);
                    ElementType B_value = log(B[i][T_obs]);
                    scorearr[k] = preValue + log(A[preState][i]) + log(B[i][T_obs]); // 求出所有概率值,用log避免数值下溢出,其中T1已经为log
                }
                int scoreNo = 0;
                ElementType scoreMax = 0;
                int *p = &scoreNo;
                ElementType *q = &scoreMax;
                max_score(scorearr, MAX_ELEMENTS, p, q);
                ElementType probability_i = scoreMax; // 下溢出
                int T3_state_test=heap_pre[scoreNo + 1].State;
                int changestate;
                int changgelabel = generate_state_heap(probability_i, i, &heap_total, &changestate, heap_pre[scoreNo + 1].State);
                // chaneg_T2_i(changestate,changgelabel,i,-1);//前半不需要存路径
                // snvT2[i] = scoreNo;//前半不需要存路径
            }
            // printf("over:\n");
            element *heap_tem = heap_pre;
            heap_pre = heap_total;
            heap_total = heap_tem;
            initial_heap_element(heap_total);



            // 输出排序后的结果,测试用
            //  print_heap_element(&heap_pre);
        }
        // 第midpoint+1列                               5
        else if (j == midpoint + 1)
        {
            T_obs = Obroute[midpoint + 1];
            for (i = 0; i < K_STATE; i++) // 行
            {
                for (int k = 0; k < MAX_ELEMENTS; k++)
                {
                    int preState = heap_pre[k + 1].State; // 第0行为存储信息，因此为k+1
                    int pre_T3_State = heap_pre[k + 1].T3_State; // 第0行为存储信息，因此为k+1
                    ElementType A_value = log(A[preState][i]);
                    ElementType B_value = log(B[i][T_obs]);
                    ElementType preValue = heap_pre[k + 1].Value;
                    scorearr[k] = preValue + log(A[preState][i]) + log(B[i][T_obs]); // 求出所有概率值,用log避免数值下溢出,其中T1已经为log
                }
                int scoreNo = 0;
                ElementType scoreMax = 0;
                int *p = &scoreNo;
                ElementType *q = &scoreMax;
                max_score(scorearr, MAX_ELEMENTS, p, q);
                ElementType probability_i = scoreMax; // 下溢出
                int changestate;
                int changgelabel = generate_state_heap(probability_i, i, &heap_total, &changestate, heap_pre[scoreNo + 1].State);
                // chaneg_T2_i(changestate,changgelabel,i,heap_pre[scoreNo+1].State);//后半路径开始保存T2;与堆内操作合并
            }
            // printf("over:\n");
            element *heap_tem = heap_pre;
            heap_pre = heap_total;
            heap_total = heap_tem;
            initial_heap_element(heap_total);


            // 输出排序后的结果,测试用
            //  print_heap_element(&heap_pre);
            //  Change_T3_element(&heap_pre,snvT3);

            // for (i = 0; i < K_STATE; i++){ // 将暂存概率值拷贝，并更新T3列，T2和T3的迭代更新
            //         snvT3[i] = snvT2[i];
            // }
        }
        // 第midpoint+2列到第qsucNode列         6-8
        else 
        {
            // if(j=qsucNode)               //test
            //     printf("over:\n");
            T_obs = Obroute[j];           // 更新观测序列当前值
            for (i = 0; i < K_STATE; i++) // 行
            {
                for (int k = 0; k < MAX_ELEMENTS; k++)
                {
                    int preState = heap_pre[k + 1].State; // 第0行为存储信息，因此为k+1
                    ElementType preValue = heap_pre[k + 1].Value;
                    ElementType A_value = log(A[preState][i]);
                    ElementType B_value = log(B[i][T_obs]);
                    scorearr[k] = preValue + log(A[preState][i]) + log(B[i][T_obs]); // 求出所有概率值,用log避免数值下溢出,其中T1已经为log
                }
                int scoreNo = 0;
                ElementType scoreMax = 0;
                int *p = &scoreNo;
                ElementType *q = &scoreMax;
                max_score(scorearr, MAX_ELEMENTS, p, q);
                ElementType probability_i = scoreMax; // 下溢出
                int changestate;
                int T3_state =  heap_pre[scoreNo + 1].T3_State;            // Find_T3_State(&heap_pre, heap_pre[scoreNo + 1].State);
                int changgelabel = generate_state_heap(probability_i, i, &heap_total, &changestate, T3_state); // 与堆内操作合并
                // chaneg_T2_i(changestate,changgelabel,i,snvT3[heap_pre[scoreNo+1].State]);//后半路径开始保存T2
            } // 将最大的值的下标赋值给

            // if(j=1023)
            //     printf("over:\n");

            // printf("over:\n");
            element *heap_tem = heap_pre;
            heap_pre = heap_total;
            heap_total = heap_tem;
            initial_heap_element(heap_total);




            // 输出排序后的结果,测试用
            //  print_heap_element(&heap_pre);
            //  Change_T3_element(&heap_pre,snvT3);

            // for (i = 0; i < K_STATE; i++) // 将暂存概率值拷贝，并更新T3列，T2和T3的迭代更新
            // {
            //     // snvT1[i] = snvTtmp[i];
            //     snvT3[i] = snvT2[i];
            // }
        }
    }
    // 此时保存了中间节点路径和最后节点的概率
    // 求最后一个节点的路径,并根据此回溯路径中点，输出
    int mi = (obserRouteLEN - 1) / 2;
    if (midpoint == mi) // 仅第一次执行
    {
        ElementType last_max = heap_pre[1].Value; // 回溯处本身为负值,不要用0比较
        int last_max_no = 0;
        // int last_max_state = heap_pre[1].State;
        for (int i = MAX_ELEMENTS / 2+1; i < MAX_ELEMENTS; i++)
        {
            if (last_max < heap_pre[i + 1].Value)
            {
                last_max_no=i;
                last_max = heap_pre[i + 1].Value;
                // last_max_state = heap_pre[i + 1].State;
            }
        }
        ElementType value_test=heap_pre[last_max_no+1].Value;
        int connttest = heap_pre[last_max_no+1].State; // 找到当前最末点的状态值
        int T3_Statetest = heap_pre[last_max_no+1].T3_State; // 找到当前最末点的T3状态值

        snvOutPutPath[qsucNode] = heap_pre[last_max_no+1].State; // 倒数第一个节点值 //?存入最大值

        snvOutPutPath[midpoint] = heap_pre[last_max_no+1].T3_State;    //Find_T3_State(&heap_pre, last_max_no);
        // snvOutPutPath[midpoint] = snvT3[last_max_no]; // 第midpoint个节点值，最后T2与T3是一样的

        // printf("heap_total 占用的内存大小: %zu 字节\n", sizeof(heap_1));
        // printf("heap_pre 占用的内存大小: %zu 字节\n", sizeof(heap_2));
        // printf("scorearr 占用的内存大小: %zu 字节\n", sizeof(scorearr));
        // printf("snvOutPutPath占用的内存大小: %zu 字节\n", sizeof(snvOutPutPath));
        // printf("last_max 占用的内存大小: %zu 字节\n", sizeof(last_max));
        // printf("last_max_no 占用的内存大小: %zu 字节\n", sizeof(last_max_no));
        // printf("T_obs 占用的内存大小: %zu 字节\n", sizeof(T_obs));
        // printf("总内存占用大小(不算输出)：%zu 字节\n", sizeof(heap_1) + sizeof(heap_2) + sizeof(scorearr) + sizeof(last_max) + sizeof(last_max_no) + sizeof(T_obs));
        // printf("总内存占用大小：%zu 字节\n", sizeof(heap_1) + sizeof(heap_2) + sizeof(scorearr) + sizeof(snvOutPutPath) + sizeof(last_max) + sizeof(last_max_no) + sizeof(T_obs));


        printf("\n log lastmax %lf", last_max);  // log值测试
        printf("\n lastmax %lf", exp(last_max)); // 最大路径概率值
    }
    else // 其余都执行
    {
        // if(qpreNode==320&&qsucNode==383)
        // {
        //     printf("over:\n");
        // }
        int connt = snvOutPutPath[qsucNode]; // 找到当前最末点的状态值
        snvOutPutPath[midpoint] = Find_T3_State(&heap_pre, connt);
        // snvOutPutPath[midpoint] = snvT3[connt]; //当前路径范围内输出的，将此时T3到该状态点的值进行输出
    }

}

void printRoute() // 输出路径
{

    printf("\n总状态数为%d,剪枝剩余状态为%d,隐状态最可能路径长度为%d ,路径为:\n",K_STATE,MAX_ELEMENTS,obserRouteLEN);

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
            Viterbifirst(pre, suc);
            EnterCriticalSection(&pool->lock); // 进入临界区
            EnQueue(&(pool->Q), pre, midpoint);
            LeaveCriticalSection(&pool->lock); // 离开临界区
        }           
        else //pre=suc+1,pre = suc
        {
            Viterbifirst(pre, suc);
        }
        WakeAllConditionVariable(&pool->pool_wake);
    }
    return 0;
}
// 初始化线程池
void ThreadPoolInit(ThreadPool* pool) {
    InitQueue(&(pool->Q)); 
    pool->task_count = 0;
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
void tracebackroute() // space主体        调用刚刚主体的代码
{
    int TobsLenth = obserRouteLEN; // 观察路径长
    int pre = 0;             // 当前处理的路径的起始
    int suc = TobsLenth - 1; // 结束点
    int *qpre = &pre;        // 声明指针qpre和qsuc分别指向pre和suc，用于在函数中更新起始和结束点。
    int *qsuc = &suc;
    ThreadPool pool;
    ThreadPoolInit(&pool);
    // printf("队列Q占用的内存大小: %zu 字节\n", sizeof(Q));
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
    printf("task:%d\n",pool.task_count);
}

int exmpleSNVviter()
{

    int excu_i = 0;
    char* str_MatrixA="E:\\Code\\Ccode\\test\\Nvviter\\new512data\\A_K3965_T256_prob0.075.txt";
    char* str_MatrixB="E:\\Code\\Ccode\\test\\Nvviter\\new512data\\B_K3965_T256_prob0.075.txt";
    char* str_MatrixPI="E:\\Code\\Ccode\\test\\Nvviter\\new512data\\Pi_K3965_T256_prob0.075.txt";
    char* str_ObRoute="E:\\Code\\Ccode\\test\\Nvviter\\new512data\\ob_K3965_T256_prob0.075.txt";
    InitOutputpath();
    InitMatrixA(str_MatrixA);
    InitMatrixB(str_MatrixB);
    InitMatrixPI(str_MatrixPI);
    InitObRoute(str_ObRoute);
    QueryPerformanceFrequency(&tc); // time 算执行时间
    QueryPerformanceCounter(&t1);   // time
    tracebackroute();
    QueryPerformanceCounter(&t2);                                                                 // time
    printf("\nNvvVITERBI_time:%lf s", (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart); // time
    printRoute();                                                                                 // 输出路径
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
