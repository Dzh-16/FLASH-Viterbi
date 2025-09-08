#define _POSIX_C_SOURCE 199309L
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<time.h>

//parameter set 
#define K_STATE 2048
#define T_STATE 50
#define obserRouteLEN 512
const float prob = 0.253;
const char data_path[] = "./data/";

typedef enum {true=1, false=0}bool;
typedef float ElementType;
const ElementType ElementTypeNegMin = -FLT_MAX;
typedef struct {
    ElementType Pi[K_STATE];
    ElementType A[K_STATE][K_STATE];
    ElementType B[K_STATE][T_STATE];
    int Obroute[obserRouteLEN];

    //ans
    int Ans[obserRouteLEN];
    int memory_bytes;
} VIT;

//queue
typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *front;
    Node *rear;
} LinkedQueue;

char* getAddress(char *stype)
{
    static char path[100];
    snprintf(path, sizeof(path), "%s%s_K%d_T%d_prob%.3f.txt", 
             data_path ,stype, K_STATE, obserRouteLEN, prob);
    return path;
}

// 初始化队列
void initQueue(LinkedQueue *q) {
    q->front = NULL;
    q->rear = NULL;
}

// 检查队列是否为空
bool isEmpty(LinkedQueue *q) {
    return (q->front == NULL);
}

// 入队操作
void enqueue(LinkedQueue *q, int value) {
    Node *newNode = (Node *)malloc(sizeof(Node));
    if (newNode == NULL) {
        printf("内存分配失败！\n");
        exit(EXIT_FAILURE);
    }
    newNode->data = value;
    newNode->next = NULL;

    if (isEmpty(q)) {
        // 如果是空队列，新节点既是头也是尾
        q->front = newNode;
        q->rear = newNode;
    } else {
        // 非空队列，将新节点链接到队尾，并更新rear
        q->rear->next = newNode;
        q->rear = newNode;
    }
}

// 出队操作
bool dequeue(LinkedQueue *q, int *value) {
    if (isEmpty(q)) {
        printf("队列为空，无法出队\n");
        return false;
    }
    Node *temp = q->front;
    *value = temp->data;

    q->front = q->front->next; // 移动front指针
    free(temp); // 释放原队头节点的内存

    // 如果出队后队列为空，需要将rear也置为NULL
    if (q->front == NULL) {
        q->rear = NULL;
    }
    return true;
}

// 查看队头元素
bool getFront(LinkedQueue *q, int *value) {
    if (isEmpty(q)) {
        printf("队列为空\n");
        return false;
    }
    *value = q->front->data;
    return true;
}


ElementType emax(ElementType a, ElementType b)
{
    return a < b ? b : a;
}

void InitElement(VIT *vit,char *stype)
{
    int iLen, jLen;
    if(strcmp(stype,"A") == 0)
        iLen = K_STATE, jLen = K_STATE;
    else if(strcmp(stype,"B") == 0)
        iLen = K_STATE, jLen = T_STATE;
    else if(strcmp(stype,"Pi") == 0)
        iLen = K_STATE, jLen = -1;
    else if(strcmp(stype,"ob") == 0)
        iLen = obserRouteLEN, jLen = -1;
    else perror("type error in void InitElement()");

    char *path = getAddress(stype);

    FILE *fpRead = fopen(path,"rb+");
    if(fpRead == NULL) perror("Error opening file in void InitElement()");
    
    if(strcmp(stype,"ob") == 0)
    {
        for(int i = 0; i < iLen; ++i)
            fscanf(fpRead, "%d", &(vit->Obroute[i]));
        fclose(fpRead);
        return;
    }

    for(int i = 0; i < iLen; ++i)
    {
        if(strcmp(stype,"Pi") == 0)
            fscanf(fpRead, "%f", &(vit->Pi[i]));
        else for(int j = 0; j < jLen; ++j)
        {
            if(strcmp(stype,"A") == 0)
                fscanf(fpRead, "%f", &(vit->A[i][j]));
            else
                fscanf(fpRead, "%f", &(vit->B[i][j]));
        }
    }
    fclose(fpRead);
}

void InitVitAns(VIT* vit)
{
    memset(vit->Ans, 0, sizeof(vit->Ans));
    vit->memory_bytes = 0;
}

VIT* create_vit() {
    VIT* vit = (VIT*)malloc(sizeof(VIT));
    if(vit == NULL) perror("malloc failed in VIT* create_vit()");

    InitElement(vit,"A");
    InitElement(vit,"B");
    InitElement(vit,"Pi");
    InitElement(vit,"ob");
    
    InitVitAns(vit);
    return vit;
}

void delete_vit(VIT *vit)
{
    if(vit != NULL)
    {
        free(vit);
    }
}

void printAns(VIT *vit)
{
    printf("path: [");
    for(int i = 0; i < obserRouteLEN; ++i)
        printf("%d ",vit->Ans[i]);
    puts("]");
    printf("memory: %d\n",vit->memory_bytes);
}

#define initT1(x) log(vit->Pi[x]) + log(vit->B[x][vit->Obroute[0]])

typedef struct
{
    int x,y;
}MEDIANS;

int BFS_ancestors_middlepath(VIT *vit,int K,int source,int *indices,int *visited,int b)
{
    LinkedQueue queue;
    initQueue(&queue);
    enqueue(&queue, source);
    enqueue(&queue, -1);

    int level = 0;
    int num = 2, score = 2;

    while(!isEmpty(&queue) && level < b)
    {
        int s; dequeue(&queue, &s); num--;
        if(s == -1)
        {
            level += 1;
            enqueue(&queue, -1); num++;
        }
        else
        {
            for(int i = 0; i < K; ++i)
            {
                ElementType x = vit->A[indices[i]][s];
                if(x > 0)
                {
                    if(!visited[i])
                    {
                        enqueue(&queue, indices[i]);
                        visited[i] = 1; num++;
                    }
                }
            }
        }
        
        if(num > score)
            score = num;
    }

    return score*sizeof(Node) + sizeof(LinkedQueue);
}

int BFS_descendants_middlepath(VIT *vit,int K,int source,int *indices,int *visited,int b)
{
    LinkedQueue queue;
    initQueue(&queue);
    enqueue(&queue, source);
    enqueue(&queue, -1);

    int level = 0;
    int num = 2, score = 2;

    while(!isEmpty(&queue) && level < b)
    {
        int s; dequeue(&queue, &s); num--;
        if(s == -1)
        {
            level += 1;
            enqueue(&queue, -1); num++;
        }
        else
        {
            for(int i = 0; i < K; ++i)
            {
                ElementType x = vit->A[s][indices[i]];
                if(x > 0)
                {
                    if(!visited[i])
                    {
                        enqueue(&queue, indices[i]);
                        visited[i] = 1; num++;
                    }
                }
            }
        }
        
        if(num > score)
            score = num;
    }

    return score*sizeof(Node) + sizeof(LinkedQueue);
}

MEDIANS mp_path[obserRouteLEN];
int mp_path_len = 0;
int initial_state = -1;

int sieve_middlepath(VIT *vit,int *indices, int K,
                      int *Obroute, int T,
                      ElementType *Pi, int isPiNone, int last)
{
    if(initial_state > -1)
    {
        for(int i = 0; i < K; ++i)
        {
            int x = indices[i];
            Pi[i] = (x == initial_state ? 1 : 0);
        }
        isPiNone = 0;
    }

    if(isPiNone)
    {
        ElementType tmp = 1.0/K;
        for(int i = 0; i < K; ++i)
        {
            Pi[i] = tmp;
        }
    }

    int x_a, x_b, memory_t;
{
    ElementType T1[K];
    for(int i = 0; i < K; ++i)
    {
        T1[i] = log(Pi[i]) + log(vit->B[indices[i]][Obroute[0]]);
    }

    MEDIANS previous_medians[K], new_medians[K];
    ElementType new_t1[K];
    memset(previous_medians, -1, sizeof(previous_medians));

    for(int j = 1; j < T; ++j)
    {
        memset(new_medians, -1, sizeof(new_medians));

        for(int i = 0; i < K; ++i)
        {
            ElementType score = ElementTypeNegMin;
            int arg = -1;

            for(int k = 0; k < K; ++k)
            {
                ElementType tmp = T1[k] + log(vit->A[indices[k]][indices[i]]) + log(vit->B[indices[i]][Obroute[j]]);
                if(tmp > score)
                    arg = k, score = tmp;
            }
            new_t1[i] = score;

            if(j == (int)floor(T/2))
            {
                new_medians[i].x = indices[arg];
                new_medians[i].y = indices[i];
            }
            else if (j > (int)floor(T/2))
            {
                new_medians[i] = previous_medians[arg];
            }
        }

        for(int i = 0; i < K; ++i)
        {
            previous_medians[i] = new_medians[i];
            T1[i] = new_t1[i];
        }
    }

    if(last < 0)
    {
        ElementType score = ElementTypeNegMin;
        int arg = -1;
        for(int i = 0; i < K; ++i)
        {
            if(T1[i] > score)
                arg = i, score = T1[i];
        }
        last = arg;
    }

    x_a = new_medians[last].x;
    x_b = new_medians[last].y;
    memory_t = sizeof(T1) + sizeof(previous_medians) + sizeof(new_medians) + sizeof(new_t1);
}
    int N_left = floor(T/2);
    
    int visited[K];

    int memory_left_part = 0;
    if(N_left > 1)
    {
        int y_left[N_left];
        for(int i = 0; i < N_left; ++i)
            y_left[i] = Obroute[i];
        
        memset(visited, 0, sizeof(visited));
        int memory_bfs = BFS_ancestors_middlepath(vit,K,x_a, indices, visited, N_left-1);

        int states_left_indices[K], indlen = 0;
        int index_x_a = -1;
        for(int i = 0; i < K; ++i)
        {
            if(indices[i] == x_a)
            {
                index_x_a = indlen;
                states_left_indices[indlen++] = x_a;
            }
            else if(visited[i])
            {
                states_left_indices[indlen++] = indices[i];
            }
        }

        int K_left = indlen;
        ElementType Pi_left[K_left];
        memory_left_part = 
        sieve_middlepath(vit, states_left_indices, K_left,
                         y_left, N_left, Pi_left, 1, index_x_a);
        
        memory_left_part += memory_bfs + sizeof(states_left_indices)
                            + sizeof(Pi_left) + sizeof(y_left);
    }

    int N_right = T - N_left;
    if(N_right <= 1 && N_left <= 1 && mp_path_len < obserRouteLEN-2 && mp_path_len != 0)
    {
        mp_path[mp_path_len].x = -1;
        ++mp_path_len;
    }
    else
    {
        mp_path[mp_path_len++] = (MEDIANS){x_a, x_b};
    }

    int memory_right_part = 0;
    if(N_right > 1)
    {
        int y_right[N_right];
        for(int i = T-N_right; i < T; ++i)
        {
            y_right[i - (T-N_right)] = Obroute[i];
        }
        memset(visited, 0, sizeof(visited));
        int memory_bfs = BFS_descendants_middlepath(vit,K,x_b, indices, visited, N_right-1);

        int states_right_indices[K], indlen = 0;
        int index_x_b = -1;

        for(int i = 0; i < K; ++i)
        {
            if(indices[i] == x_b)
            {
                index_x_b = indlen;
                states_right_indices[indlen++] = x_b;
            }
            else if(visited[i])
            {
                states_right_indices[indlen++] = indices[i];
            }
        }

        int K_right = indlen;
        ElementType Pi_right[K_right];

        initial_state = x_b;
        memory_right_part =
        sieve_middlepath(vit, states_right_indices, K_right,
                         y_right, N_right, Pi_right, 1, -1);
        memory_right_part += memory_bfs + sizeof(states_right_indices)
                            + sizeof(Pi_right) + sizeof(y_right);
    }

    int memory_rec = (memory_left_part < memory_right_part ? memory_right_part : memory_left_part) + sizeof(visited);
    if( memory_t > memory_rec)
        memory_rec = memory_t;
    return memory_rec;
}

void change_mp_path(VIT * vit)
{
    int len = 0;

    vit->Ans[len++] = mp_path[0].x;
    vit->Ans[len++] = mp_path[0].y;
    int i = 1;
    while(len <= mp_path_len)
    {
        if(mp_path[i].x == -1)
        {
            if(i+1 >= mp_path_len)
                break;
            vit->Ans[len++] = mp_path[i+1].x;
            vit->Ans[len++] = mp_path[i+1].y;
            i++;
        }
        else
        {
            vit->Ans[len++] = mp_path[i].y;
        }
        i++;
    }
}

void calc(VIT *vit)
{
    int indices[K_STATE];
    for(int i = 0; i < K_STATE; ++i)
    {
        indices[i] = i;
    }
    vit->memory_bytes =
    sieve_middlepath(vit, indices, K_STATE,
                     vit->Obroute, obserRouteLEN,
                     vit->Pi, 0, -1);
    vit->memory_bytes += sizeof(indices) + sizeof(mp_path);
    change_mp_path(vit);
}

int main()
{
    VIT *vit = create_vit();
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1); 
    calc(vit);
    clock_gettime(CLOCK_REALTIME, &time2);
    printf("time: %lf \n", (time2.tv_sec - time1.tv_sec) + (time2.tv_nsec - time1.tv_nsec)*1e-9);
    printAns(vit);
    InitVitAns(vit);
    delete_vit(vit);
    return 0;
}