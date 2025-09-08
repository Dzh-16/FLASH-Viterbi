#define _POSIX_C_SOURCE 199309L
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<time.h>
#include<pthread.h>

//parameter set 
#define K_STATE 2048
#define T_STATE 50
#define obserRouteLEN 512
const float prob = 0.253;
#define MAX_THREADS 16
const int BeamSearchWidth = 500;
const char data_path[] = "./data/";

typedef float ElementType;
typedef int Status;
const ElementType ElementTypeNegMin = -FLT_MAX;
typedef struct {
    int L;
    int R;
} INTERVAL;

typedef struct {
    ElementType Pi[K_STATE];
    ElementType A[K_STATE][K_STATE];
    ElementType B[K_STATE][T_STATE];
    int Obroute[obserRouteLEN];

    int Ans[obserRouteLEN];
    int memory_bytes;
    INTERVAL Q[obserRouteLEN];
} VIT;

typedef struct
{
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    pthread_t threads[MAX_THREADS];
    int qH, qT;
    int shutdown;
} ThreadPool;

VIT *vit;
ThreadPool pool;

/// Max heap related///////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct element
{
    ElementType Value;
    int State;
    int T3_State;
} element;

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
}


int Find_T3_State(element **heap, int state)
{
    int total = (*heap)[0].Value; // Get total number of elements in heap

    for (int i = 1; i <= total; i++)
    {
        if ((*heap)[i].State == state)
        {
            return (*heap)[i].T3_State; // If matching state found, return corresponding T3_State value
        }
    }

    return -1; // If no matching state found, return -1
}

void print_heap_element(element **heap)
{
    for (int i = 1; i < BeamSearchWidth + 1; i++)
    {
        printf("heap element is %lf,state is %d,T3 state is %d\n", (*((*heap) + i)).Value, (*((*heap) + i)).State, (*((*heap) + i)).T3_State);
    }
}

void create_min_heap(element **heap)
{
    int total = (**heap).Value;
    /// Find last non-leaf node
    int child = 2, parent = 1;
    for (int node = total / 2; node > 0; node--)
    {
        parent = node;
        child = 2 * node;
        // int max_node = 2 * node + 1;
        element temp = *((*heap) + parent);     // Save current parent node
        // for (; child <= total; child *= 2, max_node = 2 * parent + 1)
        for (; child <= total; child *= 2)
        {
            if (child + 1 <= total && (*((*heap) + child)).Value > (*((*heap) + child + 1)).Value)       //Find smaller child node
            {
                child++;
            }
            if (temp.Value <= (*((*heap) + child)).Value)       //If parent node <= smallest child node, heap property satisfied, break loop
            { // Changed from < to <=, no swap for child nodes equal to parent
                break;
            }
            *((*heap) + parent) = *((*heap) + child);
            parent = child;
        }
        *((*heap) + parent) = temp;     //Assign value to parent node position, *heap is heap head address, plus parent position is parent address, then * gets value
    }
}

// **
//  * Replace smallest element in heap and maintain min heap property
//  * @param heap Min heap
//  * @param newValue New replacement value
//  * @param newState New state
//  */
void replace_min_heap_element(element **heap, ElementType newValue, int newState, int newT3_State)
{
    // Replace heap top element with new value
    (*heap)[1].Value = newValue;
    (*heap)[1].State = newState;
    (*heap)[1].T3_State = newT3_State;

    int total = (*heap)[0].Value; // Total element count
    int parent = 1;
    int child = 2;

    // Top-down heap adjustment
    while (child <= total)
    {
        // Find smaller of left/right child nodes
        if (child + 1 <= total && (*heap)[child].Value > (*heap)[child + 1].Value)
        {
            child++;
        }

        // If parent <= smallest child, heap property satisfied
        if ((*heap)[parent].Value <= (*heap)[child].Value)
        {
            break;
        }

        // Swap parent and child
        element temp = (*heap)[parent];
        (*heap)[parent] = (*heap)[child];
        (*heap)[child] = temp;

        parent = child;
        child *= 2;
    }
}

Status generate_state_heap(ElementType probability_i, int i, element **heap_total, int T3_State)
{
    element *num = *heap_total;      // Position storing tree count
    element *position = *heap_total; // Current node position in tree
    position = position + i + 1;     // positi moves back i+1 positions each time, already calculated i points
    if (i < BeamSearchWidth - 1)
    { // When tree not full
        insert_binary_tree(position, probability_i, i, T3_State);
        (*num).Value++;
        // Output result after insertion, for testing
        //  printf("total number is %d ,new element is %lf  ,state is %d  ,T3 State is%d\n",(int)((*num).Value),(*position).Value,(*position).State,(*position).T3_State);
        return 0;
    }
    else if (i == BeamSearchWidth - 1)
    { // Tree just filled
        insert_binary_tree(position, probability_i, i, T3_State);
        (*num).Value++;
        // Output result after insertion, for testing
        //  printf("\ntotal number is %d  ,new element is %lf  ,state is %d  \n",(int)((*num).Value),(*position).Value,(*position).State);
        create_min_heap(heap_total);
        // Output sorted result, for testing
        //  print_heap_element(heap_total);
        return 0;
    }
    else
    { // Tree full, and next element > min value, replace and output 1
        if (probability_i > (*heap_total)[1].Value)
        {
            // printf("\nChange:new element value is %lf state is %d T3 State is%d,Total min element is %lf  state is %d T3 State is%d \n",probability_i,i,T3_State,(*heap_total)[1].Value,(*heap_total)[1].State,(*heap_total)[1].T3_State);
            // (*heap_total)[1].Value=probability_i;
            // (*heap_total)[1].State=i;
            // create_max_heap(heap_total);
            replace_min_heap_element(heap_total, probability_i, i, T3_State);
            // Output sorted result, for testing
            //  print_heap_element(heap_total);
            return 1;
        }
        // Tree full, and next element <= min value, no replace, output 2
        else
        {
            // printf("Keep:new element value is %lf  state is %d  ,Total min element is %lf  state is %d  \n",probability_i,i,(*heap_total)[1].Value,(*heap_total)[1].State);
            return 2;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


char* getAddress(char *stype)
{
    static char path[100];
    snprintf(path, sizeof(path), "%s%s_K%d_T%d_prob%.3f.txt", 
             data_path ,stype, K_STATE, obserRouteLEN, prob);
    return path;
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

VIT* create_vit() {
    VIT* vit = (VIT*)malloc(sizeof(VIT));
    if(vit == NULL) perror("malloc failed in VIT* create_vit()");

    InitElement(vit,"A");
    InitElement(vit,"B");
    InitElement(vit,"Pi");
    InitElement(vit,"ob");

    return vit;
}

static inline void delete_vit(VIT *vit)
{
    if(vit != NULL)
    {
        free(vit);
    }
}

static inline void printAns(VIT *vit)
{
    printf("path: [");
    for(int i = 0; i < obserRouteLEN; ++i)
        printf("%d ",vit->Ans[i]);
    puts("]");
    printf("memory: %d\n",vit->memory_bytes);
}

void nvviterNdivide(int L, int R, int *N, int *midpoint, element H[2][(*N)-1][BeamSearchWidth+1])
{
    int T = vit->Obroute[L];
    int gap_length = (R - L) / (*N), gap_extra = (R - L) % (*N);
    midpoint[0] = L + gap_length;
    if(gap_extra) --gap_extra, ++midpoint[0];
    for(register int i = 1; i+1 < (*N); ++i)
    {
        midpoint[i] = midpoint[i-1] + gap_length;
        if(gap_extra) --gap_extra, ++midpoint[i];
    }

    for(register int i = 0; i+1 < (*N); ++i)
        initial_heap_element(H[0][i]);

    if(L == 0)
    {
        element *Hpre;
        ElementType tmp;
        for(register int i = 0; i < K_STATE; ++i)
        {
            tmp = log(vit->Pi[i]) + log(vit->B[i][T]);
            for(register int j = 0; j+1 < (*N); ++j)
                Hpre = H[0][j],
                generate_state_heap(tmp, i, &Hpre, -1);
        }
    }
    else
    {
        element *Hpre;
        ElementType tmp;
        register int state = vit->Ans[L-1];
        for(register int i = 0; i < K_STATE; ++i)
        {
            tmp = log(vit->A[state][i]) + log(vit->B[i][T]);
            for(register int j = 0; j+1 < (*N); ++j)
                Hpre = H[0][j],
                generate_state_heap(tmp, i, &Hpre, -1);
        }
    }

    register ElementType score, tmp, ktmp;
    register int arg, cur = 0;
    for(register int j = L+1, p = -1; j <= R; ++j)
    {
        T = vit->Obroute[j];
        
        while(p+2 < (*N) && j > midpoint[p+1]+1) ++p;

        for(register int i = 0; i+1 < (*N); ++i)
            initial_heap_element(H[cur^1][i]);

        for(register int i = 0; i < K_STATE; ++i)
        {
            score = ElementTypeNegMin; arg = -1; tmp = log(vit->B[i][T]);
            for(register int k = 0; k < BeamSearchWidth; ++k)
            {
                int preState = H[cur][1][k+1].State;
                ktmp = tmp + H[cur][1][k+1].Value + log(vit->A[preState][i]);
                if(ktmp > score)
                    arg = k, score = ktmp;
            }
            
            element *Hpre;
            for(register int k = 0; k <=p; ++k)
            {
                Hpre = H[cur^1][k];
                generate_state_heap(score, i, &Hpre, H[cur][k][arg+1].T3_State);
            }
            for(register int k = p+1; k+1 < (*N); ++k)
            {
                Hpre = H[cur^1][k];
                generate_state_heap(score, i, &Hpre, H[cur][k][arg+1].State);
            }
        }

        cur ^= 1;
    }

    if(L == 0 && R == obserRouteLEN-1)
    {
        score = H[cur][1][1].Value; arg = 0;
        for(register int i = BeamSearchWidth / 2+1; i < BeamSearchWidth; ++i)
        {
            if(H[cur][1][i+1].Value > score)
                arg = i, score = H[cur][1][i+1].Value;
        }

        vit->Ans[R] = H[cur][1][arg+1].State;
        for(register int i = 0; i+1 < (*N); ++i)
        {
            vit->Ans[midpoint[i]] = H[cur][i][arg+1].T3_State;
        }
    }
    else
    {
        element *Hcur; arg = vit->Ans[R];
        for(register int i = 0; i+1 < (*N); ++i)
        {
            Hcur = H[cur][i];
            vit->Ans[midpoint[i]] = Find_T3_State(&Hcur,arg);
        }
    }

}

void nvviter(int L, int R, int midpoint, element H[2][BeamSearchWidth+1])
{
    int T = vit->Obroute[L];

    initial_heap_element(H[0]);

    if(L == 0)
    {
        element *Hpre = H[0];
        ElementType tmp;
        for(register int i = 0; i < K_STATE; ++i)
        {
            tmp = log(vit->Pi[i]) + log(vit->B[i][T]);
            generate_state_heap(tmp, i, &Hpre, -1);
        }
    }
    else
    {
        element *Hpre = H[0];
        ElementType tmp;
        register int state = vit->Ans[L-1];
        for(register int i = 0; i < K_STATE; ++i)
        {
            tmp = log(vit->A[state][i]) + log(vit->B[i][T]);
            generate_state_heap(tmp, i, &Hpre, -1);
        }
    }

    register ElementType score, tmp, ktmp;
    register int arg, cur = 0;
    for(register int j = L+1; j <= R; ++j)
    {
        T = vit->Obroute[j];

        initial_heap_element(H[cur^1]);

        for(register int i = 0; i < K_STATE; ++i)
        {
            score = ElementTypeNegMin; arg = -1; tmp = log(vit->B[i][T]);
            for(register int k = 0; k < BeamSearchWidth; ++k)
            {
                int preState = H[cur][k+1].State;
                ktmp = tmp + H[cur][k+1].Value + log(vit->A[preState][i]);
                if(ktmp > score)
                    arg = k, score = ktmp;
            }
            element *Hcur = H[cur^1];
            generate_state_heap(score, i, &Hcur,  (j > midpoint+1 ? H[cur][arg+1].T3_State : H[cur][arg+1].State));
        }

        cur ^= 1;
    }

    if(L == 0 && R == obserRouteLEN-1)
    {
        score = H[cur][1].Value; arg = 0;
        for(register int i = BeamSearchWidth / 2+1; i < BeamSearchWidth; ++i)
        {
            if(H[cur][i+1].Value > score)
                arg = i, score = H[cur][i+1].Value;
        }

        vit->Ans[R] = H[cur][arg+1].State;
        vit->Ans[midpoint] = H[cur][arg+1].T3_State;
    }
    else
    {
        arg = vit->Ans[R];
        element *Hcur = H[cur];
        vit->Ans[midpoint] = Find_T3_State(&Hcur,arg);
    }

}

void *worker(void *arg)
{
    ThreadPool* pool = (ThreadPool*)arg;
    element H[2][BeamSearchWidth+1];
    while(1)
    {
        pthread_mutex_lock(&pool->lock);

        while (pool->qH == pool->qT && !pool->shutdown)
        {
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        if (pool->shutdown)
        {
            pthread_mutex_unlock(&pool->lock);
            pthread_exit(NULL);
        }

        int L = vit->Q[++(pool->qT)].L, R = vit->Q[pool->qT].R;
        if((pool->qT) == obserRouteLEN-2)
            pool->shutdown = 1;

        pthread_mutex_unlock(&pool->lock);

        int mid = (L + R) >> 1;
        nvviter(L, R, mid, H);

        if (R <= L + 1) {
            pthread_cond_broadcast(&(pool->cond));
            continue;
        }

        pthread_mutex_lock(&pool->lock);

        vit->Q[++(pool->qH)].L = L, vit->Q[pool->qH].R = mid;
        if(R > mid + 1)
            vit->Q[++(pool->qH)].L = mid+1, vit->Q[pool->qH].R = R;
        
        pthread_mutex_unlock(&pool->lock);
        
        pthread_cond_broadcast(&(pool->cond));
    }
}

static inline void addQ(int *qH, int L, int R)
{
    vit->Q[++(*qH)].L = L;
    vit->Q[*qH].R = R;
}

static inline void ThreadPoolInit()
{
    pool.shutdown = 0;
    pthread_mutex_init(&pool.lock, NULL);
    pthread_cond_init(&pool.cond, NULL);

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_create(&pool.threads[i], NULL, worker, &pool);
    }
}

static inline void ThreadPoolDestory()
{
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(pool.threads[i], NULL);
    }

    pthread_mutex_destroy(&pool.lock);
    pthread_cond_destroy(&pool.cond);
}


void calc()
{
    int N = MAX_THREADS;
    vit->memory_bytes = 0;
    if (N > 2 && obserRouteLEN >= N << 1)
    {
        int midpoint[N-1];
        element H[2][N-1][BeamSearchWidth+1];
        nvviterNdivide(0, obserRouteLEN-1, &N, midpoint, H);

        pool.qH = pool.qT = N-2;
        addQ(&(pool.qH), 0, midpoint[0]);
        for(register int i = 0; i+2 < N; ++i)
            addQ(&(pool.qH), midpoint[i]+1, midpoint[i+1]);
        addQ(&(pool.qH), midpoint[N-2]+1, obserRouteLEN-1);

        vit->memory_bytes = sizeof(midpoint) + sizeof(H);
    }
    else
        pool.qH = pool.qT = -1,
        addQ(&(pool.qH), 0, obserRouteLEN-1);
    
    ThreadPoolInit();
    ThreadPoolDestory();

    int tmp = MAX_THREADS*(2*(BeamSearchWidth+1)*sizeof(element));
    if(tmp > vit->memory_bytes)
        vit->memory_bytes = tmp;
    vit->memory_bytes += sizeof(ThreadPool)+sizeof(obserRouteLEN*sizeof(INTERVAL));
}

int main()
{
    vit = create_vit();
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1);
    calc();
    clock_gettime(CLOCK_REALTIME, &time2); 
    printf("time: %lf \n", (time2.tv_sec - time1.tv_sec) + (time2.tv_nsec - time1.tv_nsec)*1e-9);
    printAns(vit);
    delete_vit(vit);
    return 0;
}