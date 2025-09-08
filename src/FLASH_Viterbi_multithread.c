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
#define MAX_THREADS 30
const char data_path[] = "./data/";

typedef float ElementType;
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

void nvviterNdivide(int L, int R, int *N, int *midpoint, ElementType T1[2][K_STATE], int T2[2][(*N) - 1][K_STATE])
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

    if(L == 0)
    {
        for(register int i = 0; i < K_STATE; ++i)
        {
            T1[0][i] = log(vit->Pi[i]) + log(vit->B[i][T]);
        }
    }
    else
    {
        register int state = vit->Ans[L-1];
        for(register int i = 0; i < K_STATE; ++i)
        {
            T1[0][i] = log(vit->A[state][i]) + log(vit->B[i][T]);
        }
        for(register int j = 0; j+1 < (*N); ++j)
            for(register int i = 0; i < K_STATE; ++i)
            T2[0][j][i] = state;
    }

    register ElementType score, tmp, ktmp;
    register int arg, cur = 0;
    for(register int j = L+1, p = -1; j <= R; ++j)
    {
        T = vit->Obroute[j];
        
        while(p+2 < (*N) && j > midpoint[p+1]+1) ++p;

        for(register int i = 0; i < K_STATE; ++i)
        {
            score = ElementTypeNegMin; arg = -1; tmp = log(vit->B[i][T]);
            for(register int k = 0; k < K_STATE; ++k)
            {
                ktmp = tmp + T1[cur][k] + log(vit->A[k][i]);
                if(ktmp > score)
                    arg = k, score = ktmp;
            }
            T1[cur^1][i] = score;

            for(register int k = 0; k <= p; ++k)
                T2[cur^1][k][i] = T2[cur][k][arg];
            for(register int k = p+1; k+1 < (*N); ++k)
                T2[cur^1][k][i] = arg;
        }

        cur ^= 1;
    }

    arg = vit->Ans[R];
    if(L == 0 && R == obserRouteLEN-1)
    {
        score = T1[cur][0]; arg = 0;
        for(register int i = 1; i < K_STATE; ++i)
        {
            if(T1[cur][i] > score)
                arg = i, score = T1[cur][i];
        }

        vit->Ans[R] = arg;
    }

    for(register int i = 0; i+1 < (*N); ++i)
    {
        vit->Ans[midpoint[i]] = T2[cur][i][arg];
    }
}

void nvviter(int L, int R, int midpoint, ElementType T1[2][K_STATE], int T2[2][K_STATE])
{
    int T = vit->Obroute[L];

    if(L == 0)
    {
        for(register int i = 0; i < K_STATE; ++i)
        {
            T1[0][i] = log(vit->Pi[i]) + log(vit->B[i][T]);
        }
    }
    else
    {
        register int state = vit->Ans[L-1];
        for(register int i = 0; i < K_STATE; ++i)
        {
            T1[0][i] = log(vit->A[state][i]) + log(vit->B[i][T]);
            T2[0][i] = state;
        }
    }

    register ElementType score, tmp, ktmp;
    register int arg, cur = 0;
    for(register int j = L+1; j <= R; ++j)
    {
        T = vit->Obroute[j];

        for(register int i = 0; i < K_STATE; ++i)
        {
            score = ElementTypeNegMin; arg = -1; tmp = log(vit->B[i][T]);
            for(register int k = 0; k < K_STATE; ++k)
            {
                ktmp = tmp + T1[cur][k] + log(vit->A[k][i]);
                if(ktmp > score)
                    arg = k, score = ktmp;
            }

            T1[cur^1][i] = score;
            T2[cur^1][i] = (j > midpoint+1 ? T2[cur][arg] : arg);
        }

        cur ^= 1;
    }

    arg = vit->Ans[R];
    if(L == 0 && R == obserRouteLEN-1)
    {
        score = T1[cur][0]; arg = 0;
        for(register int i = 1; i < K_STATE; ++i)
        {
            if(T1[cur][i] > score)
                arg = i, score = T1[cur][i];
        }

        vit->Ans[R] = arg;
    }

    vit->Ans[midpoint] = T2[cur][arg];
}

void *worker(void *arg)
{
    ThreadPool* pool = (ThreadPool*)arg;
    ElementType T1[2][K_STATE];
    int T2[2][K_STATE];
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
        nvviter(L, R, mid, T1, T2);

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
        ElementType T1[2][K_STATE];
        int T2[2][N - 1][K_STATE];
        nvviterNdivide(0, obserRouteLEN-1, &N, midpoint, T1, T2);

        pool.qH = pool.qT = N-2;
        addQ(&(pool.qH), 0, midpoint[0]);
        for(register int i = 0; i+2 < N; ++i)
            addQ(&(pool.qH), midpoint[i]+1, midpoint[i+1]);
        addQ(&(pool.qH), midpoint[N-2]+1, obserRouteLEN-1);

        vit->memory_bytes = sizeof(midpoint) + sizeof(T1) + sizeof(T2);
    }
    else
        pool.qH = pool.qT = -1,
        addQ(&(pool.qH), 0, obserRouteLEN-1);
    
    ThreadPoolInit();
    ThreadPoolDestory();

    int tmp = MAX_THREADS*(2*K_STATE*sizeof(ElementType)+2*K_STATE*sizeof(int));
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