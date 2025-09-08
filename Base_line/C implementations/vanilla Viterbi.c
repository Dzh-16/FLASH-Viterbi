#define _POSIX_C_SOURCE 199309L
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<time.h>

//parameter set 
#define K_STATE 512
#define T_STATE 50
#define obserRouteLEN 128
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

ElementType emax(ElementType a, ElementType b)
{
    return a < b ? b : a;
}

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

ElementType T1[K_STATE][obserRouteLEN];
int T2[K_STATE][obserRouteLEN];

void viterbi(VIT *vit)
{
    for(int i = 0; i < K_STATE; ++i)
    {
        T1[i][0] = initT1(i);
        T2[i][0] = 0;
    }

    for(int j = 1; j < obserRouteLEN; ++j)
        for(int i = 0; i < K_STATE; ++i)
        {
            ElementType tmp = ElementTypeNegMin;
            int arg = -1;
            for(int k = 0; k < K_STATE; ++k)
            {
                ElementType tmp2 = T1[k][j-1] + log(vit->A[k][i]) + log(vit->B[i][vit->Obroute[j]]);
                if(tmp2 > tmp)
                {
                    tmp = tmp2;
                    arg = k;
                }
            }
            if(arg < 0) perror("calc error in void viterbi()");
            T1[i][j] = tmp;
            T2[i][j] = arg;
        }
    
    ElementType tmp = ElementTypeNegMin;
    int arg = -1;
    for(int i = 0; i < K_STATE; ++i)
    {
        if(T1[i][obserRouteLEN-1] > tmp)
        {
            tmp = T1[i][obserRouteLEN-1];
            arg = i;
        }
    }

    if(arg < 0) perror("calc error in void viterbi()");

    vit->Ans[obserRouteLEN-1] = arg;

    for(int i = obserRouteLEN-1; i; --i)
    {
        vit->Ans[i-1] = T2[vit->Ans[i]][i];
    }

    vit->memory_bytes = sizeof(T1)+sizeof(T2);
}

int main()
{
    VIT *vit = create_vit();
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1); 
    viterbi(vit);
    clock_gettime(CLOCK_REALTIME, &time2);
    printf("time: %lf \n", (time2.tv_sec - time1.tv_sec) + (time2.tv_nsec - time1.tv_nsec)*1e-9);
    printAns(vit);
    InitVitAns(vit);
    delete_vit(vit);
    return 0;
}