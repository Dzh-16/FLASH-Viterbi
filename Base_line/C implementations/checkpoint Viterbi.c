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

int viterbi_checkpoint_subroutine(VIT *vit, int *Obroute, ElementType *initial_probabilities, int this_step, int *count)
{
    int T_sub = this_step+((*count) != obserRouteLEN-1);
    ElementType T1_sub[K_STATE][T_sub];
    int T2_sub[K_STATE][T_sub];

    for(int i = 0; i < K_STATE; ++i)
    {
        T1_sub[i][0] = initial_probabilities[i];
        T2_sub[i][0] = 0;
    }

    for(int j = 1; j < T_sub; ++j)
        for(int i = 0; i < K_STATE; ++i)
        {
            ElementType tmp = ElementTypeNegMin;
            int arg = -1;
            for(int k = 0; k < K_STATE; ++k)
            {
                ElementType tmp2 = T1_sub[k][j-1] + log(vit->A[k][i]) + log(vit->B[i][Obroute[j]]);
                if(tmp2 > tmp)
                {
                    tmp = tmp2;
                    arg = k;
                }
            }
            T1_sub[i][j] = tmp;
            T2_sub[i][j] = arg;
        }
    
    if((*count) == obserRouteLEN-1)
    {
        ElementType tmp = ElementTypeNegMin;
        int arg = -1;
        for(int k = 0; k < K_STATE; ++k)
        {
            if(T1_sub[k][T_sub-1] > tmp)
            {
                tmp = T1_sub[k][T_sub-1];
                arg = k;
            }
        }
        vit->Ans[(*count)--] = arg;
    }

    for(int i = T_sub-1; i > 0; --i)
    {
        vit->Ans[*count] =  T2_sub[vit->Ans[(*count)+1]][i];
        (*count)--;
    }

    return sizeof(T1_sub)+sizeof(T2_sub);
}

void viterbi_checkpoint(VIT *vit, int step)
{
    int T = obserRouteLEN;
    if(step <= 0)
        step = floor(sqrt(1.0*T));
    
    ElementType T1_previous[K_STATE];
    for(int i = 0; i < K_STATE; ++i)
    {
        T1_previous[i] = initT1(i);
    }

    int checkpoints[T/step+1], checkpointslen = 0;
    for(int i = 0; i < T; i += step)
        checkpoints[checkpointslen++] = i;
    ElementType T1[K_STATE][checkpointslen];
    memset(T1,0,sizeof(T1));

    for(int i = 0; i < K_STATE; ++i)
    {
        T1[i][0] = T1_previous[i];
    }

    int cnt_checks = 0;
    ElementType T1_current[K_STATE];
    for(int j = 1; j < T; ++j)
    {
        for(int i = 0; i < K_STATE; ++i)
        {
            ElementType tmp = ElementTypeNegMin;
            for(int k = 0; k < K_STATE; ++k)
            {
                tmp = emax(tmp, T1_previous[k] + log(vit->A[k][i]) + log(vit->B[i][vit->Obroute[j]]));
            }
            T1_current[i] = tmp;
        }

        for(int i = 0; i < K_STATE; ++i)
            T1_previous[i] = T1_current[i];

        for(int i = 0; i < checkpointslen && j >= checkpoints[i]; ++i)
            if(j==checkpoints[i])
            {
                cnt_checks++;
                for(int k = 0; k < K_STATE; ++k)
                {
                    T1[k][cnt_checks] = T1_current[k];
                }
                break;
            }
    }

    int count = obserRouteLEN-1;
    int viterbi_checkpoint_subroutine_memory = 0;
    for(int i = checkpointslen-1; i >= 0; --i)
    {
        ElementType initial_probabilities[K_STATE];
        for(int j = 0; j < K_STATE; ++j)
        {
            initial_probabilities[j] = T1[j][i];
        }
        int this_step = step;
        if(i == checkpointslen-1)
        {
            this_step = T - checkpoints[checkpointslen-1];
        }

        int Obroute[this_step+(i != checkpointslen-1)];
        for(int k = 0; k < this_step + (i != checkpointslen-1); ++k)
            Obroute[k] = vit->Obroute[checkpoints[i]+k];
        int tmp = viterbi_checkpoint_subroutine(vit,Obroute,initial_probabilities,this_step,&count);
        if(tmp > viterbi_checkpoint_subroutine_memory) viterbi_checkpoint_subroutine_memory = tmp;
    }

    vit->memory_bytes = sizeof(T1_previous)+sizeof(T1)+sizeof(T1_current)+sizeof(checkpoints)+viterbi_checkpoint_subroutine_memory;
}

int main()
{
    VIT *vit = create_vit();
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1); 
    viterbi_checkpoint(vit,0);
    clock_gettime(CLOCK_REALTIME, &time2);
    printf("time: %lf \n", (time2.tv_sec - time1.tv_sec) + (time2.tv_nsec - time1.tv_nsec)*1e-9);
    printAns(vit);

    delete_vit(vit);
    return 0;
}