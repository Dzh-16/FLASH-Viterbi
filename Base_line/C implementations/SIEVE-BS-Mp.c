#define _POSIX_C_SOURCE 199309L
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<time.h>
#include<glib.h>
#include<stdlib.h>

//parameter set 
#define K_STATE 256
#define T_STATE 50
#define obserRouteLEN 128
const float prob = 0.253;
const char data_path[] = "./data/";
const int BeamSearchWidth = 256;

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

char* getAddress(char *stype)
{
    static char path[100];
    snprintf(path, sizeof(path), "%s%s_K%d_T%d_prob%.3f.txt", 
             data_path ,stype, K_STATE, obserRouteLEN, prob);
    return path;
}

int gsize_hash_table(GHashTable *table, int valuesize)
{
    if(table == NULL)
        return 0;
    return sizeof(table) + (sizeof(gpointer)*2 + valuesize)*g_hash_table_size(table);
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

GHashTable* create_hashtable() {
    return g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, g_free);
}

void insert_ele(GHashTable* hash, int key, ElementType value) {
    ElementType* value_ptr = g_new(ElementType, 1);
    *value_ptr = value;
    g_hash_table_insert(hash, GINT_TO_POINTER(key), value_ptr);
}

void insert_int(GHashTable* hash, int key, int value) {
    int* value_ptr = g_new(int, 1);
    *value_ptr = value;
    g_hash_table_insert(hash, GINT_TO_POINTER(key), value_ptr);
}

gboolean set_add(GHashTable *hash, int key) {
    if (g_hash_table_contains(hash, GINT_TO_POINTER(key))) {
        return FALSE;
    }
    insert_int(hash, key, key);
    return TRUE;
}

gboolean set_contain(GHashTable *hash, int key) {
    return g_hash_table_contains(hash, GINT_TO_POINTER(key));
}

gboolean gremove(GHashTable *hash, int key) {
    return g_hash_table_remove(hash, GINT_TO_POINTER(key));
}

ElementType find_ele(GHashTable* hash, int key, int type) {
    ElementType *tmp = (ElementType*)g_hash_table_lookup(hash, GINT_TO_POINTER(key));
    if(tmp == NULL)
    {
        if(type == 1)
        {
            return -ElementTypeNegMin;
        }
        if(type == -1)
        {
            return ElementTypeNegMin;
        }
        perror("ELE ERROR");
    }
    return *tmp;
}

int find_int(GHashTable* hash, int key,int type) {
    int *tmp = (int*)g_hash_table_lookup(hash, GINT_TO_POINTER(key));
    if(tmp == NULL)
    {
        if(type == 1)
        {
            return INT_MAX;
        }
        if(type == -1)
        {
            return -1;
        }
        perror("INT ERROR");
    }
    return *tmp;
}

int single_node_ancestors(int source, int b, GHashTable *b_hop_ancestors_nodes, VIT *vit)
{
    GHashTable *visited = create_hashtable();
    GQueue *queue = g_queue_new();
    g_queue_push_tail(queue, GINT_TO_POINTER(source));
    insert_int(visited, source, 1);

    int max_queue_size = 1, queue_size = 1;
    while(! g_queue_is_empty(queue))
    {
        int s = GPOINTER_TO_INT(g_queue_pop_head(queue)); queue_size--;
        int arg = find_int(visited, s, 0);
        if(arg < b)
            for(int i = 0; i < K_STATE; ++i)
                if(vit->A[i][s] > 0)
                {
                    if(! g_hash_table_contains(visited, GINT_TO_POINTER(i)))
                    {
                        set_add(b_hop_ancestors_nodes, i);
                        insert_int(visited, i, arg+1);
                        g_queue_push_tail(queue, GINT_TO_POINTER(i)); queue_size++;
                    }
                }
        
        if(queue_size > max_queue_size)
                max_queue_size = queue_size;
    }
    int memory_total = sizeof(GQueue) + sizeof(GList)*max_queue_size +
                       gsize_hash_table(visited, sizeof(int)) +
                       gsize_hash_table(b_hop_ancestors_nodes, sizeof(int));
    g_queue_free(queue);
    g_hash_table_destroy(visited);

    return memory_total;
}

int single_node_descendant(int source, int b, GHashTable *b_hop_descendants_nodes, VIT *vit)
{
    GHashTable *visited = create_hashtable();
    GQueue *queue = g_queue_new();
    g_queue_push_tail(queue, GINT_TO_POINTER(source));
    insert_int(visited, source, 1);

    int max_queue_size = 1, queue_size = 1;
    while(! g_queue_is_empty(queue))
    {
        int s = GPOINTER_TO_INT(g_queue_pop_head(queue)); queue_size--;
        int arg = find_int(visited, s, 0);
        if(arg < b)
            for(int i = 0; i < K_STATE; ++i)
                if(vit->A[s][i] > 0)
                {
                    if(! g_hash_table_contains(visited, GINT_TO_POINTER(i)))
                    {
                        set_add(b_hop_descendants_nodes, i);
                        insert_int(visited, i, arg+1);
                        g_queue_push_tail(queue, GINT_TO_POINTER(i)); queue_size++;
                    }
                }
        
        if(queue_size > max_queue_size)
                max_queue_size = queue_size;
    }
    int memory_total = sizeof(GQueue) + sizeof(GList)*max_queue_size +
                       gsize_hash_table(visited, sizeof(int)) +
                       gsize_hash_table(b_hop_descendants_nodes, sizeof(int));
    g_queue_free(queue);
    g_hash_table_destroy(visited);

    return memory_total;
}

int b_hop_ancestors[K_STATE], b_hop_descendants[K_STATE];

void gswap(GHashTable **a, GHashTable **b)
{
    GHashTable* tmp = *a;
    *a = *b;
    *b = tmp;
}

typedef struct
{
    int x,y;
}MEDIANS;
MEDIANS mp_path[obserRouteLEN];
int mp_path_len = 0;

void change_mp_path(VIT *vit)
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

typedef struct {
    gpointer key;
    gpointer value;
} KeyValuePair;

static gint compare_pairs(gconstpointer a, gconstpointer b) {
    const KeyValuePair *pair1 = (const KeyValuePair *)a;
    const KeyValuePair *pair2 = (const KeyValuePair *)b;
    
    ElementType val1 = *(ElementType*)(pair1->value);
    ElementType val2 = *(ElementType*)(pair2->value);
    
    if (val1 - val2 > 1e-8) return -1;
    if (val2 - val1 > 1e-8) return 1;
    return 0;
}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

int sieve_bs_middlepath(VIT *vit, int *indices, int K,
              int *Obroute, int T, int last,
              int *activeTokensStates, int alen)
{
    GHashTable *T1 = create_hashtable();
    for(int i = 0; i < K; ++i)
    {
        insert_ele(T1, indices[i], log(vit->Pi[indices[i]]) + log(vit->B[indices[i]][Obroute[0]]));
    }

    int cur_len = (alen > 0 ? alen : K);
    int current_indices[MAX(cur_len,K)];
    if(alen > 0)
    {
        for(int i = 0; i < alen; ++i)
            current_indices[i] = activeTokensStates[i];
    }
    else
    {
        for(int i = 0; i < K; ++i)
            current_indices[i] = indices[i];
    }

    GHashTable *previous_medians_a = create_hashtable();
    GHashTable *previous_medians_b = create_hashtable();
    GHashTable *new_t1 = NULL;
    GHashTable *new_medians_a = NULL;
    GHashTable *new_medians_b = NULL;

    int th = floor(T/2);
    int *next_subproblems_indices, nlen;

    int max_memory_t = 0;

    for(int j = 1; j < T; ++j)
    {
        new_t1 = create_hashtable();
        new_medians_a = create_hashtable();
        new_medians_b = create_hashtable();

        int memory_t = 0;

        for(int i = 0; i < cur_len; ++i)
            for(int h = 0; h < K; ++h)
            if(vit->A[current_indices[i]][indices[h]] > 0)
            {
                ElementType prob = log(vit->A[current_indices[i]][indices[h]]);
                ElementType h_mapped_t1 = find_ele(T1, current_indices[i],-1) + prob +
                                              (vit->B[indices[h]][Obroute[j]] > 0 ? log(vit->B[indices[h]][Obroute[j]]) : 0);
                if(h_mapped_t1 > find_ele(new_t1, indices[h], -1))
                {
                    insert_ele(new_t1, indices[h], h_mapped_t1);
                    if(j == th)
                    {
                        insert_int(new_medians_a, indices[h], current_indices[i]);
                        insert_int(new_medians_b, indices[h], indices[h]);
                    }
                    else if(j > th)
                    {
                        insert_int(new_medians_a, indices[h], find_int(previous_medians_a, current_indices[i], -1));
                        insert_int(new_medians_b, indices[h], find_int(previous_medians_b, current_indices[i], -1));
                    }
                }
            }
        gswap(&new_medians_a, &previous_medians_a);
        gswap(&new_medians_b, &previous_medians_b);
        gswap(&new_t1, &T1);
        memory_t += gsize_hash_table(new_medians_a, sizeof(int));
        memory_t += gsize_hash_table(new_medians_b, sizeof(int));
        memory_t += gsize_hash_table(previous_medians_a, sizeof(int));
        memory_t += gsize_hash_table(previous_medians_b, sizeof(int));
        memory_t += gsize_hash_table(new_t1, sizeof(ElementType));
        memory_t += gsize_hash_table(T1, sizeof(ElementType));
        g_hash_table_destroy(new_t1);
        g_hash_table_destroy(new_medians_a);
        g_hash_table_destroy(new_medians_b);

        GArray *pairs = g_array_new(FALSE, FALSE, sizeof(KeyValuePair));
        GHashTableIter iter;
        gpointer key, value;

        g_hash_table_iter_init(&iter, T1);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            KeyValuePair pair = {key, value};
            g_array_append_val(pairs, pair);
        }
        g_array_sort(pairs, compare_pairs);

        guint count = MIN(BeamSearchWidth, pairs->len);
        cur_len = 0;
        for (guint i = 0; i < count; i++) {
            KeyValuePair *pair = &g_array_index(pairs, KeyValuePair, i);
            current_indices[cur_len++] = GPOINTER_TO_INT(pair->key);
        }
        g_array_free(pairs, TRUE);

        qsort(current_indices, cur_len, sizeof(int), cmpfunc);

        if(j == th)
        {
            next_subproblems_indices = (int *)g_malloc(cur_len * sizeof(int));
            for(int i = 0; i < cur_len; ++i)
                next_subproblems_indices[i] = current_indices[i];
            nlen = cur_len;
        }

        if(memory_t > max_memory_t)
            max_memory_t = memory_t;
    }

    if(last < 0)
    {
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, T1);
        ElementType score = ElementTypeNegMin;
        int arg = -1;
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            ElementType tmp = *(ElementType*)value;
            if(tmp > score)
            {
                score = tmp;
                arg = GPOINTER_TO_INT(key);
            }
        }
        last = arg;
    }

    g_hash_table_destroy(T1);
    int x_a = find_int(previous_medians_a, last, 0);
    int x_b = find_int(previous_medians_b, last, 0);
    int N_left = floor(T/2), N_right = T - N_left;
    g_hash_table_destroy(previous_medians_a);
    g_hash_table_destroy(previous_medians_b);

    int memory_left = 0;
    if(N_left > 1)
    {
        int y_left[N_left];
        for(int i = 0; i < N_left; ++i)
            y_left[i] = Obroute[i];
        GHashTable *output_set = create_hashtable();
        int memory_node = single_node_ancestors(x_a, N_left, output_set, vit);
        set_add(output_set, x_a);
        if(set_contain(output_set, -1))
            gremove(output_set, -1);
        
        int K_left = g_hash_table_size(output_set), tmptop = 0;
        int states_left_indices[K_left];
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, output_set);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            states_left_indices[tmptop++] = GPOINTER_TO_INT(key);
        }
        g_hash_table_destroy(output_set);

        qsort(states_left_indices, tmptop, sizeof(int), cmpfunc);

        memory_left = 
        sieve_bs_middlepath(vit, states_left_indices, K_left,
                            y_left, N_left, x_a,
                            activeTokensStates, alen);
        memory_left += sizeof(y_left) + memory_node + sizeof(states_left_indices);
    }

    mp_path[mp_path_len].x = x_a, mp_path[mp_path_len].y = x_b; mp_path_len++;

    int memory_right = 0;
    if(N_right > 1)
    {
        int y_right[N_right];
        for(int i = T-N_right; i < T; ++i)
        {
            y_right[i - (T-N_right)] = Obroute[i];
        }
        GHashTable *output_set = create_hashtable();
        int memory_node = single_node_descendant(x_b, N_right, output_set, vit);
        set_add(output_set, x_b);
        if(set_contain(output_set, -1))
            gremove(output_set, -1);

        int K_right = g_hash_table_size(output_set), tmptop = 0;
        int states_right_indices[K_right];
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, output_set);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            states_right_indices[tmptop++] = GPOINTER_TO_INT(key);
        }
        g_hash_table_destroy(output_set);

        qsort(states_right_indices, tmptop, sizeof(int), cmpfunc);

        memory_right =
        sieve_bs_middlepath(vit, states_right_indices, K_right,
                 y_right, N_right, last,
                 next_subproblems_indices, nlen);
        memory_right += sizeof(y_right) + memory_node + sizeof(states_right_indices);
    }
    g_free(next_subproblems_indices);

    return sizeof(current_indices) + MAX(max_memory_t, MAX(memory_left, memory_right)) + sizeof(int)*nlen;
}

void calc(VIT *vit)
{
    int memory_total = 0;
    for(int i = 0; i < K_STATE; ++i)
    {
        GHashTable *output_set = create_hashtable();
        memory_total = MAX(memory_total, single_node_ancestors(i, obserRouteLEN, output_set, vit));
        b_hop_ancestors[i] = g_hash_table_size(output_set);
        g_hash_table_destroy(output_set);
    }
    for(int i = 0; i < K_STATE; ++i)
    {
        GHashTable *output_set = create_hashtable();
        memory_total = MAX(memory_total, single_node_descendant(i, obserRouteLEN, output_set, vit));
        b_hop_descendants[i] = g_hash_table_size(output_set);
        g_hash_table_destroy(output_set);
    }

    int indices[K_STATE];
    for(int i = 0; i < K_STATE; ++i)
    {
        indices[i] = i;
    }

    memory_total += sizeof(indices) + sizeof(mp_path) + sizeof(b_hop_ancestors) + sizeof(b_hop_descendants);

    memory_total +=
    sieve_bs_middlepath(vit, indices, K_STATE,
             vit->Obroute, obserRouteLEN, -1,
             NULL, 0);

    change_mp_path(vit);
    vit->memory_bytes = memory_total;
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