import numpy as np
from collections import defaultdict
import copy
from math import floor, ceil
import heapq
import sys

# --- Memory helpers (container shallow + elements) ---
def _size_list(lst):
    try:
        return sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)
    except Exception:
        return sys.getsizeof(lst)

def _size_set(st):
    try:
        return sys.getsizeof(st) + sum(sys.getsizeof(x) for x in st)
    except Exception:
        return sys.getsizeof(st)

def _size_dict(d):
    try:
        total = sys.getsizeof(d)
        for k, v in d.items():
            total += sys.getsizeof(k)
            if isinstance(v, dict):
                total += _size_dict(v)
            elif isinstance(v, set):
                total += _size_set(v)
            elif isinstance(v, list) or isinstance(v, tuple):
                total += _size_list(v)
            else:
                total += sys.getsizeof(v)
        return total
    except Exception:
        return sys.getsizeof(d)





class SIEVE_BEAMSEARCH:
    
    def __init__(self, pi, A_out, A_in, acustic_costs,  beam_width): 
        self.Pi = pi 
        self.A_out = A_out 
        self.A_in = A_in 
        self.acustic_costs = acustic_costs
        self.initial_state = None 
        self.n_components = 39
        self.visited_median_nodes = set()
        self.B = beam_width
        self.path = []
        self.memory_bytes = 0
        self.memory_bytes2 = 0
    
        self.max_bs_sieve_memory_less = 0


        self.max_BFS_memory = 0
        self.max_mp_bs_sieve_memory = 0
        self.max_mp_bs_sieve_memory_less = 0


    def viterbi_space_efficient(self, indices, frames, Pi = None, K = None, last = None, activeTokensStates=None):
        """
        SIEVE-BS
        """

        T = len(frames)

        # indices set
        overall_indices_set = set(copy.deepcopy(indices)) # this indices are the active states

        if K == None:
            K = len(indices)

        if K == 1:
            print(  [indices[0] for i in range(len(frames))]  )


        if K > 1:
            Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))
            # T1 = Pi                                                                   #修改

            T1 = defaultdict(lambda: float('-inf'))                                     #修改
            for node_i in indices:                                                      #修改
                T1[node_i] = Pi[node_i]+self.acustic_costs[frames[0]][(0,node_i)]       #修改

            previous_n = defaultdict(float)
            previous_medians = defaultdict(lambda: (-1,-1))
            previous_medians_value = defaultdict(lambda: float('inf'))
            previous_active_states = set()

            if activeTokensStates!=None:
                current_indices = activeTokensStates

            else:
                current_indices = copy.deepcopy(indices)


            for j in range(1,T):

                new_medians = defaultdict(lambda: (-1,-1))
                new_t1 = defaultdict(lambda: float('-inf'))
                new_n = defaultdict(float)
                new_median_values = defaultdict(lambda: float('inf'))
                updated_medians = set()
                active_states = defaultdict(set)

                for node_i in current_indices:

                    for node_h_tuple in self.A_out[ node_i ]:
                        h = node_h_tuple[0] # index
                        # if h in overall_indices_set and h!=node_i:                            修改
                        if h in overall_indices_set :                                           #修改

                            prob = float( node_h_tuple[1] )
                            if (node_i , h) in self.acustic_costs[frames[j]].keys():
                                 # transition likelihood
                                h_mapped_t1 = T1[node_i] + prob + self.acustic_costs[frames[j]][(node_i,h)]
                            else:
                                h_mapped_t1 = T1[node_i] + prob


                            if h_mapped_t1  > new_t1[h]:
                                new_t1[h] = h_mapped_t1


                                this_pair_to_compare = max(self.b_hop_ancestors[node_i], self.b_hop_descendants[h])

                                if this_pair_to_compare < previous_medians_value[node_i] :

                                    new_median_values[h] = this_pair_to_compare
                                    new_medians[h] = (node_i , h)
                                    new_n[h] = j
                                    updated_medians.add(h)

                                elif this_pair_to_compare == previous_medians_value[node_i]:

                                    if abs(j-T/2) < abs(previous_n[node_i]  - T/2):

                                        new_median_values[h] = this_pair_to_compare
                                        new_medians[h] = (node_i,h)
                                        new_n[h] = j
                                        updated_medians.add(h)

                                    else:
                                        if previous_medians[node_i]!= (-1,-1):

                                            new_medians[h] = previous_medians[node_i]
                                            new_n[h] = previous_n[node_i]
                                            new_median_values[h]  = previous_medians_value[node_i]
                                            if h in updated_medians:
                                                updated_medians.remove(h)

                                            active_states[h] = previous_active_states[node_i]

                                else:
                                    if previous_medians[node_i]!=  (-1,-1):
                                    #    M[i] = previous_medians[maximizer]
                                    #    N[i] = previous_n[maximizer]
                                        new_medians[h] = previous_medians[node_i]
                                        new_n[h] = previous_n[node_i]
                                        new_median_values[h]  = previous_medians_value[node_i]
                                        if h in updated_medians:
                                            updated_medians.remove(h)
                                        active_states[h] = previous_active_states[node_i]



                effectiveB = min(self.B , len(new_t1))
                current_indices = heapq.nlargest(effectiveB, new_t1, key=new_t1.get)

                for nod in updated_medians:
                    active_states[nod] = current_indices


                previous_n = new_n
                previous_medians = new_medians
                previous_medians_value = new_median_values
                T1 = new_t1
                previous_active_states = active_states


            overall_indices_set_size = _size_set(overall_indices_set)
            T1_size = _size_dict(T1)
            previous_n_size = _size_dict(previous_n)
            previous_medians_size = _size_dict(previous_medians)
            previous_medians_value_size = _size_dict(previous_medians_value)
            previous_active_states_size = _size_set(previous_active_states)
            current_indices_size = _size_list(current_indices)
            new_medians_size = _size_dict(new_medians)
            new_t1_size = _size_dict(new_t1)
            new_n_size = _size_dict(new_n)
            new_median_values_size = _size_dict(new_median_values)
            updated_medians_size = _size_set(updated_medians)
            active_states_size = _size_dict(active_states)

            sieve_bs_memory = T1_size+previous_n_size+previous_medians_size+previous_medians_value_size+previous_active_states_size+current_indices_size+new_medians_size+new_t1_size+new_n_size+new_median_values_size+updated_medians_size+active_states_size+overall_indices_set_size



            if last == None:
                last = heapq.nlargest(1, T1, key=T1.get)[0] #np.argmax(T1)

            else:
                last = last

            x_a, x_b =  new_medians[last]

            N_left =  int(new_n[last])  #floor(len(frames)/2)

            left_size = 0
            if N_left >1:

                left_frames = frames[:N_left]
                b_hop_ancestors_nodes_x_a,single_node_ancestors_size = self.single_node_ancestors( x_a , N_left )
                left_size = _size_list(left_frames) + single_node_ancestors_size
                # 移除集合中的 -1 元素
                b_hop_ancestors_nodes_x_a.discard(-1)
                states_left_indices =  sorted( list(b_hop_ancestors_nodes_x_a.union({x_a})) ) # basically indicdes is the list of node string ids (not numeric)
                left_size += _size_list(states_left_indices)
                K_left = len(states_left_indices) # - 1
                # if  K_left==33:
                #     print(states_left_indices)
                left_size += self.viterbi_space_efficient(states_left_indices, left_frames, Pi = Pi, K = K_left, last = x_a, activeTokensStates = activeTokensStates)


            N_right = len(frames) - N_left

            
            #print("(" + str(x_a) + " " + str(x_b) + ")")
            self.path.append(new_medians[last])

            right_size = 0
            if N_right >1:

                right_frames = frames[-N_right:]
                b_hop_descendants_nodes_x_b,single_node_descendant_size = self.single_node_descendant( x_b , N_right )
                right_size = _size_list(right_frames) + single_node_descendant_size
                # 移除集合中的 -1 元素
                b_hop_descendants_nodes_x_b.discard(-1)
                states_right_indices = sorted( list(b_hop_descendants_nodes_x_b.union({x_b})) )
                right_size += _size_list(states_right_indices)
                K_right = len(states_right_indices)
                # if  K_right==33:
                #     print(states_right_indices)
                right_frames = frames[-N_right:]
                # pi = defaultdict(lambda: float('-inf'))           #修改
                # pi[x_b]=0                                         #修改
                right_size += self.viterbi_space_efficient(states_right_indices, right_frames, Pi = Pi, K = K_right, last = last, activeTokensStates =  active_states[last])  # append to the right

        large_sieve_bs_memory = sieve_bs_memory + max(right_size,left_size)

        if sieve_bs_memory > self.memory_bytes :
            self.memory_bytes = sieve_bs_memory
        if large_sieve_bs_memory > self.memory_bytes2:
            self.memory_bytes2 = large_sieve_bs_memory

        return large_sieve_bs_memory
    
    
    
    
    
    def beam_search(self, indices, frames, Pi = None, K = None): 
        """
        STANDARD BEAM SEARCH ALGORITHM 
        """
        
        
        T = len(frames) 
            
       
        tot_memory = len(indices)
        
        if K == None: 
            K = len(indices)
                    
        if self.initial_state!=None: # known initial state 
            #Pi = np.array([-float("inf") if it!=self.initial_state else 0 for it in indices]) # we start from the 
            Pi = defaultdict(lambda: float('-inf'))
            Pi[self.initial_state] = 0                                
            
                                             # initial states for sure   
        Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))                            
         
        T1 = defaultdict(lambda: defaultdict(lambda: float('-inf')))    
        T2 = defaultdict(lambda: defaultdict(float))    
        
        for t in Pi: 
            T1[0][t] = Pi[t]
            T2[0][t] = 0 
     
        current_indices = copy.deepcopy(indices) 
        
        for j in frames[1:]: 
            
            
           
            this_j_T1 =  defaultdict(lambda: float('-inf')) #[float("-inf")  for _ in range(K)] 
            this_j_T2 =  defaultdict(float) 
        
            for node_i in current_indices: 
             
                for node_h_tuple in self.A_out[ node_i ]: 
                    h = node_h_tuple[0] # index 
                    if h!=node_i:
                    
                        prob = float( node_h_tuple[1] )
                        if (node_i , h) in self.acustic_costs[j].keys(): 
                             # transition likelihood 
                            h_mapped_t1 = T1[j-1][node_i] + prob + self.acustic_costs[j][(node_i,h)]
                        else:
                            h_mapped_t1 = T1[j-1][node_i] + prob
                        
                        
                        if h_mapped_t1  > this_j_T1[h]: 
                            this_j_T1[h] = h_mapped_t1
                            this_j_T2[h] = node_i 
                       
            
            tot_memory+= 2 * len(this_j_T1)
            
            
            for k in this_j_T1:
                T1[j][k] = this_j_T1[k]
                T2[j][k] = this_j_T2[k]               
                     
            
            effectiveB = min(self.B , len(this_j_T1))
            current_indices = heapq.nlargest(effectiveB, this_j_T1, key=this_j_T1.get)
            
        # backtracking 
        x = np.zeros(T, dtype=int)
        
        top_pair = heapq.nlargest(1, T1[T-1], key=T1[T-1].get)
            
        x[-1] = int(top_pair[0])
        top_likelihood =  T1[T-1][top_pair[0]] #float(top_pair[1])  #heapq.nlargest(1, T1[T-1], key=T1[T-1].get)[1]
        
        for i in reversed(range(1, T)):
          
            x[i - 1] = T2[i][x[i]]
                              
        return x , top_likelihood , tot_memory



    def viterbi_middlepath(self, indices, frames, Pi = None, K = None, last = None, activeTokensStates=None):
        """
        SIEVE-BS middlepath
        Space Efficient Beam Search SIEVE-BS-Middlepath algorithm
        """



        T = len(frames)
        th = floor(T/2)

        # indices set
        overall_indices_set = set(copy.deepcopy(indices)) # this indices are the active states

        if K == None:
            K = len(indices)


        # Base case number
        if K == 1:
            print(  [indices[0] for i in range(len(frames))]  )


        if K > 1:
            Pi = Pi if Pi is not None else defaultdict(lambda: float(1/K))
            # T1 = Pi

            T1 = defaultdict(lambda: float('-inf'))                                     #修改
            for node_i in indices:                                                      #修改
                T1[node_i] = Pi[node_i]+self.acustic_costs[frames[0]][(0,node_i)]       #修改

            if T1[node_i] == float('-inf'):
                # 执行相应的操作，例如打印提示信息
                print(f"T1[{node_i}] is -inf")


            previous_middlepath =  defaultdict(lambda: (-1,-1))

            if activeTokensStates!=None:
                current_indices = activeTokensStates
            else:
                current_indices = copy.deepcopy(indices)

            for j in range(1,T):
                new_middlepath = defaultdict(lambda: (-1,-1))
                new_t1 = defaultdict(lambda: float('-inf'))

                for node_i in current_indices:
                    for node_h_tuple in self.A_out[ node_i ]:
                        h = node_h_tuple[0] # index
                        # if h in overall_indices_set and h!=node_i:
                        if h in overall_indices_set:                                    #修改

                            prob = float( node_h_tuple[1] )
                            if (node_i , h) in self.acustic_costs[frames[j]].keys():
                                 # transition likelihood
                                h_mapped_t1 = T1[node_i] + prob + self.acustic_costs[frames[j]][(node_i,h)]
                            else:
                                h_mapped_t1 = T1[node_i] + prob


                            if h_mapped_t1  > new_t1[h]:
                                new_t1[h] = h_mapped_t1

                                if  j == th:
                                    new_middlepath[h] = (node_i , h)


                                elif j > th: #floor(T/2):
                                    new_middlepath[h] = previous_middlepath[node_i]

                effectiveB = min(self.B , len(new_t1))
                current_indices = heapq.nlargest(effectiveB, new_t1, key=new_t1.get)        #排序后的前B个元素

                if j == th:
                    next_subproblems_indices = current_indices

                previous_middlepath = new_middlepath
                T1 = new_t1

            overall_indices_set_size = _size_set(overall_indices_set)
            T1_size = _size_dict(T1)
            new_t1_size = _size_dict(new_t1)
            new_medians_size = _size_dict(new_middlepath)
            previous_medians_size = _size_dict(previous_middlepath)
            current_indices_size = _size_set(current_indices)
            mp_sieve_memory= T1_size+new_t1_size+new_medians_size+previous_medians_size+current_indices_size+overall_indices_set_size


            if last == None:
                last = heapq.nlargest(1, T1, key=T1.get)[0] #np.argmax(T1)

            else:
                last = last


            x_a, x_b =  new_middlepath[last]
            N_left = floor(len(frames)/2)
            N_right = len(frames) - N_left
            single_node_size = 0
            left_size = 0
            if N_left >1:

                left_frames = frames[:N_left]
                b_hop_ancestors_nodes_x_a,single_node_ancestors_size = self.single_node_ancestors( x_a , N_left )
                left_size = _size_list(left_frames) + single_node_ancestors_size
                # 移除集合中的 -1 元素

                b_hop_ancestors_nodes_x_a.discard(-1)
                states_left_indices =  sorted( list(b_hop_ancestors_nodes_x_a.union({x_a})) ) # basically indicdes is the list of node string ids (not numeric)
                left_size += _size_list(states_left_indices)
                K_left = len(states_left_indices) # - 1
                # if  x_a==-1:
                #     print(states_right_indices)
                left_size += self.viterbi_middlepath(states_left_indices, left_frames, Pi = Pi, K = K_left, last = x_a, activeTokensStates = activeTokensStates)


            N_right = len(frames) - N_left

            #print("(" + str(x_a) + " " + str(x_b) + ")")
            self.path.append(new_middlepath[last])

            right_size = 0
            if N_right >1:

                b_hop_descendants_nodes_x_b,single_node_descendant_size = self.single_node_descendant( x_b , N_right )
                # 移除集合中的 -1 元素

                b_hop_descendants_nodes_x_b.discard(-1)
                states_right_indices = sorted( list(b_hop_descendants_nodes_x_b.union({x_b})) )
                
                right_size = single_node_descendant_size + _size_list(states_right_indices)
                
                K_right = len(states_right_indices)
                # if  x_b==-1:
                #     print(states_right_indices)
                right_frames = frames[-N_right:]

                right_size += _size_list(right_frames)
                # pi = defaultdict(lambda: float('-inf'))           #修改
                # pi[x_b]=0

                right_size += self.viterbi_middlepath(states_right_indices, right_frames, Pi = Pi, K = K_right, last = last, activeTokensStates = next_subproblems_indices)  # append to the right

        large_mp_sieve_memory = mp_sieve_memory+max(right_size, left_size)
        if mp_sieve_memory > self.memory_bytes :
            self.memory_bytes = mp_sieve_memory
        if large_mp_sieve_memory > self.memory_bytes2:
            self.memory_bytes2 = large_mp_sieve_memory

        return large_mp_sieve_memory
    

    def single_node_descendant(self, source, b): 
        
        visited = set() 
        visited_emitting = dict() 
        
        b_hop_descendants_nodes = set() 
         
        to_be_mantained = set() 
         # Create a queue for BFS
        queue = []
          
        queue.append(source)
        visited_emitting[source] = 1
   
        max_queue_size = sys.getsizeof(queue)

        while queue: # and level < b:
        
            s = queue.pop(0)
             
            if visited_emitting[s] < b: 
             
                for tup  in self.A_out[s]:    
                      
                    node_id = tup[0]
                    
                    if node_id not in visited: 
                        b_hop_descendants_nodes.add(node_id) 
                        visited_emitting[node_id] = visited_emitting[s] + 1 
                         
                        queue.append(node_id)
                        visited.add(node_id)

                    current_queue_size = sys.getsizeof(queue)
                    if current_queue_size > max_queue_size:
                        max_queue_size = current_queue_size
                
        single_node_descendant_size = max_queue_size+sys.getsizeof(b_hop_descendants_nodes)+sys.getsizeof(visited)+sys.getsizeof(visited_emitting)+sys.getsizeof(to_be_mantained)
 
        return b_hop_descendants_nodes,single_node_descendant_size
        
        
        
    def single_node_ancestors(self,source, b): 

        visited = set() 
        visited_emitting = dict() 
        to_be_mantained = set() 
        b_hop_ancestors_nodes = set() 
         # Create a queue for BFS
        queue = []
        queue.append(source)
        # queue.append("null") # for level 
        visited_emitting[source] = 1

        max_queue_size = sys.getsizeof(queue)


        while queue: # and level < b:

            s = queue.pop(0)
            
             
            if visited_emitting[s] <b : 
             
                for tup  in self.A_in[s]:   
                                          
                    node_id = tup[0]
                    
                    if node_id not in visited: 

                        b_hop_ancestors_nodes.add(node_id) 
                      
                        visited_emitting[node_id] = visited_emitting[s] + 1 
                         
                        queue.append(node_id)
                        visited.add(node_id)    

                    current_queue_size = sys.getsizeof(queue)
                    if current_queue_size > max_queue_size:
                        max_queue_size = current_queue_size

        single_node_ancestors_size = max_queue_size+sys.getsizeof(b_hop_ancestors_nodes)+sys.getsizeof(visited)+sys.getsizeof(visited_emitting)+sys.getsizeof(to_be_mantained)

        return b_hop_ancestors_nodes,single_node_ancestors_size
    

    def viterbi_preprocessing_descendants_pruning_root(self, indices, b, K): 
        
        ''' 
        Implement preprocessing to count descendants  
        necessary to find the median pairs. 
        Perform one BFS to search the b-hop neighbourhood of each state amd updates number of ancestors and descendants 
        
        Parameters: 
        indices: sequence of states 
        b: number of hops to be explored 
        K: number of states 
        
        ''' 
        max_single_node_descendant_size = 0
        
        self.b_hop_descendants = defaultdict(int)
        
        self.b_hop_descendants_nodes = defaultdict(set)
        
        for source in range(K): 
            
            output_set,single_node_descendant_size = self.single_node_descendant(source, b)
            
            if single_node_descendant_size > max_single_node_descendant_size:
                max_single_node_descendant_size = single_node_descendant_size

            self.b_hop_descendants[source] = len( output_set )

        return max_single_node_descendant_size+sys.getsizeof(self.b_hop_descendants)+sys.getsizeof(self.b_hop_descendants_nodes)
        
            
        
    def viterbi_preprocessing_ancestors_pruning_root(self, indices, b, K): 
        
        ''' 
        Implement preprocessing to count ancestors 
        necessary to find the median pairs. 
        Perform one BFS to search the b-hop neighbourhood of each state amd updates number of ancestors and descendants 
        
        Parameters: 
        indices: sequence of states 
        b: number of hops to be explored 
        K: number of states 
        
        ''' 
        max_single_node_ancestors_size = 0

        self.b_hop_ancestors = defaultdict(int)
        
        self.b_hop_ancestors_nodes = defaultdict(set)
    
        for source in range(K): 
            
            output_set,single_node_ancestors_size = self.single_node_ancestors(source, b)

            if single_node_ancestors_size > max_single_node_ancestors_size:
                max_single_node_ancestors_size = single_node_ancestors_size
            
            self.b_hop_ancestors[source] = len( output_set )

        return max_single_node_ancestors_size+sys.getsizeof(self.b_hop_ancestors)+sys.getsizeof(self.b_hop_ancestors_nodes)

    def pretty_print_path(self, path): 
        ''' print sieve output ''' 
        if len(path) == 0: 
            raise ValueError("You must call sieve first")

        output_path = [] 
        
        output_path.append(path[0][0])
        output_path.append(path[0][1])
        i = 1
        while len(output_path) <= len(path): 
            if path[i] == -1 : 
                output_path.append(path[i+1][0])
                output_path.append(path[i+1][1])
                i+=1 
            else: 
                output_path.append(path[i][1])
            i+=1 

        print("Path " + "|" + ",".join(list(map(str, output_path))) + "|")
    
