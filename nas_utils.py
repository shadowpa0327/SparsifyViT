import math
import random
import heapq
import copy

from collections import defaultdict
def convert_to_hashable(entry):
    """
    Convert an unhashable object (e.g., 2D list) to a hashable one (e.g., tuple of tuples).

    Parameters:
        entry: The unhashable object to be converted.

    Returns:
        object: The converted hashable object.
    """
    if isinstance(entry, list):
        return tuple(map(convert_to_hashable, entry))
    return entry


def convert_from_hashable(entry):
    """
    Convert a hashable object (e.g., tuple of tuples) back to its original unhashable form (e.g., 2D list).

    Parameters:
        entry: The hashable object to be converted.

    Returns:
        object: The converted unhashable object.
    """
    if isinstance(entry, tuple):
        return list(map(convert_from_hashable, entry))
    return entry

class CandidatePool:
    def __init__(self, candidate_pool_size=1000):
        """
        Object for candidate pools

        Parameters:
            candidate_pool_size (int, optional): The maximum size of the promising pools (max-heap). Default is 1000.
        """
        self.max_pool_size = candidate_pool_size
        self.candidate_pools = {}


    def get_size(self):
        return len(self.candidate_pools)

    def get_one_subnet(self):
        """
        Selects a subnet
        Returns:
            The selected subnet (from `candidate_pools`) if `candidate_pools` is not empty.
            Otherwise, return None
        """
        if len(self.candidate_pools) == 0:
            return None
        else:
            return convert_from_hashable(random.choice(list(self.candidate_pools.keys())))  # Extract only the subnet (index 1)
    
    def _sort_and_limit(self):
        self.candidate_pools = dict(sorted(self.candidate_pools.items(), key=lambda item: item[1][0]))
        self.candidate_pools = dict(list(self.candidate_pools.items())[:self.max_pool_size])
    
    def add_one_subnet_with_score_and_flops(self, subnet, score, flops):
        """
        Adds a subnet to the `candidate_pools` if it belongs to the top candidates.

        Parameters:
            subnet: The subnet to add to the promising pools.
            score: The score of the subnet (used for max-heap comparison).
        """
        
        self.candidate_pools[convert_to_hashable(subnet)] = (score, flops)
        self._sort_and_limit()
    
    def clear_candidate_pools(self):
        """
        Clears the `candidate_pools`.
        """
        self.candidate_pools = []

    def state_dict(self):
        return {
            "max_pool_size": self.max_pool_size,
            #"candidate_pools" : self.candidate_pools
            "candidate_pools" : copy.deepcopy(self.candidate_pools)
        }

    def load_state_dict(self, state_dict):
        if "max_pool_size" not in state_dict:
            raise ValueError(f"Expect `max_pool_size` in the state_dict")
        if "candidate_pools" not in state_dict:
            raise ValueError(f"Expect `candidate_pools` in the state_dict")
        self.max_pool_size = state_dict['max_pool_size']
        self.candidate_pools = state_dict['candidate_pools']



def build_candidate_pool(args, config):
    return CandidatePool(config['cand_pool_size'])



class LinearEpsilonScheduler:
    def __init__(self, total_epochs, min_eps, max_eps):
        self.total_epochs = total_epochs
        self.min_eps = min_eps
        self.max_eps = max_eps

    def get_epsilon(self, current_epoch):
        progress = min(current_epoch / (self.total_epochs - 1), 1.0)
        eps = self.min_eps + progress * (self.max_eps - self.min_eps)
        return eps


class RandomCandGenerator():
    def __init__(self, sparsity_config):
        self.sparsity_config               = sparsity_config
        self.num_candidates_per_block = len(sparsity_config[0]) # might have bug if each block has different number of choices
        self.config_length            = len(sparsity_config)    # e.g., the len of DeiT-S is 48 (12 blocks, each has qkv, fc1, fc2, and linear projection)
        self.m = defaultdict(list)        # m: the magic dictionary with {index: cand_config}
        #random.seed(seed)
        v = []                            # v: a temp vector for function rec()
        self.rec(v, self.m)
        
    def calc(self, v):                    # generate the unique index for each candidate
        res = 0
        for i in range(self.num_candidates_per_block):
            res += i * v[i]
        return res

    def rec(self, v, m, idx=0, cur=0):    # recursively enumerate all possible candidates and attach unique indexes for them
        if idx == (self.num_candidates_per_block-1) :
            v.append(self.config_length - cur)
            m[self.calc(v)].append(copy.copy(v))
            v.pop()
            return

        i = self.config_length - cur
        while i >= 0:
            v.append(i)
            self.rec(v, m, idx+1, cur+i)
            v.pop()
            i -= 1
            
    def random(self):                     # generate a random index and return its corresponding candidate
        row = random.choice(random.choice(self.m))
        ratios = []
        for num, ratio in zip(row, [i for i in range(self.num_candidates_per_block)]):
            ratios += [ratio] * num
        random.shuffle(ratios)
        res = []
        for idx, ratio in enumerate(ratios):
            res.append(tuple(self.sparsity_config[idx][ratio])) # Fixme: 
        return res                        # return a cand_config




class TradeOffLoss:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, err, efficiency):
        #log_efficiency = math.log(efficiency)
        score = err * self.alpha * math.pow(efficiency, self.beta)
        return score

if __name__ == '__main__':
    # # Create the SearchSpace instance with candidate choices, epsilon, and promising pool size
    search_space = CandidatePool(candidate_pool_size=3)

    # # Add some subnets to candidate_pools with their scores (you can do this during the search process)
    search_space.add_one_subnet_with_score_and_flops(subnet=[[2, 4], [2, 4]], score=0.91, flops = 1)
    search_space.add_one_subnet_with_score_and_flops(subnet=[[1, 4], [2, 4]], score=0.89, flops = 2)
    search_space.add_one_subnet_with_score_and_flops(subnet=[[3, 4], [2, 4]], score=0.6, flops = 3)
    search_space.add_one_subnet_with_score_and_flops(subnet=[[2, 4], [1, 3]], score=0.6, flops = 4)
    search_space.add_one_subnet_with_score_and_flops(subnet=[[3, 4], [2, 4]], score=0.90, flops = 3)
    # # ... (add more subnets with scores)
    print(search_space.candidate_pools)
    # print(search_space.candidate_pools)
    # s = search_space.state_dict()
    # search_space.add_one_subnet_with_score(subnet=[[2, 4], [2, 4]], score=0.5)
    # search_space_new = CandidatePool(candidate_pool_size=2)
    # print(s)
    # search_space_new.load_state_dict(s)
    #print(search_space_new.candidate_pools)


    #eps_scheduler = LinearEpsilonScheduler(100, 0, 0.8)
    #for i in range(100):
    #    print(eps_scheduler.get_epsilon(i))
