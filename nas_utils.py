import math
import random
import heapq
import copy

class CandidatePool:
    def __init__(self, candidate_pool_size=1000):
        """
        Object for candidate pools
        
        Parameters:
            candidate_pool_size (int, optional): The maximum size of the promising pools (max-heap). Default is 1000.
        """
        self.max_pool_size = candidate_pool_size
        self.candidate_pools = []
        
    def get_one_subnet(self):
        """
        Selects a subnet
        Returns:
            selected_subnet: The selected subnet (from `candidate_pools`).
        """
        assert len(self.candidate_pool_size), "CandidatePool is empty, no subnet inside"
        selected_subnet = random.choice(self.candidate_pools)[1]  # Extract only the subnet (index 1)
        return selected_subnet

    def add_one_subnet_with_score(self, subnet, score):
        """
        Adds a subnet to the `candidate_pools` if it belongs to the top candidates.

        Parameters:
            subnet: The subnet to add to the promising pools.
            score: The score of the subnet (used for max-heap comparison).
        """
        heapq.heappush(self.candidate_pools, (score, subnet))
        self.candidate_pools = self.candidate_pools[:min(len(self.candidate_pools), self.max_pool_size)]

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
    # search_space = CandidatePool(candidate_pool_size=2)

    # # Add some subnets to candidate_pools with their scores (you can do this during the search process)
    # search_space.add_one_subnet_with_score(subnet=[[2, 4], [2, 4]], score=0.91)
    # search_space.add_one_subnet_with_score(subnet=[[2, 4], [2, 4]], score=0.89)
    # search_space.add_one_subnet_with_score(subnet=[[2, 4], [2, 4]], score=0.92)
    # # ... (add more subnets with scores)

    # print(search_space.candidate_pools)
    # s = search_space.state_dict()
    # search_space.add_one_subnet_with_score(subnet=[[2, 4], [2, 4]], score=0.5)
    # search_space_new = CandidatePool(candidate_pool_size=2)
    # print(s)
    # search_space_new.load_state_dict(s)
    # print(search_space_new.candidate_pools)
    
    
    eps_scheduler = LinearEpsilonScheduler(100, 0, 0.8)
    for i in range(100):
        print(eps_scheduler.get_epsilon(i))