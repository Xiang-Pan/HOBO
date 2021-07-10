from bohb import BOHB
import bohb.configspace as cs

from env import ENV



def objective(step, alpha, beta):
    return 1 / (alpha * step + 0.1) + beta


def evaluate(params, n_iterations):
    print(params)
    loss = 1 - env.env_input(params = params)
    return loss
    # loss = 0.0
    # for i in range(int(n_iterations)):
    #     loss += objective(**params, step=i)
    # return loss/n_iterations


nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 20)
# nprobe = cs.IntegerUniformHyperparameter('nprobe', [8, 16, 32])
configspace = cs.ConfigurationSpace([nprobe], seed=123)

# env = cs.Environment
env = ENV()


opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1)
logs = opt.optimize()
print(logs)
