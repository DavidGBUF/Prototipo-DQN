import random
import networkx as nx
import numpy as np


import torch
import torch.nn as nn

import torch.optim as optim

from collections import deque




# Definição da topologia da rede com capacidade de banda em cada link
network_topology = {
    1: {2: {'bandwidth': 100}, 6: {'bandwidth': 50}},
    2: {1: {'bandwidth': 100}, 3: {'bandwidth': 100}, 6: {'bandwidth': 50}},
    3: {2: {'bandwidth': 100}, 4: {'bandwidth': 100}},
    4: {3: {'bandwidth': 100}, 5: {'bandwidth': 0}},
    5: {4: {'bandwidth': 0}, 8: {'bandwidth': 50}},
    6: {1: {'bandwidth': 50}, 2: {'bandwidth': 50}, 7: {'bandwidth': 100}},
    7: {6: {'bandwidth': 100}, 8: {'bandwidth': 100}},
    8: {5: {'bandwidth': 50}, 7: {'bandwidth': 100}}
}

# Criação do grafo representando a rede
G = nx.Graph()
for node, edges in network_topology.items():
    for target, edge_attr in edges.items():
        G.add_edge(node, target, bandwidth=edge_attr['bandwidth'], weight=1)

# Recursos dos servidores
server_resources = {
    1: {'cpu': 100, 'cache': 100, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': ['s1', 's3']},
    2: {'cpu': 100, 'cache': 80, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': []},
    3: {'cpu': 100, 'cache': 60, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': []},
    4: {'cpu': 100, 'cache': 100, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': ['s2']},
    5: {'cpu': 100, 'cache': 150, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': []},
    6: {'cpu': 100, 'cache': 60, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': []},
    7: {'cpu': 100, 'cache': 100, 'allocated_cpu': 0, 'allocated_cache': 0, 'reuse': []},
    8: {'cpu': 100, 'cache': 100, 'allocated_cpu': 50, 'allocated_cache': 10, 'reuse': []},
}

# Requisitos dos serviços
service_requirements = {
    's0': {'cpu': 20, 'cache': 20, 'bandwidth_output': 10},
    's1': {'cpu': 20, 'cache': 20, 'bandwidth_output': 15},
    's2': {'cpu': 20, 'cache': 20, 'bandwidth_output': 20},
    's3': {'cpu': 20, 'cache': 20, 'bandwidth_output': 25},
    's4': {'cpu': 0, 'cache': 0, 'bandwidth_output': 0},
}

# Localização onde o último serviço deve estar
last_service_server = 8







def calculate_cost(individual):
    total_cost = 0
    # Dicionário para acompanhar o uso de recursos em cada servidor
    server_usage = {server_id: {'cpu': server['cpu'], 'cache': server['cache']} for server_id, server in server_resources.items()}
    
    # Dicionário para acompanhar o uso de banda em cada link
    link_usage = {tuple(sorted((u, v))): 0 for u, v in G.edges()}  # Arestas padronizadas

    for service_index, server_id in enumerate(individual):
        service_key = list(service_requirements.keys())[service_index]
        service = service_requirements[service_key]
        server = server_resources[server_id]

        # Verifica e acumula o uso de CPU e cache
        if service_key not in server['reuse']:
            server_usage[server_id]['cpu'] -= service['cpu']
            server_usage[server_id]['cache'] -= service['cache']
            if server_usage[server_id]['cpu'] < 0 or server_usage[server_id]['cache'] < 0:
                return float('inf')  # Recursos insuficientes
            total_cost += 1  # Exemplo de custo de CPU/Cache

    for i in range(len(individual) - 1):
        current_server = individual[i]
        next_server = individual[i + 1]
        if current_server == next_server:
            continue

        # Calcula o caminho mínimo e verifica a capacidade de banda
        path = nx.shortest_path(G, source=current_server, target=next_server, weight='weight')
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            edge = tuple(sorted((u, v)))  # Ordenar os nós para acessar de forma consistente
            link_usage[edge] += service_requirements[list(service_requirements.keys())[i]]['bandwidth_output']
            if link_usage[edge] > G[u][v]['bandwidth']:
                return float('inf')  # Banda insuficiente
        total_cost += len(path) - 1  # Adiciona custo baseado no número de links usados

    return total_cost



def display_paths_and_create_service_dict(best_individual):
    paths = []
    service_to_server_dict = {}
    service_names = list(service_requirements.keys())

    for i, server_id in enumerate(best_individual):
        service_name = service_names[i]
        if i < len(best_individual) - 1:
            next_server_id = best_individual[i + 1]
            if server_id == next_server_id:
                service_to_server_dict[service_name] = [server_id]
                paths.append([server_id])
            else:
                path = nx.shortest_path(G, source=server_id, target=next_server_id, weight='weight')
                service_to_server_dict[service_name] = path
                paths.append(path)
        else:
            service_to_server_dict[service_name] = [server_id]
            paths.append([server_id])

    return paths, service_to_server_dict




class NetworkEnv:
    def __init__(self, G, server_resources, service_requirements, last_service_server):
        self.G = G
        self.server_resources = server_resources
        self.service_requirements = service_requirements
        self.last_service_server = last_service_server
        self.current_service_index = 0
        self.individual = []
        self.current_state = self._get_state()

    def _get_state(self):
        state = []
        for server in self.server_resources.values():
            state.extend([server['cpu'], server['cache']])
        state.append(self.current_service_index)
        return np.array(state)

    def step(self, action):
        server_id = action + 1
        self.individual.append(server_id)
        service_key = list(self.service_requirements.keys())[self.current_service_index]
        service = self.service_requirements[service_key]
        server = self.server_resources[server_id]

        if service_key not in server['reuse']:
            if server['cpu'] < service['cpu'] or server['cache'] < service['cache']:
                return self.current_state, -100, True
            server['cpu'] -= service['cpu']
            server['cache'] -= service['cache']

        self.current_service_index += 1
        done = self.current_service_index >= len(self.service_requirements)
        self.current_state = self._get_state()
        if done:
            cost = calculate_cost(self.individual)
            reward = -cost if cost != float('inf') else -100
        else:
            reward = 0
        return self.current_state, reward, done

    def reset(self):
        self.current_service_index = 0
        self.individual = []
        for server in self.server_resources.values():
            server['cpu'] = 100
            server['cache'] = 100
        self.current_state = self._get_state()
        return self.current_state




class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)


# Função de seleção de ação com política ε-greedy
def select_action(state, epsilon, action_dim, q_network):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)  # Ação aleatória
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Adiciona dimensão de lote
        q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()  # Melhor ação


def train_dqn(env, q_network, target_network, episodes=1200, batch_size=64, gamma=0.99):
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, epsilon, len(server_resources), q_network)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = q_network(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    targets = rewards + (1 - dones) * gamma * next_q_values

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")



env = NetworkEnv(G, server_resources, service_requirements, last_service_server)
state_dim = len(env._get_state())
action_dim = len(server_resources)

q_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())

train_dqn(env, q_network, target_network)
