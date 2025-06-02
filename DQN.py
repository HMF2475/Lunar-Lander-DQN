import numpy as np
import random
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from collections import deque

from lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.state_size= state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        # Definimos la arquitectura de la red neuronal
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # Capa 1: input → primera oculta
            nn.ReLU(),                           # Activación para la primera capa
            nn.Linear(hidden_size, hidden_size), # Capa 2: segunda oculta
            nn.ReLU(),                           # Activación para la segunda capa
            nn.Linear(hidden_size, action_size)  # Capa de salida
        )

    def forward(self, x):
        return self.model(x)
  
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos

    def push(self, state, action, reward, next_state, done):
        # insert into buffer
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience) #Como usamos un deque no necesitamos borrar el elemento mas viejo, ya que lo borra solo al meter uno que sobrepase el tamaño del buffer
        

    def sample(self, batch_size):
        # get a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
 
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.0005, batch_size=64, 
                memory_size=100000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=32, hidden_size=256):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """
        



        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode

        # Para poder usar la GPU (que es más rápida para estos calculos), tenemos que definir el dispositivo
        # Si no hay GPU, se usara la CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.q_network = DQN(
            state_size=observation_space.shape[0], # de entrada: la cantidad de datos que tenemos para definir cada estado
            action_size=action_space.n, #de salida: la cantidad de acciones que podemos tomar
            hidden_size=hidden_size #elegir un tamaño de capa oculta
        ).to(self.device)
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=hidden_size #elegir un tamaño de capa oculta
        ).to(self.device)
        



        # Set weights of target network to be the same as those of the q network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
      
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr= self.learning_rate) # Optimizador de torch 
        # Usamos Adam porque es un optimizador adaptativo que ajusta la tasa de aprendizaje para cada parametro de forma individual.
    

        print(f"QNetwork:\n {self.q_network}")



          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        current_env_state = self.lunar.state #Obtenemos el estado actual del entorno

        if random.random() < self.epsilon: #Exploracion vs Explotacion tradeoff según epsilon
            action = self.lunar.env.action_space.sample()
            # Exploracion: Tomamos una acción aleatoria del espacio de acciones
            #print(f"Exploration: Taking random action {action} with epsilon {self.epsilon}")
        else:
            
            # Explotacion: Usamos la red neuronal para predecir la acción con el Q-valor más alto para tomarla
            state_tensor = torch.FloatTensor(current_env_state).unsqueeze(0).to(self.device) # Lo convertimos en un tensor de PyTorch (unsqueeze agrega una dimensión extra para que sea compatible con la red -> las redes esperan un batch de datos, aunque sea de tamaño 1 así que con esto agregamos el batch size de 1 [bathch_size, state_size])
            
            self.q_network.eval() # Pone la red en modo evaluación para inferencia (para tomar decisiones, no entrenarse)
            
            with torch.no_grad(): # No necesitamos que se calculen gradientes, así que se deshabilita para que vaya más rapido y consumer menos mmemoria
                q_values = self.q_network(state_tensor) # Pasamos el estado a la red para obtener los Q-valores 
            
            self.q_network.train() # Volvemos a poner la red en modo entrenamiento para que pueda seguir aprendiendo

            
            action = torch.argmax(q_values, dim=1).item() #Con argmax obtenemos el índice del Q-valor más alto, que corresponde a la acción a tomar

            #print(f"Exploitation: Taking action {action} with epsilon {self.epsilon}")
       
        # Tomamos la acción en el entorno y obtenemos el siguiente estado, recompensa y si el episodio ha terminado
        # verbose=False para no imprimir el estado en cada paso
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """

        raw_states, raw_actions, raw_rewards, raw_next_states, raw_dones = self.memory.sample(self.batch_size)
        #(estado1, accion1, recompensa1, estado2, done1), (estado2, accion2, recompensa2, estado3, done2), ...

        # Convertimos los datos a tensores de PyTorch y los movemos al dispositivo correcto
        # Aseguramos que los estados sean float32. Los entornos Gymnasium suelen devolver float32 o float64.
        states_np = np.array(raw_states, dtype=np.float32)
        states = torch.tensor(states_np, device=self.device)

        next_states_np = np.array(raw_next_states, dtype=np.float32)
        next_states = torch.tensor(next_states_np, device=self.device)

        # Las acciones son índices, deben ser enteros largos (long)
        actions_np = np.array(raw_actions) 
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device).unsqueeze(1)

        rewards_np = np.array(raw_rewards, dtype=np.float32)
        rewards = torch.tensor(rewards_np, device=self.device).unsqueeze(1)

        # Los 'dones' son booleanos
        dones_np = np.array(raw_dones, dtype=bool)
        dones = torch.tensor(dones_np, dtype=torch.bool, device=self.device).unsqueeze(1)

        #Calculamos los Q-valores actuales
        current_q_values = self.q_network(states).gather(1, actions) # Obtenemos los Q-valores actuales para las acciones tomadas
        
        with torch.no_grad(): # No necesitamos calcular gradientes para el target network
            # Calculamos los Q-valores futuros usando la red objetivo
            next_q_values = self.target_network(next_states).max(dim=1)[0].unsqueeze(1)
        
        target_q_values = rewards + (self.gamma * next_q_values * (~dones)) # Calculamos los Q-valores objetivo usando la formula de Bellman

        loss_fn = nn.MSELoss() 
        loss = loss_fn(current_q_values, target_q_values)  # la perdida entre los Q-valores actuales y los Q-valores objetivo

         

        #Optimizamos la red
        self.optimizer.zero_grad()   # Borramos los gradientes anteriores
        loss.backward()              # Calculamos los nuevos gradientes
        self.optimizer.step()        # Actualizamos los parámetros con los nuevos gradientes
        # Actualizamos los pesos de la red q usando el optimizador
        
        avg_current_q = current_q_values.mean().item()
        avg_target_q = target_q_values.mean().item()
        #print(f"Loss: {loss.item()}, Avg Current Q: {avg_current_q:.2f}, Avg Target Q: {avg_target_q:.2f}") 
        
        return loss.item(), avg_current_q, avg_target_q  # Devolvemos el valor de la perdida como un escalar (sino es un tensonr) para poder imprimirlo en consola y ver como va el entrenamiento
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict()) # Sincronizar la red objetivo con la red q
        self.q_network.eval() # Se setea a modo de evaluacion
        print(f"Model loaded from {path}")
        
    def train(self, nombre_archivo="modelo_DQN", save_every_500=False, save_graphs=False):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        scores = []
        episode_logs = []  # Lista para guardar los resultados de cada episodio
        losses = []
        steps_per_episode = []

        print("Starting training...")
        # Iniciamos un temporizador para medir el tiempo de entrenamiento
        start_time = time.time()
        for episode in range(self.episodes):
            self.lunar.reset() #Estado inicial del entorno
            
 
            total_reward = 0
            done = False
            steps = 0
            max_steps_per_episode = 500 # Limite de pasos por episodio para evitar bucles infinitos
            while not done and steps < max_steps_per_episode:
                previous_state = self.lunar.state # Guardamos el estado anterior
                next_state, reward, done, action = self.act()
                self.memory.push(previous_state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                
            episode_loss = 0.0
            episode_q_current = 0.0
            episode_q_target = 0.0
            num_updates = 0

            if len(self.memory) >= self.batch_size:
                for _ in range(self.replays_per_episode):
                    loss_value, q_current, q_target = self.update_model()

                    # Para los graficos
                    episode_loss += loss_value
                    episode_q_current += q_current
                    episode_q_target += q_target
                    num_updates += 1
                    
                if num_updates > 0:
                    episode_loss /= num_updates
                    episode_q_current /= num_updates
                    episode_q_target /= num_updates
            else:
                print("Not enough samples in memory to update the model")
                break 

            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (episode + 1) % self.target_updt_freq == 0:
                print(f"Updating target network at episode {episode + 1} before training step {steps}")
                self.update_target_network()
            
            
            print(f"--------------- Episode: {episode+1}/{self.episodes} | Steps:{steps} | Score: {total_reward:.2f} | Avg Loss: {episode_loss:.4f} | Epsilon: {self.epsilon:.3f} | Q-current mean values: {episode_q_current:.2f} | Q-target mean values: {episode_q_target:.2f} ---------------")
            episode_logs.append({
                "episodio": episode + 1,
                "steps": steps,
                "score": total_reward,
                "avg_loss": episode_loss,
                "epsilon": self.epsilon,
                "avg_q_current": episode_q_current,
                "avg_q_target": episode_q_target,
            })
            scores.append(total_reward)
            losses.append(episode_loss)
            steps_per_episode.append(steps)  

            
            if save_every_500 and (episode + 1) % 500 == 0: #Que se guarde el modelo cada 500 episodios por las dudas que se apague el ordenador xd
                self.save_model(f"modelos/modelo_DQN_episode_{episode+1}.h5")
            
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        # print("\nResumen de todos los episodios:")
        # for log in episode_logs:
        #    print(f"Episodio {log['episodio']:4d} | Steps: {log['steps']:3d} | Score: {log['score']:7.2f} | Avg Loss: {log['avg_loss']:.4f} | Epsilon: {log['epsilon']:.3f} | Q-current mean values: {log['avg_q_current']:.2f} | Q-target mean values: {log['avg_q_target']:.2f}")


        # Guardamos el modelo final

        self.save_model(f"{nombre_archivo}.h5" )

        avg_q_current = [log["avg_q_current"] for log in episode_logs]
        avg_q_target = [log["avg_q_target"] for log in episode_logs]
        print_summary(scores, losses, steps_per_episode, start_time, end_time)
        plot_training_metrics(scores, losses, steps_per_episode,avg_q_current, avg_q_target, save_graphs)         

       

def print_summary(scores, losses, steps_per_episode, start_time, end_time):
    avg_score = sum(scores[-100:]) / min(100, len(scores))
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_steps = sum(steps_per_episode) / len(steps_per_episode)
    success_rate = 100 * sum([1 for s in scores if s >= 100]) / len(scores)

    print("\nRESUMEN DEL ENTRENAMIENTO:")
    print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")
    print(f"Puntaje promedio últimos 100 episodios: {avg_score:.2f}")
    print(f"Promedio de pérdida (loss): {avg_loss:.4f}")
    print(f"Promedio de pasos por episodio: {avg_steps:.2f}")
    print(f"Tasa de éxito (score ≥ 100): {success_rate:.2f}%")


def plot_training_metrics(scores, losses, steps_per_episode, avg_q_current, avg_q_target, save_graphs, filename_base="resultados lunar"):
    import pandas as pd
    import matplotlib.pyplot as plt

    window = 50
    scores_ma = pd.Series(scores).rolling(window).mean()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Score + promedio móvil
    axs[0, 0].plot(scores, label="Score")
    axs[0, 0].plot(scores_ma, label=f"Promedio móvil ({window})", color="orange")
    axs[0, 0].set_title("Score por Episodio")
    axs[0, 0].set_xlabel("Episodio")
    axs[0, 0].set_ylabel("Score")
    axs[0, 0].legend()

    # Loss
    axs[0, 1].plot(losses, label="Loss", color="red")
    axs[0, 1].set_title("Pérdida promedio por Episodio")
    axs[0, 1].set_xlabel("Episodio")
    axs[0, 1].set_ylabel("Loss")

    # Duración del episodio
    axs[1, 0].plot(steps_per_episode, label="Steps por episodio", color="purple")
    axs[1, 0].set_title("Duración del Episodio")
    axs[1, 0].set_xlabel("Episodio")
    axs[1, 0].set_ylabel("Pasos")

    # Valores Q promedio
    axs[1, 1].plot(avg_q_current, label="Q-valor actual", color="green")
    axs[1, 1].plot(avg_q_target, label="Q-valor objetivo", color="blue")
    axs[1, 1].set_title("Valores Q Promedio")
    axs[1, 1].set_xlabel("Episodio")
    axs[1, 1].set_ylabel("Valor Q")
    axs[1, 1].legend()

    plt.tight_layout()
    if(save_graphs):
        fig.savefig(f"graficos/{filename_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"graficos/{filename_base}.svg", bbox_inches='tight')
    print(f"Gráficos guardados como: {filename_base}.png y .svg")
    plt.show()


