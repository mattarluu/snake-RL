import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
from collections import deque

SNAKE_LEN_GOAL = 30
STACKED_ACTIONS = 5

tableSize = 500
halfTable = int(tableSize / 2)
GRID_SIZE = 10

def collision_with_boundaries(snake_head):
    if snake_head[0] >= tableSize or snake_head[0] < 0 or snake_head[1] >= tableSize or snake_head[1] < 0:
        return True
    return False

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    return False

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }
    
    def __init__(self, render_mode=None, action_size=1, apple_spawn_radius=None, render_fps=10):
        super(SnakeEnv, self).__init__()

        self.apple_spawn_radius = apple_spawn_radius
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16 + STACKED_ACTIONS,),
            dtype=np.float32
        )
        
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.score = 0
        self.max_score = 0
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.action_size = action_size
        self.steps_since_apple = 0
        self.max_steps_without_apple = 1000  
        
        self.episode_deaths_by_wall = 0
        self.episode_deaths_by_body = 0

    def _spawn_apple(self):
        max_attempts = 100
        for _ in range(max_attempts):
            if self.apple_spawn_radius is None:
                apple_x = random.randrange(1, tableSize // GRID_SIZE) * GRID_SIZE
                apple_y = random.randrange(1, tableSize // GRID_SIZE) * GRID_SIZE
            else:
                head_grid_x = self.snake_head[0] // GRID_SIZE
                head_grid_y = self.snake_head[1] // GRID_SIZE
                rad = self.apple_spawn_radius
                max_grid_idx = (tableSize // GRID_SIZE) - 1

                min_grid_x = max(1, head_grid_x - rad)
                max_grid_x = min(max_grid_idx, head_grid_x + rad)
                min_grid_y = max(1, head_grid_y - rad)
                max_grid_y = min(max_grid_idx, head_grid_y + rad)

                if min_grid_x > max_grid_x: min_grid_x = max_grid_x
                if min_grid_y > max_grid_y: min_grid_y = max_grid_y

                apple_x = random.randrange(min_grid_x, max_grid_x + 1) * GRID_SIZE
                apple_y = random.randrange(min_grid_y, max_grid_y + 1) * GRID_SIZE

            apple_position = [apple_x, apple_y]
            if apple_position not in self.snake_position:
                return apple_position
        
        return [random.randrange(1, tableSize // GRID_SIZE) * GRID_SIZE,
                random.randrange(1, tableSize // GRID_SIZE) * GRID_SIZE]
    
    def _get_observation(self):
        head_x, head_y = self.snake_head
        apple_x, apple_y = self.apple_position
        
        #normalizada
        norm_head_x = (head_x / tableSize) * 2 - 1
        norm_head_y = (head_y / tableSize) * 2 - 1
        
        #vec a la manzana
        apple_delta_x = (apple_x - head_x) / tableSize
        apple_delta_y = (apple_y - head_y) / tableSize
        distance_to_apple = np.sqrt(apple_delta_x**2 + apple_delta_y**2)
        
        #angulo a la manzana
        angle_to_apple = np.arctan2(apple_delta_y, apple_delta_x) / np.pi  #normalizado [-1, 1]
        
        #distancia a paredes
        dist_to_left_wall = head_x / tableSize
        dist_to_right_wall = (tableSize - head_x) / tableSize
        dist_to_top_wall = head_y / tableSize
        dist_to_bottom_wall = (tableSize - head_y) / tableSize
        
        #deteccion de peligros
        danger_up = self._check_danger_in_direction(0, -1)
        danger_down = self._check_danger_in_direction(0, 1)
        danger_left = self._check_danger_in_direction(-1, 0)
        danger_right = self._check_danger_in_direction(1, 0)
        
        #longitud
        snake_length_norm = len(self.snake_position) / SNAKE_LEN_GOAL
        
        #direccion actual
        
        observation = [
            norm_head_x, norm_head_y,
            apple_delta_x, apple_delta_y, distance_to_apple, angle_to_apple,
            dist_to_left_wall, dist_to_right_wall, dist_to_top_wall, dist_to_bottom_wall,
            danger_up, danger_down, danger_left, danger_right,
            snake_length_norm,
            min(self.steps_since_apple / 100.0, 1.0)
        ] + list(self.prev_actions)
        
        return np.array(observation, dtype=np.float32)
    
    def _check_danger_in_direction(self, dx, dy):
        head_x, head_y = self.snake_head
        max_danger = 0.0
        look_ahead = 3  #mirar 3 pasos adelante
        
        for step in range(1, look_ahead + 1):
            check_x = head_x + (dx * GRID_SIZE * step)
            check_y = head_y + (dy * GRID_SIZE * step)
            
            #peligro de pared
            if check_x < 0 or check_x >= tableSize or check_y < 0 or check_y >= tableSize:
                danger = 1.0 / step  #más cercano = más peligroso
                max_danger = max(max_danger, danger)
                break
            
            #peligro de cuerpo
            if [check_x, check_y] in self.snake_position[1:]:  
                danger = 1.0 / step
                max_danger = max(max_danger, danger)
                break
        
        return min(max_danger, 1.0)
    
    def step(self, action):
        self.prev_actions.append(action)
        self.steps_since_apple += 1
        
        prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        prev_head = list(self.snake_head)
        self._take_action(action)
        reward = 0
        terminated = False
        truncated = False
        if self.snake_head == self.apple_position:
            self.apple_position = self._spawn_apple()
            self.score += 1
            self.snake_position.insert(0, list(self.snake_head))
            reward = 100
            self.steps_since_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        if collision_with_boundaries(self.snake_head):
            terminated = True
            reward = -100
            self.episode_deaths_by_wall += 1
        elif collision_with_self(self.snake_position):
            terminated = True
            reward = -100
            self.episode_deaths_by_body += 1
        else:
  
            current_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            
            if current_distance < prev_distance:
                reward += 2.0  
            else:
                reward -= 2.0  
            
            reward += 0.1
        
        #timeout
        if self.steps_since_apple > self.max_steps_without_apple:
            truncated = True
            reward -= 50

        info = {
            'score': self.score,
            'steps_since_apple': self.steps_since_apple,
            'deaths_by_wall': self.episode_deaths_by_wall,
            'deaths_by_body': self.episode_deaths_by_body
        }
        
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"Nuevo máximo score: {self.max_score}")
        
        #stats de muerte
        if self.episode_deaths_by_wall + self.episode_deaths_by_body > 0:
            total_deaths = self.episode_deaths_by_wall + self.episode_deaths_by_body
            wall_pct = (self.episode_deaths_by_wall / total_deaths) * 100 if total_deaths > 0 else 0
            body_pct = (self.episode_deaths_by_body / total_deaths) * 100 if total_deaths > 0 else 0
            print(f"Muertes - Pared: {wall_pct:.0f}%, Cuerpo: {body_pct:.0f}%")
        
        self.episode_deaths_by_wall = 0
        self.episode_deaths_by_body = 0
        
        #posicion inicial aleatoria
        start_x = random.randrange(10, 40) * GRID_SIZE
        start_y = random.randrange(10, 40) * GRID_SIZE
        
        self.snake_position = [
            [start_x, start_y],
            [start_x - GRID_SIZE, start_y],
            [start_x - 2*GRID_SIZE, start_y]
        ]
        
        self.snake_head = [start_x, start_y]
        self.apple_position = self._spawn_apple()
        
        self.score = 0
        self.direction = 1  #Derecha
        self.steps_since_apple = 0

        self.prev_actions = deque(maxlen=STACKED_ACTIONS)
        for _ in range(STACKED_ACTIONS):
            self.prev_actions.append(-1)

        observation = self._get_observation()
        
        return observation, {}

    def render(self, mode='human'):
        if mode is None:
            mode = self.render_mode

        self._update_ui()

        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000/self.render_fps))
        elif mode == 'rgb_array':
            return self.img.copy()

    def close(self):
        cv2.destroyAllWindows()

    def _update_ui(self):
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + GRID_SIZE, self.apple_position[1] + GRID_SIZE),
            (0, 0, 255),
            -1
        )
    
        for i, position in enumerate(self.snake_position):
            color = (0, 255, 255) if i == 0 else (0, 255, 0)
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + GRID_SIZE, position[1] + GRID_SIZE),
                color,
                -1
            )
        
        cv2.putText(self.img, f'Score: {self.score}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.img, f'Max: {self.max_score}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.img, f'Steps: {self.steps_since_apple}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _take_action(self, action):
        #Prevenir movimientos opuestos
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        self.direction = action

        if action == 0:  # Izquierda
            self.snake_head[0] -= GRID_SIZE * self.action_size
        elif action == 1:  # Derecha
            self.snake_head[0] += GRID_SIZE * self.action_size
        elif action == 2:  # Abajo
            self.snake_head[1] += GRID_SIZE * self.action_size
        elif action == 3:  # Arriba
            self.snake_head[1] -= GRID_SIZE * self.action_size