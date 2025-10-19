from typing import List

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import time
import pygame

import numpy as np


class Env2048(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 2,
    }

    ACTIONS_ID = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT",
    }

    GRID_SIZE = 4
    ACTIONS = 4

    def __init__(self, render_mode=None, max_box_number: List[int]=[2048]):
        super(Env2048).__init__()

        self.__base_grid = self.__make_grid__()

        self.max_box_number = list(max_box_number) if not isinstance(max_box_number, list) else max_box_number

        self.observation_space = Box(-10, 10, (self.GRID_SIZE ** 2,), dtype=np.float32)

        #self.observation_space = MultiDiscrete([self.max_box_number + 1] * self.__base_grid.shape[0]**2)

        self.action_space = Discrete(self.ACTIONS)

        self.score = 0

        assert render_mode in self.metadata["render_modes"] or render_mode is None, f"Invalid render mode: {render_mode}. Available modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

        # Variabili per il controllo degli FPS
        self._last_render_time = 0
        self._render_fps = self.metadata["render_fps"]

        # Variabili per pygame
        self.window = None
        self.clock = None
        self.cell_size = 100
        self.window_size = self.GRID_SIZE * self.cell_size


    def __make_grid__(self):

        return np.array([[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)], dtype=np.int32)
    
    def __init_grid__(self):
        self.__game_grid = self.__make_grid__()

        # Add two random boxes to the grid
        for _ in range(2):
            empty_positions = np.argwhere(self.__game_grid == 0)
            if empty_positions.size > 0:
                row, col = empty_positions[np.random.choice(empty_positions.shape[0])]
                self.__game_grid[row, col] = np.random.choice([2, 4], p=[0.9, 0.1])

    def __add_new_two_or_four(self):

        empty_positions = np.argwhere(self.__game_grid == 0)
        if empty_positions.size > 0:
            row, col = empty_positions[np.random.choice(empty_positions.shape[0])]
            self.__game_grid[row, col] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.__init_grid__()
        observation = self.__game_grid.flatten()
        observation = (observation - observation.min()) / (observation.std() + 1e-10)
        self.steps = 0
        self.score = 0

        self._last_render_time = 0

        return observation.astype(np.float32), {}


    def step(self, action: int):

        move_id = action.item()

        old_score = self.score

        move = self.ACTIONS_ID[move_id]

        changed = self.__update_grid(move)

        if changed:
            self.__add_new_two_or_four()

        terminated = False
        truncated = False

        reward = self.score - old_score

        observation = self.__game_grid.flatten()

        info = {}

        obj_check = [i in set(observation) for i in self.max_box_number]

        if any(obj_check):
            terminated = True
            reward = self.score
        elif self.__check_lost():
            terminated = True
            reward = -100

        observation = (observation - observation.min()) / (observation.std() + 1e-10)

        return observation, reward, terminated, truncated, info
    

    def __check_lost(self):

        if 0 in self.__game_grid:
            return False
        
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE - 1):
                if self.__game_grid[i, j] == self.__game_grid[i, j + 1]:
                    return False
        
        for i in range(self.GRID_SIZE - 1):
            for j in range(self.GRID_SIZE):
                if self.__game_grid[i, j] == self.__game_grid[i + 1, j]:
                    return False
        
        return True

    def action_masks(self) -> np.ndarray:

        mask = np.zeros(self.ACTIONS, dtype=bool)

        original_grid = self.__game_grid.copy()
        original_score = self.score

        # UP (0)
        self.__game_grid = original_grid.copy()
        mask[0] = self.__move_up()

        # RIGHT (1)
        self.__game_grid = original_grid.copy()
        mask[1] = self.__move_right()

        # DOWN (2)
        self.__game_grid = original_grid.copy()
        mask[2] = self.__move_down()

        # LEFT (3)
        self.__game_grid = original_grid.copy()
        mask[3] = self.__move_left()

        self.__game_grid = original_grid
        self.score = original_score

        return mask


    def __compress(self):

        changed = False
        new_mat = np.zeros_like(self.__game_grid)

        for i in range(self.GRID_SIZE):
            pos = 0

            for j in range(self.GRID_SIZE):
                if (self.__game_grid[i, j] != 0):

                    new_mat[i, pos] = self.__game_grid[i, j]

                    if j != pos:
                        changed = True

                    pos += 1

        self.__game_grid = new_mat

        return changed

    def __merge_and_increment_score(self):

        changed = False
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE - 1):

                if self.__game_grid[i, j] == self.__game_grid[i, j + 1] and self.__game_grid[i, j] != 0:
                    self.__game_grid[i, j] = self.__game_grid[i, j] * 2
                    self.score += self.__game_grid[i, j]
                    self.__game_grid[i, j + 1] = 0

                    changed = True

        return changed

    def __move_left(self):

        changed1 = self.__compress()

        changed2 = self.__merge_and_increment_score()

        self.__compress()

        return changed1 or changed2

    def __move_right(self):

        self.__game_grid = np.flip(self.__game_grid, axis=1)

        changed = self.__move_left()

        self.__game_grid = np.flip(self.__game_grid, axis=1)

        return changed

    def __move_up(self):

        self.__game_grid = self.__game_grid.T

        changed = self.__move_left()

        self.__game_grid = self.__game_grid.T

        return changed

    def __move_down(self):

        self.__game_grid = self.__game_grid.T

        changed = self.__move_right()

        self.__game_grid = self.__game_grid.T

        return changed

    def __update_grid(self, move: str):

        if move == "UP":

            changed = self.__move_up()

        elif move == "DOWN":

            changed = self.__move_down()

        elif move == "LEFT":

            changed = self.__move_left()

        elif move == "RIGHT":

            changed = self.__move_right()

        return changed


    def render(self):
        """
        Renderizza l'ambiente corrente.
        
        Args:
            mode (str): Modalità di rendering ("human" o "rgb_array")
        
        Returns:
            None per mode="human", array RGB per mode="rgb_array"
        """
        # Calcola il tempo minimo tra render successivi
        min_render_interval = 1.0 / self._render_fps
        current_time = time.time()
        
        # Controlla se è passato abbastanza tempo dall'ultimo render
        time_since_last_render = current_time - self._last_render_time
        if time_since_last_render < min_render_interval:
            # Aspetta il tempo rimanente per rispettare gli FPS
            time.sleep(min_render_interval - time_since_last_render)
        
        # Aggiorna il timestamp dell'ultimo render
        self._last_render_time = time.time()
        
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Modalità di rendering non supportata: {self.render_mode}")

    def _render_human(self):
        """Renderizza la griglia usando pygame."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 60))  # +60 per il punteggio
            pygame.display.set_caption("2048 - RL Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Gestisci eventi pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                
        # Colori
        colors = {
            0: (205, 193, 180),     # Vuoto
            2: (238, 228, 218),     # 2
            4: (237, 224, 200),     # 4
            8: (242, 177, 121),     # 8
            16: (245, 149, 99),     # 16
            32: (246, 124, 95),     # 32
            64: (246, 94, 59),      # 64
            128: (237, 207, 114),   # 128
            256: (237, 204, 97),    # 256
            512: (237, 200, 80),    # 512
            1024: (237, 197, 63),   # 1024
            2048: (237, 194, 46),   # 2048
        }
        
        # Riempi il background
        self.window.fill((187, 173, 160))
        
        # Disegna il punteggio
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (119, 110, 101))
        self.window.blit(score_text, (10, 10))
        
        # Disegna la griglia
        grid_start_y = 60
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                value = self.__game_grid[i, j]
                
                # Coordinate della cella
                x = j * self.cell_size
                y = grid_start_y + i * self.cell_size
                
                # Colore della cella
                color = colors.get(value, colors[2048])

                # Disegna il rettangolo della cella
                pygame.draw.rect(self.window, color, (x, y, self.cell_size, self.cell_size))
                
                # Disegna il bordo
                pygame.draw.rect(self.window, (187, 173, 160), (x, y, self.cell_size, self.cell_size), 3)
                
                # Disegna il numero se la cella non è vuota
                if value != 0:
                    # Scegli il colore del testo
                    text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                    
                    # Scegli la dimensione del font basata sul numero di cifre
                    font_size = 55 if value < 100 else (45 if value < 1000 else 35)
                    number_font = pygame.font.Font(None, font_size)
                    
                    # Renderizza il testo
                    text = number_font.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                    self.window.blit(text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(self._render_fps)


    def _render_rgb_array(self):
            """
            Restituisce una rappresentazione RGB della griglia.
            
            Returns:
                numpy.ndarray: Array RGB della griglia (height, width, 3)
            """
            # Dimensioni dell'immagine
            cell_size = 100
            grid_size = self.GRID_SIZE * cell_size
            
            # Crea un'immagine RGB
            img = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 187  # Colore di sfondo grigio
            
            # Colori per i diversi valori delle celle
            colors = {
                0: (205, 193, 180),     # Vuoto
                2: (238, 228, 218),     # 2
                4: (237, 224, 200),     # 4
                8: (242, 177, 121),     # 8
                16: (245, 149, 99),     # 16
                32: (246, 124, 95),     # 32
                64: (246, 94, 59),      # 64
                128: (237, 207, 114),   # 128
                256: (237, 204, 97),    # 256
                512: (237, 200, 80),    # 512
                1024: (237, 197, 63),   # 1024
                2048: (237, 194, 46),   # 2048
            }
            
            # Disegna ogni cella
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    value = self.__game_grid[i, j]
                    
                    # Coordinate della cella
                    y1, y2 = i * cell_size, (i + 1) * cell_size
                    x1, x2 = j * cell_size, (j + 1) * cell_size
                    
                    # Colore della cella
                    color = colors.get(value, colors[2048])  # Default al colore del 2048
                    
                    # Riempi la cella con il colore
                    img[y1:y2, x1:x2] = color
                    
                    # Aggiungi bordi neri
                    border_width = 2
                    img[y1:y1+border_width, x1:x2] = 0  # Bordo superiore
                    img[y2-border_width:y2, x1:x2] = 0  # Bordo inferiore
                    img[y1:y2, x1:x1+border_width] = 0  # Bordo sinistro
                    img[y1:y2, x2-border_width:x2] = 0  # Bordo destro
            
            return img
    
    def close(self):
        """Chiude la finestra pygame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()