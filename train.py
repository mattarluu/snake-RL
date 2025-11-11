# train_curriculum_radius.py (Modificado)

import gymnasium as gym
import snake_env  # <-- IMPORTANTE: Esto registra los entornos
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. CREACIÓN DEL CALLBACK PERSONALIZADO ---
class SaveOnBestScoreCallback(BaseCallback):
    """
    Un callback personalizado para guardar el modelo cada vez que
    se alcanza un nuevo récord de 'score' (manzanas comidas).
    """
    def __init__(self, save_path, verbose=1):
        super(SaveOnBestScoreCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_score = -1  # Empezamos con -1 para guardar en el primer score

    def _init_callback(self):
        # Asegurarse de que el directorio de guardado existe
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self):
        # self.dones es un array (uno por entorno); [0] accede al primero.
        # Comprueba si un episodio acaba de terminar.
        if self.locals["dones"][0]:
            # Acceder al 'score' desde el diccionario 'info'
            # self.locals['infos'] es una lista; [0] accede al primer entorno
            current_score = self.locals['infos'][0].get('score', 0)
            
            # Si el score actual es mejor que el mejor guardado
            if current_score > self.best_score:
                self.best_score = current_score
                
                if self.verbose > 0:
                    print(f"\n¡Nuevo récord de score! {self.best_score}. Guardando modelo en {self.save_path}")
                
                # Guardar el modelo
                self.model.save(self.save_path)
        
        return True  # Continuar el entrenamiento

# --- 2. SCRIPT DE ENTRENAMIENTO CON EL CALLBACK ---

# Asegurarse de que la carpeta 'models' existe
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Crear UNA instancia del callback ---
# Esta única instancia se usará en las 3 fases,
# por lo que 'self.best_score' seguirá aumentando.
best_model_callback = SaveOnBestScoreCallback(save_path=os.path.join(MODELS_DIR, "best_model.zip"))

# --- FASE 1: Entorno Fácil (Radio 5) ---
print("--- Iniciando Fase 1: Radio Fácil (Radio 5) ---")
env_easy = gym.make("Snake-Radius-Easy-v0")
model = PPO("MlpPolicy", env_easy, verbose=1, tensorboard_log="./snake_tensorboard/")

# Entrena en el entorno fácil, usando el callback
model.learn(
    total_timesteps=300000,
    tb_log_name="PPO_Snake_Radius_Easy",
    callback=best_model_callback  # <-- AÑADIDO
)
# Guardamos el modelo final de la fase para transferir el aprendizaje
model.save(os.path.join(MODELS_DIR, "snake_radius_easy_final"))
print("Fase 1 completada.")

# --- FASE 2: Entorno Medio (Radio 15) ---
print("\n--- Iniciando Fase 2: Radio Medio (Radio 15) ---")
# CORRECCIÓN: El ID del entorno no incluye la carpeta de modelos
env_medium = gym.make("Snake-Radius-Medium-v0")
model.set_env(env_medium)

# Continúa el entrenamiento (reset_num_timesteps=False es clave)
model.learn(
    total_timesteps=500000,
    tb_log_name="PPO_Snake_Radius_Medium",
    reset_num_timesteps=False,
    callback=best_model_callback  # <-- AÑADIDO (misma instancia)
)
model.save(os.path.join(MODELS_DIR, "snake_radius_medium_final"))
print("Fase 2 completada.")

# --- FASE 3: Entorno Difícil (Radio Completo) ---
print("\n--- Iniciando Fase 3: Radio Difícil (Completo) ---")
# CORRECCIÓN: El ID del entorno no incluye la carpeta de modelos
env_hard = gym.make("Snake-Radius-Hard-v0") 
model.set_env(env_hard)

# Entrenamiento final
model.learn(
    total_timesteps=1000000,
    tb_log_name="PPO_Snake_Radius_Hard",
    reset_num_timesteps=False,
    callback=best_model_callback  # <-- AÑADIDO (misma instancia)
)
model.save(os.path.join(MODELS_DIR, "snake_radius_final"))

print("\n¡Entrenamiento con currículo de radio completado!")
print(f"El mejor modelo (basado en score) se ha guardado en {os.path.join(MODELS_DIR, 'best_model.zip')}")