import json
import os
import datetime
import uuid
from pathlib import Path

class GameSaveSystem:
    """Sistema para guardar y cargar datos de usuario de forma persistente"""
    
    def __init__(self, save_dir=None):
        """
        Inicializa el sistema de guardado
        
        Args:
            save_dir (str, optional): Directorio donde guardar perfiles. Por defecto en 'saves'.
        """
        # Configurar directorio de guardado
        if save_dir is None:
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.save_dir = base_dir / 'saves'
        else:
            self.save_dir = Path(save_dir)
        
        # Crear directorio si no existe
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Archivo de listado de perfiles
        self.profiles_file = self.save_dir / 'profiles.json'
        
        # Cargar lista de perfiles o crear una nueva
        self.profiles_list = self._load_profiles_list()
        
        # Perfil actual
        self.current_profile = None
    
    def _load_profiles_list(self):
        """Carga la lista de perfiles existentes"""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error al leer el archivo de perfiles. Creando nuevo.")
                return {"profiles": []}
        else:
            return {"profiles": []}
    
    def _save_profiles_list(self):
        """Guarda la lista de perfiles"""
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.profiles_list, f, ensure_ascii=False, indent=2)
    
    def create_profile(self, player_name):
        """
        Crea un nuevo perfil de usuario
        
        Args:
            player_name (str): Nombre del jugador
            
        Returns:
            str: ID del perfil creado
        """
        # Crear ID único
        profile_id = str(uuid.uuid4())
        
        # Fecha y hora actuales
        now = datetime.datetime.now().isoformat()
        
        # Estructura del perfil inicial
        profile = {
            "profile_id": profile_id,
            "player_name": player_name,
            "created_date": now,
            "last_played": now,
            "total_points": 0,
            "planetary_progress": {
                "oceanos_limpiados": 0,
                "ozono_restaurado": 0,
                "aire_purificado": 0,
                "bosques_replantados": 0,
                "ciudades_enfriadas": 0
            },
            "levels": {
                "unlocked": 1,
                "current": 1,
                "completed": []
            },
            "enemies_defeated": [],
            "achievements": [],
            "settings": {
                "music_volume": 0.7,
                "sfx_volume": 0.8,
                "difficulty": "normal"
            },
            "stats": {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "time_played": 0
            }
        }
        
        # Guardar perfil en archivo individual
        self._save_profile(profile)
        
        # Añadir a la lista de perfiles
        self.profiles_list["profiles"].append({
            "profile_id": profile_id,
            "player_name": player_name,
            "created_date": now,
            "last_played": now,
            "total_points": 0
        })
        
        self._save_profiles_list()
        self.current_profile = profile
        
        return profile_id
    
    def load_profile(self, profile_id):
        """
        Carga un perfil existente
        
        Args:
            profile_id (str): ID del perfil a cargar
            
        Returns:
            dict: Datos del perfil o None si no existe
        """
        profile_path = self.save_dir / f"{profile_id}.json"
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    self.current_profile = profile
                    return profile
            except json.JSONDecodeError:
                print(f"Error al cargar perfil {profile_id}")
                return None
        else:
            print(f"Perfil {profile_id} no encontrado")
            return None
    
    def save_current_profile(self):
        """Guarda el perfil actual"""
        if self.current_profile:
            # Actualizar fecha de último uso
            self.current_profile["last_played"] = datetime.datetime.now().isoformat()
            self._save_profile(self.current_profile)
            
            # Actualizar también en la lista
            profile_id = self.current_profile["profile_id"]
            for profile in self.profiles_list["profiles"]:
                if profile["profile_id"] == profile_id:
                    profile["last_played"] = self.current_profile["last_played"]
                    profile["total_points"] = self.current_profile["total_points"]
                    profile["player_name"] = self.current_profile["player_name"]
                    break
            
            self._save_profiles_list()
            return True
        return False
    
    def _save_profile(self, profile):
        """Guarda un perfil en archivo individual"""
        profile_id = profile["profile_id"]
        profile_path = self.save_dir / f"{profile_id}.json"
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    
    def get_all_profiles(self):
        """
        Obtiene lista de todos los perfiles disponibles, ordenados por última vez jugado
        
        Returns:
            list: Lista de perfiles resumidos
        """
        # Asegurarse de que los perfiles estén ordenados por última vez jugado
        self.profiles_list["profiles"].sort(
            key=lambda x: x.get("last_played", ""),
            reverse=True  # El más reciente primero
        )
        return self.profiles_list["profiles"]
    
    def get_last_used_profile(self):
        """
        Obtiene y carga el último perfil usado
        
        Returns:
            dict: Datos del último perfil usado o None si no hay perfiles
        """
        profiles = self.get_all_profiles()
        if profiles:
            return self.load_profile(profiles[0]["profile_id"])
        return None
    
    def delete_profile(self, profile_id):
        """
        Elimina un perfil
        
        Args:
            profile_id (str): ID del perfil a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        # Eliminar archivo
        profile_path = self.save_dir / f"{profile_id}.json"
        if os.path.exists(profile_path):
            os.remove(profile_path)
        
        # Eliminar de la lista
        self.profiles_list["profiles"] = [p for p in self.profiles_list["profiles"] 
                                        if p["profile_id"] != profile_id]
        self._save_profiles_list()
        
        # Limpiar perfil actual si es el eliminado
        if self.current_profile and self.current_profile["profile_id"] == profile_id:
            self.current_profile = None
        
        return True
    
    def update_game_progress(self, level_data):
        """
        Actualiza el progreso del juego en el perfil actual
        
        Args:
            level_data (dict): Datos del nivel completado
            
        Returns:
            bool: True si se actualizó correctamente
        """
        if not self.current_profile:
            return False
            
        # Actualizar puntos totales
        if "points" in level_data:
            self.current_profile["total_points"] += level_data["points"]
        
        # Actualizar progreso planetario si existe
        if "planetary_progress" in level_data:
            for key, value in level_data["planetary_progress"].items():
                if key in self.current_profile["planetary_progress"]:
                    # Tomar el valor máximo para que siempre progrese
                    self.current_profile["planetary_progress"][key] = max(
                        self.current_profile["planetary_progress"][key],
                        value
                    )
        
        # Actualizar enemigos derrotados
        if "enemy_defeated" in level_data and level_data["enemy_defeated"]:
            enemy = level_data["enemy_defeated"]
            if enemy not in self.current_profile["enemies_defeated"]:
                self.current_profile["enemies_defeated"].append(enemy)
        
        # Actualizar niveles completados
        if "level_completed" in level_data and level_data["level_completed"]:
            level = level_data["level_completed"]
            if level not in self.current_profile["levels"]["completed"]:
                self.current_profile["levels"]["completed"].append(level)
                
            # Desbloquear siguiente nivel si corresponde
            next_level = level + 1
            if next_level > self.current_profile["levels"]["unlocked"]:
                self.current_profile["levels"]["unlocked"] = next_level
                
            # Actualizar nivel actual
            self.current_profile["levels"]["current"] = next_level
        
        # Actualizar estadísticas
        if "stats" in level_data:
            for key, value in level_data["stats"].items():
                if key in self.current_profile["stats"]:
                    self.current_profile["stats"][key] += value
        
        # Guardar cambios
        self.save_current_profile()
        return True
    
    def update_settings(self, settings):
        """
        Actualiza las configuraciones del perfil actual
        
        Args:
            settings (dict): Nuevos valores de configuración
            
        Returns:
            bool: True si se actualizó correctamente
        """
        if not self.current_profile:
            return False
            
        for key, value in settings.items():
            if key in self.current_profile["settings"]:
                self.current_profile["settings"][key] = value
                
        self.save_current_profile()
        return True
    
    def get_current_profile_data(self):
        """
        Obtiene los datos del perfil actual
        
        Returns:
            dict: Datos del perfil o None si no hay perfil activo
        """
        return self.current_profile
