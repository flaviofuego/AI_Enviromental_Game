import pygame
import os
from constants import get_screen_dimensions, get_scale_factors

class SpriteLoader:
    """Clase para cargar y gestionar sprites personalizados por nivel"""
    
    @staticmethod
    def load_sprite(image_path, target_size=None, preserve_aspect=True):
        """
        Carga una imagen y la escala al tamaño objetivo
        
        Args:
            image_path: Ruta de la imagen
            target_size: Tupla (width, height) del tamaño objetivo
            preserve_aspect: Si mantener la proporción de aspecto
            
        Returns:
            pygame.Surface con la imagen cargada y escalada
        """
        try:
            # Cargar imagen
            image = pygame.image.load(image_path).convert_alpha()
            
            if target_size is None:
                return image
                
            # Escalar imagen
            if preserve_aspect:
                # Calcular escala manteniendo proporción
                img_width, img_height = image.get_size()
                target_width, target_height = target_size
                
                scale_x = target_width / img_width
                scale_y = target_height / img_height
                scale = min(scale_x, scale_y)
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                return pygame.transform.smoothscale(image, (new_width, new_height))
            else:
                # Escalar directamente al tamaño objetivo
                return pygame.transform.smoothscale(image, target_size)
                
        except pygame.error as e:
            print(f"Error cargando sprite {image_path}: {e}")
            return None
    
    @staticmethod
    def create_circular_mask(surface, radius):
        """
        Crea una máscara circular para un sprite
        
        Args:
            surface: La superficie del sprite
            radius: Radio del círculo
            
        Returns:
            pygame.mask.Mask circular
        """
        mask_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(mask_surface, (255, 255, 255, 255), (radius, radius), radius)
        return pygame.mask.from_surface(mask_surface)
    
    @staticmethod
    def load_level_sprites(level_id):
        """
        Carga todos los sprites de un nivel específico
        
        Args:
            level_id: ID del nivel
            
        Returns:
            dict con todos los sprites cargados
        """
        from game.config.level_config import get_asset_path
        
        sprites = {}
        scale_x, scale_y = get_scale_factors()
        
        # Cargar fondo
        bg_path = get_asset_path(level_id, "cancha.png")
        if os.path.exists(bg_path):
            current_width, current_height = get_screen_dimensions()
            sprites['background'] = SpriteLoader.load_sprite(bg_path, (current_width, current_height), False)
        
        # Cargar puck
        puck_path = get_asset_path(level_id, "puck.png")
        if os.path.exists(puck_path):
            puck_size = int(30 * min(scale_x, scale_y))  # Diámetro del puck
            sprites['puck'] = SpriteLoader.load_sprite(puck_path, (puck_size, puck_size))
        
        # Cargar mallet IA
        mallet_path = get_asset_path(level_id, "mallet_IA.png")
        if os.path.exists(mallet_path):
            mallet_size = int(64 * min(scale_x, scale_y))  # Diámetro del mallet
            sprites['mallet_ai'] = SpriteLoader.load_sprite(mallet_path, (mallet_size, mallet_size))
            # Crear versión para el jugador (con tinte diferente)
            sprites['mallet_player'] = sprites['mallet_ai'].copy()
        
        # Cargar porterías
        goal_left_path = get_asset_path(level_id, "porteria_izq.png")
        if os.path.exists(goal_left_path):
            sprites['goal_left'] = SpriteLoader.load_sprite(goal_left_path)
        
        goal_right_path = get_asset_path(level_id, "porteria_der.png")
        if os.path.exists(goal_right_path):
            sprites['goal_right'] = SpriteLoader.load_sprite(goal_right_path)
        
        return sprites
    
    @staticmethod
    def apply_tint(surface, color):
        """
        Aplica un tinte de color a una superficie
        
        Args:
            surface: Superficie a teñir
            color: Color RGB para el tinte
            
        Returns:
            Nueva superficie con el tinte aplicado
        """
        tinted = surface.copy()
        tinted.fill(color + (128,), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted 