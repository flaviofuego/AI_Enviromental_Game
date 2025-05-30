import pygame
import os

class LevelThumbnail:
    def __init__(self, level_id, size=(350, 200)):
        """
        Inicializa una miniatura de nivel
        
        Args:
            level_id (int): ID del nivel
            size (tuple): Tamaño de la miniatura (ancho, alto)
        """
        self.level_id = level_id
        self.size = size
        self.image = None
        self.load_thumbnail()
    
    def load_thumbnail(self):
        """Carga la imagen de la miniatura desde la carpeta de assets"""
        try:
            # Construir la ruta a la miniatura del nivel
            thumbnail_path = os.path.join('game', 'assets', 'niveles', f'nivel_{self.level_id}.png')
            
            # Cargar y escalar la imagen
            original_image = pygame.image.load(thumbnail_path)
            self.image = pygame.transform.scale(original_image, self.size)
        except (pygame.error, FileNotFoundError):
            # Si no se encuentra la imagen, crear una miniatura por defecto
            self.create_default_thumbnail()
    
    def create_default_thumbnail(self):
        """Crea una miniatura por defecto cuando no se encuentra la imagen"""
        # Crear superficie con color de fondo
        self.image = pygame.Surface(self.size)
        self.image.fill((40, 40, 60))  # Color de fondo oscuro
        
        # Dibujar un marco
        pygame.draw.rect(self.image, (100, 100, 150), self.image.get_rect(), 2)
        
        # Añadir texto "Nivel X"
        font = pygame.font.Font(None, 24)
        #text = font.render(f"Nivel {self.level_id}", True, (200, 200, 220))
        #text_rect = text.get_rect(center=(self.size[0]/2, self.size[1]/2))
        #self.image.blit(text, text_rect)
    
    def draw(self, surface, position, is_locked=False, is_completed=False):
        """
        Dibuja la miniatura en la superficie especificada
        
        Args:
            surface (pygame.Surface): Superficie donde dibujar
            position (tuple): Posición (x, y) donde dibujar
            is_locked (bool): Si el nivel está bloqueado
            is_completed (bool): Si el nivel está completado
        """
        # Crear una copia de la imagen para modificar
        display_image = self.image.copy()
        
        if is_locked:
            # Aplicar efecto de oscurecimiento si está bloqueado
            dark = pygame.Surface(self.size, pygame.SRCALPHA)
            dark.fill((0, 0, 0, 180))  # Negro semi-transparente
            display_image.blit(dark, (0, 0))
            
            # Añadir icono de candado
            lock_size = 30
            lock_surface = pygame.Surface((lock_size, lock_size), pygame.SRCALPHA)
            pygame.draw.rect(lock_surface, (200, 200, 220, 200), 
                           (lock_size//4, lock_size//4, lock_size//2, lock_size//2))
            pygame.draw.rect(lock_surface, (200, 200, 220, 200), 
                           (lock_size//3, 0, lock_size//3, lock_size//2))
            lock_pos = (self.size[0]//2 - lock_size//2, self.size[1]//2 - lock_size//2)
            display_image.blit(lock_surface, lock_pos)
        
        elif is_completed:
            # Añadir marca de completado
            check_size = 30
            check_surface = pygame.Surface((check_size, check_size), pygame.SRCALPHA)
            pygame.draw.circle(check_surface, (50, 200, 50, 200), 
                             (check_size//2, check_size//2), check_size//2)
            check_pos = (self.size[0] - check_size - 5, 5)
            display_image.blit(check_surface, check_pos)
        
        # Dibujar la imagen en la superficie
        surface.blit(display_image, position) 