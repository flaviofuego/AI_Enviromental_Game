import pygame

class Card:
    def __init__(self, 
                 max_width=None,
                 max_height=None,
                 spacing=20,
                 padding=20,
                 bg_color=(20, 20, 40, 150),
                 border_color=(173, 216, 230),
                 border_width=2):
        """
        Inicializa una tarjeta dinámica
        
        Args:
            max_width (int): Ancho máximo de la tarjeta (None para sin límite)
            max_height (int): Alto máximo de la tarjeta (None para sin límite)
            spacing (int): Espacio entre elementos internos
            padding (int): Espacio interno desde el borde
            bg_color (tuple): Color de fondo (R,G,B,A)
            border_color (tuple): Color del borde
            border_width (int): Ancho del borde
        """
        self.max_width = max_width
        self.max_height = max_height
        self.spacing = spacing
        self.padding = padding
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        
        # Dimensiones actuales (se calculan al dibujar)
        self.width = 0
        self.height = 0
        self.content_width = 0
        self.content_height = 0
        
    def calculate_dimensions(self, image_size, text_surface):
        """
        Calcula las dimensiones de la tarjeta basadas en su contenido
        
        Args:
            image_size (tuple): (ancho, alto) de la imagen
            text_surface (pygame.Surface): Superficie del texto
        """
        # Calcular ancho necesario
        content_width = max(image_size[0], text_surface.get_width())
        self.width = content_width + (self.padding * 2)
        
        # Limitar ancho si hay máximo
        if self.max_width:
            self.width = min(self.width, self.max_width)
            content_width = self.width - (self.padding * 2)
        
        # Calcular alto necesario
        self.content_height = image_size[1] + self.spacing + text_surface.get_height()
        self.height = self.content_height + (self.padding * 2)
        
        # Limitar alto si hay máximo
        if self.max_height:
            self.height = min(self.height, self.max_height)
        
        return self.width, self.height
        
    def draw(self, surface, position, image, text, text_color, is_selected=False, glow_intensity=0):
        """
        Dibuja la tarjeta con su contenido
        
        Args:
            surface (pygame.Surface): Superficie donde dibujar
            position (tuple): Posición (x, y) donde dibujar
            image (pygame.Surface): Imagen a mostrar
            text (str): Texto a mostrar
            text_color (tuple): Color del texto (R,G,B)
            is_selected (bool): Si la tarjeta está seleccionada
            glow_intensity (float): Intensidad del brillo (0-1)
        """
        # Crear superficie para el texto con tamaño adaptativo
        font_size = 32 if text == "BLOQUEADO" else 24
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, text_color)
        
        # Calcular dimensiones
        self.calculate_dimensions(image.get_size(), text_surface)
        
        # Crear superficie de la tarjeta con alpha
        card_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        card_surface.fill(self.bg_color)
        
        # Calcular posiciones centradas
        image_x = (self.width - image.get_width()) // 2
        image_y = self.padding
        text_x = (self.width - text_surface.get_width()) // 2
        text_y = image_y + image.get_height() + self.spacing
        
        # Dibujar contenido
        card_surface.blit(image, (image_x, image_y))
        
        # Si está bloqueado, agregar fondo de alerta para el texto
        if text == "BLOQUEADO":
            # Crear fondo rojo semi-transparente para el texto
            text_bg_height = text_surface.get_height() + 10
            text_bg_width = self.width - (self.padding * 2)
            text_bg_surface = pygame.Surface((text_bg_width, text_bg_height), pygame.SRCALPHA)
            text_bg_surface.fill((220, 50, 50, 100))  # Rojo semi-transparente
            text_bg_x = self.padding
            text_bg_y = text_y - 5  # 5 píxeles arriba del texto para padding
            card_surface.blit(text_bg_surface, (text_bg_x, text_bg_y))
        
        # Dibujar el texto
        card_surface.blit(text_surface, (text_x, text_y))
        
        # Dibujar en la superficie principal
        surface.blit(card_surface, position)
        
        # Dibujar borde con brillo si está seleccionado
        if is_selected:
            border_color = list(self.border_color)
            for i in range(3):
                border_color[i] = min(255, border_color[i] + int(50 * glow_intensity))
            pygame.draw.rect(surface, border_color, 
                           (*position, self.width, self.height), 
                           self.border_width)
        else:
            pygame.draw.rect(surface, self.border_color, 
                           (*position, self.width, self.height), 
                           self.border_width)
        
        # Devolver el rectángulo para detección de clics
        return pygame.Rect(*position, self.width, self.height) 