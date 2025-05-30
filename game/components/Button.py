import pygame

class Button(pygame.sprite.Sprite):
    def __init__(self, image: str, scale: tuple, position: tuple, tex_hover: str):
        super(Button, self).__init__()
        # Cargar y guardar la imagen original sin modificar
        self.original_image = pygame.image.load(f"game/assets/{image}.png")
        if not self.original_image:
            raise ValueError(f"Image at {image} could not be loaded.")
        
        self.scale = scale
        # Crear la imagen mostrada desde la original
        self.image = pygame.transform.scale(self.original_image, scale)
        self.rect = self.image.get_rect(center=position)
        self.clicked = False
        self.tex_hover = tex_hover
        self.is_hovering = False  # Estado de hover

        # Fuente
        self.font_small = pygame.font.SysFont('Arial', 14)
        self.name = image

    def update_image(self, image):
        self.original_image = pygame.image.load(f"game/assets/{image}.png")
        self.image = pygame.transform.scale(self.original_image, self.scale)
        self.rect = self.image.get_rect(center=self.rect.center)

    def draw(self, surface):
        action = False

        # Check if the mouse is hovering over the button
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] and not self.clicked:
                action = True
                self.clicked = True
            if not pygame.mouse.get_pressed()[0]:
                self.clicked = False

            # Si no estaba en hover antes, escalar desde la imagen original
            if not self.is_hovering:
                self.is_hovering = True
                # Escalar desde la imagen original, no desde la actual
                hover_scale = (int(self.scale[0] * 1.1), int(self.scale[1] * 1.1))
                self.image = pygame.transform.scale(self.original_image, hover_scale)
                self.rect = self.image.get_rect(center=self.rect.center)
        else:
            # Si estaba en hover pero ya no, resetear
            if self.is_hovering:
                self.is_hovering = False
                self.reset()

        # Mostrar texto de hover
        if self.is_hovered() and self.tex_hover:
            hover_surface = self.font_small.render(self.tex_hover, True, (255, 255, 255))
            hover_rect = hover_surface.get_rect(center=(self.rect.center[0], self.rect.center[1] + self.scale[1]/2))
            surface.blit(hover_surface, hover_rect)
  
        surface.blit(self.image, self.rect)
        return action
    
    def reset(self):
        """Reset the button state."""
        self.clicked = False
        # Volver a escalar desde la imagen original
        self.image = pygame.transform.scale(self.original_image, self.scale)
        self.rect = self.image.get_rect(center=self.rect.center)

    def is_hovered(self):
        """Check if the button is hovered."""
        pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(pos)

    def is_clicked(self, pos):
        """Check if the button is clicked."""
        return self.rect.collidepoint(pos) and pygame.mouse.get_pressed()[0]