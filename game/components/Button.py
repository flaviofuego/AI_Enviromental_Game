import pygame

class Button(pygame.sprite.Sprite):
    def __init__(self, image: str, scale: tuple, position: tuple):
        super(Button, self).__init__()
        self.image = pygame.image.load(f"game/assets/{image}.png")
        if not self.image:
            raise ValueError(f"Image at {image} could not be loaded.")
        
        self.scale = scale
        self.image = pygame.transform.scale(self.image, scale)
        self.rect = self.image.get_rect(center=position)
        self.clicked = False

    def update_image(self, image):
        self.image = pygame.image.load(f"game/assets/{image}.png")
        self.image = pygame.transform.scale(self.image, self.scale)
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

            # Change the image or color if needed when hovered
            # Scale up the image
            self.image = pygame.transform.scale(self.image, (self.scale[0] * 1.1, self.scale[1] * 1.1))
            self.rect = self.image.get_rect(center=self.rect.center)
        else:
            # Reset the image size if not hovered
            self.reset()
  
        surface.blit(self.image, self.rect)
        return action
    
    def reset(self):
        """Reset the button state."""
        self.clicked = False
        self.image = pygame.transform.scale(self.image, self.scale)

    def is_hovered(self):
        """Check if the button is hovered."""
        pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(pos)

    def is_clicked(self, pos):
        """Check if the button is clicked."""
        return self.rect.collidepoint(pos) and pygame.mouse.get_pressed()[0]