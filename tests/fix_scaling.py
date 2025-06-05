#!/usr/bin/env python3
"""
Script temporal para arreglar las referencias a WIDTH y HEIGHT en main_improved.py
en la función main_with_config para usar dimensiones dinámicas.
"""

import re

def fix_scaling_in_main_improved():
    file_path = "main_improved.py"
    
    # Leer el archivo
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontrar la función main_with_config
    main_with_config_pattern = r'(def main_with_config.*?)(?=def|\Z)'
    match = re.search(main_with_config_pattern, content, re.DOTALL)
    
    if match:
        function_content = match.group(1)
        original_function = function_content
        
        # Reemplazar WIDTH y HEIGHT por current_width y current_height en la función
        replacements = [
            # Reemplazos para dimensiones de pantalla
            (r'WIDTH // 2 - score_text\.get_width\(\) // 2', 'current_width // 2 - score_text.get_width() // 2'),
            (r'WIDTH // 2 - button_width // 2', 'current_width // 2 - button_width // 2'),
            (r'HEIGHT // 2 \+ 50', 'current_height // 2 + 50'),
            (r'WIDTH // 2, HEIGHT // 2 - 50', 'current_width // 2, current_height // 2 - 50'),
            (r'WIDTH // 2 - mode_text\.get_width\(\) // 2', 'current_width // 2 - mode_text.get_width() // 2'),
            (r'HEIGHT - 50', 'current_height - 50'),
            (r'HEIGHT - 30', 'current_height - 30'),
            (r'WIDTH // 2, HEIGHT // 2 - 50', 'current_width // 2, current_height // 2 - 50'),
            (r'WIDTH // 2, button_y \+ button_height // 2', 'current_width // 2, button_y + button_height // 2'),
            (r'pygame\.Surface\(\(WIDTH, HEIGHT\)', 'pygame.Surface((current_width, current_height)'),
            
            # Reemplazos para movimiento de IA
            (r'HEIGHT - ai_mallet\.radius', 'current_height - ai_mallet.radius'),
            (r'WIDTH - ai_mallet\.radius', 'current_width - ai_mallet.radius'),
            (r'WIDTH \* 0\.85', 'current_width * 0.85'),
            (r'WIDTH \* 0\.55', 'current_width * 0.55'),
            (r'WIDTH \* 0\.75', 'current_width * 0.75'),
            (r'puck\.position\[0\] > WIDTH // 2', 'puck.position[0] > current_width // 2'),
            
            # Reemplazos para posiciones de reset
            (r'WIDTH // 4, HEIGHT // 2', 'current_width // 4, current_height // 2'),
            (r'WIDTH \* 3 // 4, HEIGHT // 2', 'current_width * 3 // 4, current_height // 2'),
        ]
        
        # Aplicar reemplazos
        for pattern, replacement in replacements:
            function_content = re.sub(pattern, replacement, function_content)
        
        # Si hubo cambios, actualizar el archivo
        if function_content != original_function:
            new_content = content.replace(original_function, function_content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ Scaling fixes applied successfully!")
            print("Changed references to WIDTH and HEIGHT to use current_width and current_height")
            return True
        else:
            print("ℹ️ No changes needed - scaling already fixed")
            return False
    else:
        print("❌ Could not find main_with_config function")
        return False

if __name__ == "__main__":
    fix_scaling_in_main_improved()
