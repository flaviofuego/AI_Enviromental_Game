"""
Factory functions for common pop-up dialogs.
Extracted from PopUp.py to follow Single Responsibility — PopUp is only the
generic overlay component; this module knows *what* to show.
"""
import pygame
from .PopUp import PopUp


def create_help_popup(screen: pygame.Surface, screen_name: str) -> PopUp:
    """Create a contextual help pop-up for the given screen.

    Parameters
    ----------
    screen : pygame.Surface
    screen_name : str
        One of ``"home"``, ``"level_select"``, etc.
    """
    help_content = {
        "home": {
            "title": "Menu Principal - Ayuda",
            "content": [
                "CONTROLES PRINCIPALES:",
                "",
                "  Clic en JUGAR: Accede a la seleccion de niveles",
                "  Clic en HISTORIAL: Muestra informacion de GAIA",
                "  Clic en JUGADOR: Gestiona perfiles de jugador",
                "  Clic en CONFIGURACION: Ajusta opciones del juego",
                "",
                "PANEL DE PROGRESO:",
                "  Muestra tu avance planetario en tiempo real",
                "  Oceanos, ozono, aire, bosques y ciudades",
                "  Puntos GAIA acumulados por tus acciones",
                "",
                "MISION:",
                "  La Tierra esta en crisis climatica",
                "  5 agentes de EcoNull controlan el clima",
                "  Derrota a cada uno para restaurar el planeta",
                "",
                "CONSEJOS:",
                "  Crea o selecciona un perfil antes de jugar",
                "  Revisa el historial para entender la crisis",
                "  Tu progreso se guarda automaticamente",
            ],
        },
        "level_select": {
            "title": "Seleccion de Niveles - Ayuda",
            "content": [
                "SELECCION DE MISIONES:",
                "",
                "  Clic en cualquier nivel para ver detalles",
                "  Los niveles se desbloquean progresivamente",
                "  Completa un nivel para acceder al siguiente",
                "",
                "ESTADOS DE NIVEL:",
                "  DISPONIBLE (azul): Listo para jugar",
                "  COMPLETADO (verde): Ya derrotaste al enemigo",
                "  BLOQUEADO (gris): Necesitas completar anteriores",
                "",
                "CONTROLES:",
                "  Selecciona un nivel y presiona Jugar Nivel",
                "  Volver al Menu regresa a la pantalla principal",
                "  ESC para salir del juego en cualquier momento",
                "",
                "ESTRATEGIA:",
                "  Lee la descripcion de cada mision",
                "  Cada enemigo tiene mecanicas unicas",
                "  Puedes repetir niveles para mejorar puntuacion",
            ],
        },
    }

    content_data = help_content.get(screen_name, {
        "title": "Ayuda General",
        "content": [
            "Esta es la ventana de ayuda general.",
            "",
            "Usa los controles del mouse para navegar.",
            "Presiona ESC para cerrar ventanas.",
            "Tu progreso se guarda automaticamente.",
        ],
    })

    buttons = [
        {"text": "Entendido", "action": "close"},
        {"text": "Mas Info", "action": "more_info"},
    ]

    return PopUp(
        screen,
        content_data["title"],
        content_data["content"],
        buttons,
        "help",
    )


def create_settings_popup(screen: pygame.Surface, audio_manager) -> PopUp:
    """Create an audio-settings pop-up with volume sliders.

    Parameters
    ----------
    screen : pygame.Surface
    audio_manager
        The global ``AudioManager`` singleton.
    """
    music_volume = audio_manager.get_music_volume()
    sfx_volume = audio_manager.get_sfx_volume()
    music_enabled = audio_manager.is_music_enabled()
    sfx_enabled = audio_manager.is_sfx_enabled()

    music_btn_text = "Musica: Activada" if music_enabled else "Musica: Desactivada"
    sfx_btn_text = "Efectos: Activados" if sfx_enabled else "Efectos: Desactivados"

    settings_content = [
        "Ajusta las opciones del juego:",
        "",
        "Volumen de Musica:",
        "",
        "",
        "Volumen de Efectos:",
        "",
    ]

    buttons = [
        {"text": music_btn_text, "action": "toggle_music"},
        {"text": sfx_btn_text, "action": "toggle_sfx"},
        {"text": "Guardar", "action": "save_settings"},
        {"text": "Cancelar", "action": "cancel"},
    ]

    popup = PopUp(screen, "Configuracion", settings_content, buttons, "info")

    slider_width = int(popup.width * 0.7)
    slider_x = (popup.width - slider_width) // 2

    popup.add_slider(slider_x, 150, slider_width, 10, 0.0, 1.0, music_volume, "music_volume")
    popup.add_slider(slider_x, 230, slider_width, 10, 0.0, 1.0, sfx_volume, "sfx_volume")

    return popup
